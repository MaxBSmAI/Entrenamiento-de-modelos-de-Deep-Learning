from __future__ import annotations

"""
benchmark_onnx_runtime.py

Benchmark de inferencia para modelos ONNX usando ONNX Runtime (ORT),
portable entre:
- RTX 4080 (local)
- A100 (servidor)

Características:
- Selección determinista de GPU (--device-id) usando CUDA_VISIBLE_DEVICES
- No hardcodea nombres de inputs (detecta desde el modelo ONNX)
- Construye inputs sintéticos con shapes y dtypes del modelo
- Métricas normalizadas:
    - latency_ms_per_batch
    - qps_batches
    - throughput_samples_s (= qps_batches * batch_size)
- CPU-only soportado
- Opcional: IO Binding para reducir overhead en GPU (--use-iobinding)
- Guardado JSON:
    - Si entregas --out-dir, guarda allí (recomendado)
    - Si NO entregas --out-dir, guarda junto al ONNX (compatibilidad)
"""

import argparse
import json
import os
import socket
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnxruntime as ort


# ============================================================
# DEVICE / ENV HELPERS
# ============================================================

def set_cuda_visible_devices(device_id: Optional[int]) -> None:
    """
    Fija CUDA_VISIBLE_DEVICES antes de inicializar providers CUDA.
    En multi-GPU (A100) esto evita benchmarks en la GPU equivocada.
    """
    if device_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)


def available_providers() -> List[str]:
    return list(ort.get_available_providers())


def choose_providers(prefer_gpu: bool, provider_options: Optional[Dict[str, Any]] = None) -> Tuple[List[Any], str]:
    """
    Elige providers para ORT.
    Retorna:
        providers (lista para InferenceSession)
        selected_backend ("cuda" o "cpu")
    """
    provs = available_providers()

    if prefer_gpu and "CUDAExecutionProvider" in provs:
        if provider_options is None:
            provider_options = {}
        providers: List[Any] = [
            ("CUDAExecutionProvider", provider_options),
            "CPUExecutionProvider",
        ]
        return providers, "cuda"

    return ["CPUExecutionProvider"], "cpu"


def collect_env_info(
    selected_backend: str,
    device_id: Optional[int],
    providers: Sequence[Any],
) -> Dict[str, Any]:
    """
    Metadata mínima para trazabilidad y comparabilidad (RTX 4080 vs A100).
    """
    return {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "host": socket.gethostname(),
        "backend_selected": selected_backend,
        "device_id": device_id,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "onnxruntime_version": getattr(ort, "__version__", "unknown"),
        "providers": [p[0] if isinstance(p, tuple) else p for p in providers],
        "providers_raw": [
            {"provider": p[0], "options": p[1]} if isinstance(p, tuple) else {"provider": p, "options": {}}
            for p in providers
        ],
        "gpu_name": None,  # ORT no expone el nombre GPU de forma estándar
    }


# ============================================================
# ONNX INPUT HANDLING
# ============================================================

def ort_type_to_numpy(dtype_str: str) -> np.dtype:
    """
    Mapea tipos ORT a numpy dtype.
    dtype_str ej: 'tensor(float)', 'tensor(float16)', 'tensor(int64)', etc.
    """
    ds = dtype_str.strip().lower()

    if "float16" in ds:
        return np.float16
    if "float" in ds:
        return np.float32
    if "double" in ds:
        return np.float64
    if "int64" in ds:
        return np.int64
    if "int32" in ds:
        return np.int32
    if "int16" in ds:
        return np.int16
    if "int8" in ds:
        return np.int8
    if "uint8" in ds:
        return np.uint8
    if "bool" in ds:
        return np.bool_

    return np.float32


def resolve_dim(dim: Any, fallback: int) -> int:
    """
    ORT puede devolver dims como int, None o string simbólico.
    """
    if isinstance(dim, int) and dim > 0:
        return dim
    return fallback


def build_dummy_inputs(
    session: ort.InferenceSession,
    batch_size: int,
    img_size: int,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Construye inputs sintéticos para TODOS los inputs del modelo.

    Estrategia:
    - Si input es 4D (N,C,H,W): se asume imagen.
      - N := batch_size
      - C := 3 si no está definido
      - H/W := img_size si no están definidos
    - Si input es 1D/2D/...: se rellena con valores compatibles,
      usando fallback 1 para dims simbólicas.
    """
    rng = np.random.default_rng(seed)
    feed: Dict[str, np.ndarray] = {}

    for inp in session.get_inputs():
        name = inp.name
        dtype = ort_type_to_numpy(inp.type)
        shape = list(inp.shape)  # puede contener None / str

        if len(shape) == 4:
            n = batch_size
            c = resolve_dim(shape[1], 3)
            h = resolve_dim(shape[2], img_size)
            w = resolve_dim(shape[3], img_size)
            arr = rng.standard_normal((n, c, h, w)).astype(dtype)
            feed[name] = arr
        else:
            resolved: List[int] = []
            for i, d in enumerate(shape):
                if i == 0:
                    resolved.append(batch_size if (d is None or isinstance(d, str)) else int(d))
                else:
                    resolved.append(resolve_dim(d, 1))
            if len(resolved) == 0:
                resolved = [batch_size]
            arr = rng.standard_normal(tuple(resolved)).astype(dtype)
            feed[name] = arr

    return feed


# ============================================================
# METRICS
# ============================================================

def summarize_metrics(total_time_s: float, iters: int, batch_size: int) -> Dict[str, Any]:
    if iters <= 0 or total_time_s <= 0.0:
        return {
            "latency_ms_per_batch": None,
            "qps_batches": None,
            "throughput_samples_s": None,
        }

    qps_batches = iters / total_time_s
    latency_ms = (total_time_s / iters) * 1000.0
    throughput_samples = qps_batches * batch_size

    return {
        "latency_ms_per_batch": float(latency_ms),
        "qps_batches": float(qps_batches),
        "throughput_samples_s": float(throughput_samples),
    }


# ============================================================
# IO BINDING (GPU)
# ============================================================

def run_with_iobinding(
    session: ort.InferenceSession,
    feed: Dict[str, np.ndarray],
    warmup: int,
    iters: int,
) -> Dict[str, Any]:
    """
    Ejecuta usando IO binding (útil en CUDA). ORT manejará copies
    y bindings internamente.

    Nota: Para algunos modelos/dtypes, ORT puede rechazar ciertos bindings;
    si ocurre, usar modo normal.
    """
    io_binding = session.io_binding()

    # Bind inputs (from numpy) - ORT copiará a device si corresponde
    for name, arr in feed.items():
        io_binding.bind_cpu_input(name, arr)

    # Bind outputs (device) - se obtienen luego con copy_outputs_to_cpu()
    for out in session.get_outputs():
        io_binding.bind_output(out.name)

    # Warmup
    for _ in range(warmup):
        session.run_with_iobinding(io_binding)

    # Timed
    t0 = time.perf_counter()
    for _ in range(iters):
        session.run_with_iobinding(io_binding)
    t1 = time.perf_counter()

    # Opcional: materializar outputs (para forzar sincronización completa)
    _ = io_binding.copy_outputs_to_cpu()

    return {
        "total_time_s": float(t1 - t0),
        "warmup": warmup,
        "iters": iters,
        "mode": "iobinding",
    }


def run_normal(
    session: ort.InferenceSession,
    feed: Dict[str, np.ndarray],
    warmup: int,
    iters: int,
) -> Dict[str, Any]:
    # Warmup
    for _ in range(warmup):
        session.run(None, feed)

    # Timed
    t0 = time.perf_counter()
    for _ in range(iters):
        session.run(None, feed)
    t1 = time.perf_counter()

    return {
        "total_time_s": float(t1 - t0),
        "warmup": warmup,
        "iters": iters,
        "mode": "normal",
    }


# ============================================================
# BENCHMARK CORE
# ============================================================

def benchmark_onnx(
    onnx_path: Path,
    task: str,
    batch_size: int,
    img_size: int,
    warmup: int,
    iters: int,
    prefer_gpu: bool,
    device_id: Optional[int],
    use_iobinding: bool,
    seed: int,
) -> Dict[str, Any]:
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX no encontrado: {onnx_path}")

    # Selección determinista (importante en A100 multi-GPU)
    set_cuda_visible_devices(device_id)

    # Provider options razonables (estables)
    cuda_provider_options = {
        "cudnn_conv_algo_search": "EXHAUSTIVE",
    }

    providers, selected_backend = choose_providers(
        prefer_gpu=prefer_gpu,
        provider_options=cuda_provider_options,
    )

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=sess_options,
        providers=providers,
    )

    feed = build_dummy_inputs(session, batch_size=batch_size, img_size=img_size, seed=seed)

    # Ejecutar benchmark
    if use_iobinding and selected_backend == "cuda":
        try:
            bench_info = run_with_iobinding(session, feed, warmup=warmup, iters=iters)
        except Exception as e:
            # Fallback seguro (no romper pipeline)
            bench_info = run_normal(session, feed, warmup=warmup, iters=iters)
            bench_info["iobinding_fallback_error"] = str(e)
    else:
        bench_info = run_normal(session, feed, warmup=warmup, iters=iters)

    metrics = summarize_metrics(
        total_time_s=bench_info["total_time_s"],
        iters=iters,
        batch_size=batch_size,
    )

    env = collect_env_info(selected_backend=selected_backend, device_id=device_id, providers=providers)

    # Inputs/outputs visibles para trazabilidad
    io_info = {
        "inputs": [
            {"name": i.name, "type": i.type, "shape": i.shape}
            for i in session.get_inputs()
        ],
        "outputs": [
            {"name": o.name, "type": o.type, "shape": o.shape}
            for o in session.get_outputs()
        ],
    }

    return {
        "onnx_model_path": str(onnx_path),
        "task": task,
        "batch_size": batch_size,
        "img_size": img_size,
        "warmup": warmup,
        "iters": iters,
        "use_iobinding": bool(use_iobinding),
        "env": env,
        "io": io_info,
        "benchmark": bench_info,
        **metrics,
    }


# ============================================================
# SAVE JSON
# ============================================================

def resolve_output_json_path(onnx_path: Path, out_dir: Optional[str]) -> Path:
    """
    - Si out_dir está definido: guarda en ese directorio con nombre basado en stem.
    - Si out_dir es None: guarda junto al ONNX (compatibilidad).
    """
    if out_dir is not None:
        od = Path(out_dir)
        od.mkdir(parents=True, exist_ok=True)
        return od / f"{onnx_path.stem}_onnxruntime_benchmark.json"

    # compatibilidad: junto al ONNX
    return onnx_path.with_suffix("").with_name(onnx_path.stem + "_onnxruntime_benchmark.json")


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"[OK] Resultados guardados en: {path}")


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Benchmark ONNX Runtime")

    p.add_argument("--onnx-path", type=str, required=True, help="Ruta al modelo .onnx")
    p.add_argument("--task", type=str, required=True, choices=["classification", "detection"])

    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--img-size", type=int, default=224)

    p.add_argument("--warmup", type=int, default=50, help="Warmup iterations (no temporizadas)")
    p.add_argument("--iters", type=int, default=500, help="Iterations temporizadas")

    p.add_argument("--device-id", type=int, default=None, help="GPU id físico (CUDA_VISIBLE_DEVICES)")
    p.add_argument("--cpu-only", action="store_true", help="Forzar CPU-only")

    p.add_argument("--use-iobinding", action="store_true", help="Usar IO Binding en CUDA")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--save-json", action="store_true", help="Guardar JSON del benchmark")
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directorio donde guardar el JSON. Si no se entrega, se guarda junto al ONNX.",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    onnx_path = Path(args.onnx_path).resolve()
    prefer_gpu = not args.cpu_only

    results = benchmark_onnx(
        onnx_path=onnx_path,
        task=args.task,
        batch_size=args.batch_size,
        img_size=args.img_size,
        warmup=args.warmup,
        iters=args.iters,
        prefer_gpu=prefer_gpu,
        device_id=args.device_id,
        use_iobinding=args.use_iobinding,
        seed=args.seed,
    )

    # Print resumen mínimo
    print("\n[RESULTADOS ONNX Runtime]")
    print(f"  latency_ms_per_batch: {results.get('latency_ms_per_batch')}")
    print(f"  qps_batches         : {results.get('qps_batches')}")
    print(f"  throughput_samples_s: {results.get('throughput_samples_s')}")
    print(f"  backend_selected    : {results.get('env', {}).get('backend_selected')}")
    print(f"  providers           : {results.get('env', {}).get('providers')}")

    if args.save_json:
        out_path = resolve_output_json_path(onnx_path, args.out_dir)
        save_json(results, out_path)


if __name__ == "__main__":
    main()
