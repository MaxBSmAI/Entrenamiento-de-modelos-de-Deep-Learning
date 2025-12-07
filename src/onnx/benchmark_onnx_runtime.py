# src/onnx/benchmark_onnx_runtime.py

from __future__ import annotations

"""
Benchmark con ONNX Runtime para modelos exportados:

Soporta modelos ONNX generados por export_to_onnx.py:

    - Clasificación:
        resnet50_miniimagenet__<DEVICE>.onnx
        vit_b16_miniimagenet__<DEVICE>.onnx

    - Detección:
        fasterrcnn_coco__<DEVICE>.onnx
        retinanet_coco__<DEVICE>.onnx

Mide (con datos sintéticos pero realistas):

    - Latencia media por batch (ms)
    - FPS (samples/s)
    - Throughput (samples/s)
    - Dispositivo/proveedor ONNX Runtime utilizado

Guarda un JSON junto al modelo ONNX con sufijo:
    *_onnxruntime_benchmark.json
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import onnxruntime as ort


# ============================================================
# HELPERS
# ============================================================

def available_providers() -> Tuple[str, ...]:
    """
    Devuelve la lista de providers disponibles en la instalación de onnxruntime.
    """
    return tuple(ort.get_available_providers())


def choose_providers(prefer_gpu: bool) -> Tuple[str, ...]:
    """
    Elige providers para la sesión ONNX Runtime.

    Si prefer_gpu=True y está disponible CUDAExecutionProvider,
    se usa ['CUDAExecutionProvider', 'CPUExecutionProvider'].
    En caso contrario, solo CPU.
    """
    providers = available_providers()
    if prefer_gpu and "CUDAExecutionProvider" in providers:
        return ("CUDAExecutionProvider", "CPUExecutionProvider")
    return ("CPUExecutionProvider",)


def make_dummy_input(
    task: str,
    batch_size: int,
    img_size: int,
) -> Dict[str, np.ndarray]:
    """
    Construye un diccionario de inputs sintéticos acorde al tipo de modelo ONNX.

    Para export_to_onnx.py definimos:

        Clasificación:
            input  -> [B, 3, H, W]  (float32)

        Detección:
            images -> [B, 3, H, W]  (float32)

    """
    shape = (batch_size, 3, img_size, img_size)
    data = np.random.randn(*shape).astype(np.float32)

    if task == "classification":
        return {"input": data}
    elif task == "detection":
        return {"images": data}
    else:
        raise ValueError(f"Tarea no soportada: {task}")


def summarize_results(
    total_time: float,
    total_samples: int,
    num_iters: int,
    task: str,
    batch_size: int,
    providers: Tuple[str, ...],
    model_path: Path,
) -> Dict[str, Any]:
    """
    Construye el diccionario resumen con métricas de rendimiento.
    """
    if num_iters == 0 or total_time <= 0.0:
        mean_latency_ms = None
        fps = None
        throughput = None
    else:
        mean_latency_ms = (total_time / num_iters) * 1000.0
        fps = total_samples / total_time
        throughput = fps  # para clasificación y detección tomamos fps ~ samples/s

    result = {
        "onnx_model_path": str(model_path),
        "task": task,
        "batch_size": batch_size,
        "num_iterations": num_iters,
        "mean_latency_ms": mean_latency_ms,
        "fps": fps,
        "throughput": throughput,
        "providers": list(providers),
        "note": (
            "Benchmark con datos sintéticos (random) usando ONNX Runtime. "
            "Comparar con métricas de PyTorch para analizar interoperabilidad."
        ),
    }
    return result


# ============================================================
# BENCHMARK
# ============================================================

def benchmark_onnx_model(
    model_path: Path,
    task: str,
    batch_size: int = 8,
    img_size: int = 224,
    warmup: int = 5,
    iters: int = 50,
    prefer_gpu: bool = True,
) -> Dict[str, Any]:
    """
    Ejecuta un benchmark de inferencia sobre un modelo ONNX con datos sintéticos.

    Parámetros
    ----------
    model_path : Path
        Ruta al archivo .onnx.
    task : str
        'classification' o 'detection'.
    batch_size : int
        Tamaño de batch para el dummy input.
    img_size : int
        Tamaño de imagen (lado); 224 para clasificación, 640 para detección.
    warmup : int
        Número de iteraciones de calentamiento (no se miden).
    iters : int
        Número de iteraciones medidas.
    prefer_gpu : bool
        Si True, intenta usar CUDAExecutionProvider si está disponible.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    providers = choose_providers(prefer_gpu)
    print(f"[INFO] Providers ONNX Runtime: {providers}")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    print(f"[INFO] Creando sesión ONNX Runtime para: {model_path}")
    session = ort.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=list(providers),
    )

    # Verificamos nombres de entradas y salidas
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]

    print(f"[INFO] Entradas del modelo ONNX : {input_names}")
    print(f"[INFO] Salidas del modelo ONNX  : {output_names}")

    # Determinar img_size por defecto según tarea si el usuario no lo cambió
    if img_size is None:
        img_size = 224 if task == "classification" else 640

    dummy_input = make_dummy_input(task, batch_size, img_size)

    # --------------------------------------------------------
    # WARMUP
    # --------------------------------------------------------
    print(f"[INFO] Warmup: {warmup} iteraciones (no medidas)")
    for _ in range(warmup):
        _ = session.run(output_names, dummy_input)

    # --------------------------------------------------------
    # MEDICIÓN
    # --------------------------------------------------------
    print(f"[INFO] Benchmark: {iters} iteraciones medidas")
    total_time = 0.0
    total_samples = 0

    for i in range(iters):
        start = time.perf_counter()
        _ = session.run(output_names, dummy_input)
        end = time.perf_counter()

        elapsed = end - start
        total_time += elapsed
        total_samples += batch_size

        if (i + 1) % max(1, iters // 10) == 0:
            print(
                f"  Iter {i + 1:>3}/{iters} "
                f"- tiempo batch: {elapsed * 1000.0:7.3f} ms"
            )

    results = summarize_results(
        total_time=total_time,
        total_samples=total_samples,
        num_iters=iters,
        task=task,
        batch_size=batch_size,
        providers=providers,
        model_path=model_path,
    )

    return results


# ============================================================
# PARSER
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark de modelos ONNX usando ONNX Runtime."
    )

    parser.add_argument(
        "--onnx-path",
        type=str,
        required=True,
        help="Ruta al archivo .onnx exportado.",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["classification", "detection"],
        help="Tarea del modelo: classification o detection.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Tamaño de batch para el dummy input.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Tamaño de imagen (lado). Por defecto 224 (clasificación) o 640 (detección).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Iteraciones de warmup (no medidas).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=50,
        help="Iteraciones medidas para el benchmark.",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Forzar uso de CPUExecutionProvider (sin GPU).",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Guardar resultados en un JSON junto al modelo ONNX.",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()

    model_path = Path(args.onnx_path).resolve()
    prefer_gpu = not args.cpu_only

    # Determinar tamaño de imagen por defecto según tarea
    if args.img_size is None:
        img_size = 224 if args.task == "classification" else 640
    else:
        img_size = args.img_size

    print("============================================================")
    print(f"[INFO] Modelo ONNX : {model_path}")
    print(f"[INFO] Tarea       : {args.task}")
    print(f"[INFO] Batch size  : {args.batch_size}")
    print(f"[INFO] Img size    : {img_size} x {img_size}")
    print(f"[INFO] Warmup/iters: {args.warmup}/{args.iters}")
    print(f"[INFO] Prefer GPU  : {prefer_gpu}")
    print("============================================================")

    results = benchmark_onnx_model(
        model_path=model_path,
        task=args.task,
        batch_size=args.batch_size,
        img_size=img_size,
        warmup=args.warmup,
        iters=args.iters,
        prefer_gpu=prefer_gpu,
    )

    print("\n[RESULTADOS ONNX Runtime]")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")

    if args.save_json:
        json_path = model_path.with_suffix("").with_name(
            model_path.stem + "_onnxruntime_benchmark.json"
        )
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\n[INFO] Resultados guardados en: {json_path}")


if __name__ == "__main__":
    main()
