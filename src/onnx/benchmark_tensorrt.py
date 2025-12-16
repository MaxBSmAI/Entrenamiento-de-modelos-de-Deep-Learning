from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import onnx


# ============================================================
# ENV / VERSION HELPERS
# ============================================================

def set_cuda_visible_devices_for_subprocess(device_id: Optional[int]) -> Dict[str, str]:
    env = os.environ.copy()
    if device_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(device_id)
    return env


def collect_env_info(device_id: Optional[int], env: Dict[str, str]) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "host": socket.gethostname(),
        "device_id": device_id,
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import tensorrt as trt  # type: ignore
        info["tensorrt_version"] = trt.__version__
    except Exception:
        info["tensorrt_version"] = None
    return info


# ============================================================
# TRTEXEC CAPABILITIES
# ============================================================

def trtexec_help(trtexec_path: str, env: Dict[str, str]) -> str:
    proc = subprocess.run(
        [trtexec_path, "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    return proc.stdout or ""


def supports_flag(help_text: str, flag: str) -> bool:
    return flag in help_text


def workspace_flag_args(help_text: str, workspace_mb: int) -> List[str]:
    # TRT 8.x: --workspace=<MiB>
    if ("--workspace=" in help_text) or ("--workspace" in help_text):
        return [f"--workspace={workspace_mb}"]

    # TRT 10.x: --memPoolSize=workspace:<MiB>M
    if ("--memPoolSize=" in help_text) or ("--memPoolSize" in help_text):
        return [f"--memPoolSize=workspace:{workspace_mb}M"]

    return []


# ============================================================
# INPUT INSPECTION (robusto)
# ============================================================

def _onnx_graph_first_input(onnx_path: Path) -> Optional[Tuple[str, Tuple[int, int, int]]]:
    """
    Intenta leer el primer input desde el grafo ONNX.
    Retorna None si el modelo no expone inputs en graph.input.
    """
    model = onnx.load(str(onnx_path))
    g = model.graph
    if not g.input:
        return None

    inp = g.input[0]
    name = inp.name
    t = inp.type.tensor_type
    shape = t.shape
    if len(shape.dim) < 4:
        return None

    def dim_to_int(dim, fallback: int) -> int:
        if dim.HasField("dim_value") and int(dim.dim_value) > 0:
            return int(dim.dim_value)
        return fallback

    c = dim_to_int(shape.dim[1], 3)
    h = dim_to_int(shape.dim[2], 640)
    w = dim_to_int(shape.dim[3], 640)
    return name, (c, h, w)


def _ort_first_input(onnx_path: Path, device_id: Optional[int]) -> Tuple[str, Tuple[int, int, int]]:
    """
    Fallback: usa ONNX Runtime para leer input name + shape.
    Esto funciona incluso si el grafo ONNX no tiene graph.input poblado.
    """
    import onnxruntime as ort

    sess_opt = ort.SessionOptions()
    providers = ["CPUExecutionProvider"]
    # Solo para lectura de inputs; CPU basta y es más estable

    env = set_cuda_visible_devices_for_subprocess(device_id)
    # ORT no respeta env dentro del proceso actual como subproceso, pero aquí solo leemos inputs.

    sess = ort.InferenceSession(str(onnx_path), sess_options=sess_opt, providers=providers)
    inputs = sess.get_inputs()
    if not inputs:
        raise ValueError(f"ORT no expone inputs para: {onnx_path}")

    inp = inputs[0]
    name = inp.name
    shape = inp.shape  # típicamente [None, 3, 640, 640] o similar
    if shape is None or len(shape) < 4:
        raise ValueError(f"Shape inválida via ORT para {onnx_path}: {shape}")

    # CHW desde dims 1..3; reemplaza None por fallback
    c = int(shape[1]) if isinstance(shape[1], int) and shape[1] > 0 else 3
    h = int(shape[2]) if isinstance(shape[2], int) and shape[2] > 0 else 640
    w = int(shape[3]) if isinstance(shape[3], int) and shape[3] > 0 else 640
    return name, (c, h, w)


def get_input_name_and_chw(
    onnx_path: Path,
    device_id: Optional[int],
    override_name: Optional[str],
    override_chw: Optional[str],
) -> Tuple[str, Tuple[int, int, int]]:
    """
    Prioridad:
      1) override por CLI
      2) ONNX graph.input
      3) ORT fallback
    """
    if override_name and override_chw:
        parts = [p.strip() for p in override_chw.split(",")]
        if len(parts) != 3:
            raise ValueError("--input-chw debe ser 'C,H,W' (ej: 3,640,640)")
        c, h, w = (int(parts[0]), int(parts[1]), int(parts[2]))
        return override_name, (c, h, w)

    got = _onnx_graph_first_input(onnx_path)
    if got is not None:
        return got

    return _ort_first_input(onnx_path, device_id)


# ============================================================
# TRTEXEC OUTPUT PARSING
# ============================================================

_LAT_PATTERNS = [
    re.compile(r"GPU latency:\s*([0-9.]+)\s*ms", re.IGNORECASE),
    re.compile(r"\bLatency:\s*([0-9.]+)\s*ms\b", re.IGNORECASE),
    re.compile(r"GPU\s*Compute\s*Time:\s*mean\s*=\s*([0-9.]+)\s*ms", re.IGNORECASE),
    re.compile(r"\bmean\s*=\s*([0-9.]+)\s*ms\b", re.IGNORECASE),
]
_THR_PATTERNS = [
    re.compile(r"Throughput:\s*([0-9.]+)\s*qps", re.IGNORECASE),
    re.compile(r"Throughput:\s*([0-9.]+)", re.IGNORECASE),
]


def parse_trtexec_stdout(stdout: str) -> Dict[str, Any]:
    latency_ms: Optional[float] = None
    qps_batches: Optional[float] = None

    for line in stdout.splitlines():
        s = line.strip()

        if latency_ms is None:
            for pat in _LAT_PATTERNS:
                m = pat.search(s)
                if m:
                    try:
                        latency_ms = float(m.group(1))
                        break
                    except Exception:
                        pass

        if qps_batches is None and "throughput" in s.lower():
            for pat in _THR_PATTERNS:
                m = pat.search(s)
                if m:
                    try:
                        qps_batches = float(m.group(1))
                        break
                    except Exception:
                        pass

        if latency_ms is not None and qps_batches is not None:
            break

    return {
        "latency_ms_per_batch": latency_ms,
        "qps_batches": qps_batches,
    }


# ============================================================
# RUN TRTEXEC
# ============================================================

def run_trtexec(
    onnx_path: Path,
    input_name: str,
    chw: Tuple[int, int, int],
    batch_size: int,
    fp16: bool,
    trtexec_path: str,
    workspace_mb: int,
    warmup: int,
    duration: int,
    use_spin_wait: bool,
    device_id: Optional[int],
    extra_args: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], str, List[str]]:
    env = set_cuda_visible_devices_for_subprocess(device_id)
    help_text = trtexec_help(trtexec_path, env)

    c, h, w = chw
    shapes_arg = f"{input_name}:{batch_size}x{c}x{h}x{w}"

    cmd: List[str] = [
        trtexec_path,
        f"--onnx={str(onnx_path)}",
        f"--shapes={shapes_arg}",
        f"--warmUp={warmup}",
        f"--duration={duration}",
    ]

    cmd += workspace_flag_args(help_text, workspace_mb)

    if use_spin_wait and supports_flag(help_text, "--useSpinWait"):
        cmd.append("--useSpinWait")

    precision = "fp32"
    if fp16 and supports_flag(help_text, "--fp16"):
        cmd.append("--fp16")
        precision = "fp16"

    if extra_args:
        cmd.extend(extra_args)

    print("[INFO] Ejecutando trtexec:")
    print("       " + " ".join(cmd))

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    t1 = time.time()

    stdout = proc.stdout or ""
    if proc.returncode != 0:
        print("------ SALIDA COMPLETA (trtexec) ------")
        print(stdout)
        raise RuntimeError(f"trtexec falló con código {proc.returncode}.")

    parsed = parse_trtexec_stdout(stdout)
    qps = parsed.get("qps_batches")
    throughput_samples = float(qps) * float(batch_size) if qps is not None else None

    metrics: Dict[str, Any] = {
        "precision": precision,
        "batch_size": batch_size,
        "elapsed_wall_time_s": float(t1 - t0),
        "latency_ms_per_batch": parsed.get("latency_ms_per_batch"),
        "qps_batches": qps,
        "throughput_samples_s": throughput_samples,
    }

    return metrics, stdout, cmd


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark TensorRT (trtexec) sobre ONNX (RTX 4080 / A100 compatible).")

    p.add_argument("--onnx-path", type=str, required=True)
    p.add_argument("--task", type=str, choices=["classification", "detection"], default="classification")

    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--trtexec-path", type=str, default="trtexec")

    p.add_argument("--workspace-mb", type=int, default=2048)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--duration", type=int, default=3)
    p.add_argument("--use-spin-wait", action="store_true")
    p.add_argument("--device-id", type=int, default=None)

    # NUEVO: overrides por si el ONNX no expone inputs en graph.input
    p.add_argument("--input-name", type=str, default=None, help="Override nombre del input (ej: input)")
    p.add_argument("--input-chw", type=str, default=None, help="Override CHW como 'C,H,W' (ej: 3,640,640)")

    p.add_argument("--save-json", action="store_true")
    p.add_argument("--json-path", type=str, default=None)
    p.add_argument("--save-raw", action="store_true")
    p.add_argument("--raw-path", type=str, default=None)

    p.add_argument("--extra-trtexec-args", type=str, default=None,
                   help='Args extra para trtexec como string, ej: "--useCudaGraph --noDataTransfers".')

    return p.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()
    onnx_path = Path(args.onnx_path).resolve()
    if not onnx_path.is_file():
        print(f"[ERROR] ONNX no encontrado: {onnx_path}")
        sys.exit(1)

    print("============================================================")
    print(f"[INFO] ONNX: {onnx_path}")
    print(f"[INFO] task: {args.task}")
    print(f"[INFO] batch_size: {args.batch_size}")
    print(f"[INFO] fp16: {args.fp16}")
    print(f"[INFO] workspace_mb: {args.workspace_mb}")
    print(f"[INFO] warmup/duration: {args.warmup}/{args.duration}")
    print(f"[INFO] device_id: {args.device_id}")
    print("============================================================")

    input_name, chw = get_input_name_and_chw(
        onnx_path=onnx_path,
        device_id=args.device_id,
        override_name=args.input_name,
        override_chw=args.input_chw,
    )
    print(f"[INFO] Input name: {input_name}")
    print(f"[INFO] Input CHW : {chw}")

    extra_args = args.extra_trtexec_args.strip().split() if args.extra_trtexec_args else None

    metrics, stdout, cmd = run_trtexec(
        onnx_path=onnx_path,
        input_name=input_name,
        chw=chw,
        batch_size=args.batch_size,
        fp16=args.fp16,
        trtexec_path=args.trtexec_path,
        workspace_mb=args.workspace_mb,
        warmup=args.warmup,
        duration=args.duration,
        use_spin_wait=args.use_spin_wait,
        device_id=args.device_id,
        extra_args=extra_args,
    )

    env = set_cuda_visible_devices_for_subprocess(args.device_id)
    result: Dict[str, Any] = {
        "backend": "tensorrt",
        "onnx_path": str(onnx_path),
        "task": args.task,
        "batch_size": args.batch_size,
        "precision": metrics.get("precision"),
        "latency_ms_per_batch": metrics.get("latency_ms_per_batch"),
        "qps_batches": metrics.get("qps_batches"),
        "throughput_samples_s": metrics.get("throughput_samples_s"),
        "benchmark_tensorrt": metrics,
        "trtexec_cmd": cmd,
        "env": collect_env_info(args.device_id, env),
        "note": "Métricas normalizadas: latencia por batch, qps (batches/s), throughput (samples/s).",
    }

    print("\n[RESULTADOS TensorRT]")
    for k, v in result.items():
        print(f"  {k}: {v}")

    if args.save_raw:
        if args.raw_path:
            raw_path = Path(args.raw_path)
        else:
            base = onnx_path.with_suffix("")
            raw_path = base.with_name(base.name + "_tensorrt_trtexec_stdout.txt")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(stdout, encoding="utf-8")
        print(f"\n[INFO] Stdout trtexec guardado en: {raw_path}")

    if args.save_json:
        if args.json_path:
            json_path = Path(args.json_path)
        else:
            base = onnx_path.with_suffix("")
            json_path = base.with_name(base.name + "_tensorrt_bench.json")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)
        print(f"\n[INFO] JSON guardado en: {json_path}")


if __name__ == "__main__":
    main()
