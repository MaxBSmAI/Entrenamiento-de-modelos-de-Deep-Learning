from __future__ import annotations

"""
run_onnx_bench_all.py

Runner "one-command" para benchmark de todos los ONNX del repo.

Flujo:
1) Busca todos los .onnx en models/onnx/**.onnx (excluye models/onnx/global)
2) Ejecuta:
   - benchmark_onnx_runtime.py (ORT)
   - benchmark_tensorrt.py (TRT / trtexec)
3) Ejecuta onnx_results.py para consolidar CSV + gráficos

Diseñado para:
- RTX 4080 (local)
- A100 (servidor multi-GPU)

Características:
- Selección determinista: --device-id fija CUDA_VISIBLE_DEVICES para subprocesos
- Presets por GPU + tarea con override por CLI
- --skip-existing para no recalcular si JSON ya existe
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, List


# ============================================================
# PATHS
# ============================================================

def project_root() -> Path:
    # src/onnx/run_onnx_bench_all.py -> parents[2] = repo root
    return Path(__file__).resolve().parents[2]


def python_executable() -> str:
    return sys.executable


def script_paths(root: Path) -> Dict[str, Path]:
    return {
        "ort": root / "src" / "onnx" / "benchmark_onnx_runtime.py",
        "trt": root / "src" / "onnx" / "benchmark_tensorrt.py",
        "results": root / "src" / "onnx" / "onnx_results.py",
    }


# ============================================================
# GPU DETECTION (best-effort)
# ============================================================

def detect_gpu_tag(device_id: Optional[int]) -> str:
    """
    Detecta tag de GPU de forma best-effort:
    - Usa torch si está disponible.
    - Respeta CUDA_VISIBLE_DEVICES si device_id se define.
    """
    if device_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    try:
        import torch  # local import para no forzar dependencia
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0).lower()
            if "a100" in name:
                return "A100"
            if "4080" in name:
                return "RTX_4080"
            if "4060" in name:
                return "RTX_4060"
            # fallback sanitizado
            return name.replace(" ", "_").replace("-", "_")
        return "CPU"
    except Exception:
        # fallback si torch no está disponible
        return "UNKNOWN"


# ============================================================
# PRESETS
# ============================================================

def default_presets(gpu_tag: str) -> Dict[str, Dict[str, int]]:
    """
    Presets recomendados por GPU para comparar con estabilidad.
    Puedes ajustar luego; los defaults son conservadores.
    """
    gpu_tag = (gpu_tag or "UNKNOWN").upper()

    # Por tarea, definimos batch/iters/warmup (ORT) y batch/duration/warmup (TRT)
    # Valores razonables:
    # - RTX 4080: batch moderado, iters medianos
    # - A100: batch más alto, igual iters (o más)
    if "A100" in gpu_tag:
        return {
            "classification": {"ort_batch": 32, "ort_warmup": 20, "ort_iters": 200,
                               "trt_batch": 64, "trt_warmup": 200, "trt_duration": 3, "trt_workspace": 4096},
            "detection": {"ort_batch": 4, "ort_warmup": 20, "ort_iters": 200,
                          "trt_batch": 8, "trt_warmup": 200, "trt_duration": 3, "trt_workspace": 4096},
        }

    # Default RTX 4080 / RTX 4060
    return {
        "classification": {"ort_batch": 16, "ort_warmup": 20, "ort_iters": 200,
                           "trt_batch": 32, "trt_warmup": 200, "trt_duration": 3, "trt_workspace": 2048},
        "detection": {"ort_batch": 2, "ort_warmup": 20, "ort_iters": 200,
                      "trt_batch": 4, "trt_warmup": 200, "trt_duration": 3, "trt_workspace": 2048},
    }


# ============================================================
# FILE DISCOVERY
# ============================================================

def infer_task_from_path(onnx_path: Path, onnx_root: Path) -> str:
    """
    Infiero task desde models/onnx/<task>/...; fallback "classification".
    """
    try:
        rel = onnx_path.parent.relative_to(onnx_root)
        if rel.parts:
            t = rel.parts[0].lower()
            if t in ("classification", "detection"):
                return t
    except Exception:
        pass
    return "classification"


def list_onnx_files(onnx_root: Path) -> List[Path]:
    """
    Lista todos los ONNX excepto los que estén en models/onnx/global.
    """
    files = []
    for p in onnx_root.rglob("*.onnx"):
        # excluir global
        if "global" in p.parts:
            continue
        files.append(p)
    return sorted(files)


def json_exists_for_onnx(onnx_path: Path, suffix: str) -> bool:
    """
    suffix esperado:
      - "_onnxruntime_benchmark.json"
      - "_tensorrt_bench.json"
    """
    base = onnx_path.with_suffix("")
    json_path = base.with_name(base.name + suffix)
    return json_path.is_file()


# ============================================================
# SUBPROCESS RUNNER
# ============================================================

def run_cmd(cmd: List[str], env: Dict[str, str]) -> None:
    print("[CMD] " + " ".join(cmd))
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Comando falló con código {proc.returncode}: {' '.join(cmd)}")


def build_env(device_id: Optional[int]) -> Dict[str, str]:
    env = os.environ.copy()
    if device_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(device_id)
    return env


# ============================================================
# MAIN
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ejecuta benchmarks ORT/TRT para todos los ONNX y consolida resultados.")

    p.add_argument("--onnx-root", type=str, default=None,
                   help="Directorio base ONNX. Default: <repo>/models/onnx")
    p.add_argument("--device-id", type=int, default=None,
                   help="GPU id para CUDA_VISIBLE_DEVICES (recomendado en A100 multi-GPU).")

    p.add_argument("--cpu-only", action="store_true",
                   help="Forzar ORT en CPU (no ejecuta CUDAExecutionProvider).")
    p.add_argument("--no-ort", action="store_true",
                   help="No ejecutar ONNX Runtime benchmark.")
    p.add_argument("--no-trt", action="store_true",
                   help="No ejecutar TensorRT benchmark (trtexec).")
    p.add_argument("--fp16", action="store_true",
                   help="TensorRT en FP16 (recomendado).")
    p.add_argument("--use-iobinding", action="store_true",
                   help="ORT usando IO binding cuando haya CUDA (recomendado).")

    p.add_argument("--img-size-cls", type=int, default=224,
                   help="Fallback img_size para clasificación (ORT).")
    p.add_argument("--img-size-det", type=int, default=640,
                   help="Fallback img_size para detección (ORT).")

    # Overrides manuales (si no se proveen, se usan presets)
    p.add_argument("--ort-batch-cls", type=int, default=None)
    p.add_argument("--ort-batch-det", type=int, default=None)
    p.add_argument("--ort-warmup", type=int, default=None)
    p.add_argument("--ort-iters", type=int, default=None)

    p.add_argument("--trt-batch-cls", type=int, default=None)
    p.add_argument("--trt-batch-det", type=int, default=None)
    p.add_argument("--trt-warmup", type=int, default=None)
    p.add_argument("--trt-duration", type=int, default=None)
    p.add_argument("--trt-workspace-mb", type=int, default=None)
    p.add_argument("--trtexec-path", type=str, default="trtexec")

    p.add_argument("--skip-existing", action="store_true",
                   help="No re-ejecutar si ya existe el JSON correspondiente.")
    p.add_argument("--run-results", action="store_true",
                   help="Ejecutar onnx_results.py al final (recomendado).")

    return p.parse_args()


def main() -> None:
    root = project_root()
    paths = script_paths(root)

    onnx_root = Path(args.onnx_root).resolve() if args.onnx_root else (root / "models" / "onnx")
    if not onnx_root.is_dir():
        raise FileNotFoundError(f"onnx_root no existe: {onnx_root}")

    env = build_env(args.device_id)
    gpu_tag = detect_gpu_tag(args.device_id)
    presets = default_presets(gpu_tag)

    print("============================================================")
    print(f"[INFO] Repo root   : {root}")
    print(f"[INFO] ONNX root   : {onnx_root}")
    print(f"[INFO] device_id   : {args.device_id}")
    print(f"[INFO] CUDA_VISIBLE: {env.get('CUDA_VISIBLE_DEVICES')}")
    print(f"[INFO] gpu_tag     : {gpu_tag}")
    print(f"[INFO] ORT script  : {paths['ort']}")
    print(f"[INFO] TRT script  : {paths['trt']}")
    print(f"[INFO] Results     : {paths['results']}")
    print("============================================================")

    onnx_files = list_onnx_files(onnx_root)
    if not onnx_files:
        print("[WARN] No se encontraron .onnx en models/onnx/")
        return

    print(f"[INFO] Encontrados {len(onnx_files)} modelos ONNX.")

    # Resolve overrides (fall back to presets per task)
    ort_warmup = args.ort_warmup
    ort_iters = args.ort_iters
    trt_warmup = args.trt_warmup
    trt_duration = args.trt_duration
    trt_workspace = args.trt_workspace_mb

    for onnx_path in onnx_files:
        task = infer_task_from_path(onnx_path, onnx_root)
        preset = presets.get(task, presets["classification"])

        # ORT params
        ort_batch = preset["ort_batch"] if task == "classification" else preset["ort_batch"]
        if task == "classification" and args.ort_batch_cls is not None:
            ort_batch = args.ort_batch_cls
        if task == "detection" and args.ort_batch_det is not None:
            ort_batch = args.ort_batch_det

        ort_w = ort_warmup if ort_warmup is not None else preset["ort_warmup"]
        ort_i = ort_iters if ort_iters is not None else preset["ort_iters"]

        img_size = args.img_size_cls if task == "classification" else args.img_size_det

        # TRT params
        trt_batch = preset["trt_batch"] if task == "classification" else preset["trt_batch"]
        if task == "classification" and args.trt_batch_cls is not None:
            trt_batch = args.trt_batch_cls
        if task == "detection" and args.trt_batch_det is not None:
            trt_batch = args.trt_batch_det

        trt_w = trt_warmup if trt_warmup is not None else preset["trt_warmup"]
        trt_d = trt_duration if trt_duration is not None else preset["trt_duration"]
        trt_ws = trt_workspace if trt_workspace is not None else preset["trt_workspace"]

        print("\n------------------------------------------------------------")
        print(f"[MODEL] {onnx_path.relative_to(root)}")
        print(f"[TASK ] {task}")
        print(f"[ORT  ] batch={ort_batch} warmup={ort_w} iters={ort_i} img_size={img_size} cpu_only={args.cpu_only}")
        print(f"[TRT  ] batch={trt_batch} warmup={trt_w} duration={trt_d} fp16={args.fp16} ws={trt_ws}")
        print("------------------------------------------------------------")

        # ORT benchmark
        if not args.no_ort:
            if args.skip_existing and json_exists_for_onnx(onnx_path, "_onnxruntime_benchmark.json"):
                print("[SKIP] ORT JSON ya existe.")
            else:
                cmd = [
                    python_executable(), str(paths["ort"]),
                    "--onnx-path", str(onnx_path),
                    "--batch-size", str(ort_batch),
                    "--img-size", str(img_size),
                    "--warmup", str(ort_w),
                    "--iters", str(ort_i),
                    "--save-json",
                ]
                if args.device_id is not None:
                    cmd += ["--device-id", str(args.device_id)]
                if args.cpu_only:
                    cmd += ["--cpu-only"]
                if args.use_iobinding:
                    cmd += ["--use-iobinding"]

                run_cmd(cmd, env)

        # TRT benchmark
        if not args.no_trt:
            if args.skip_existing and json_exists_for_onnx(onnx_path, "_tensorrt_bench.json"):
                print("[SKIP] TRT JSON ya existe.")
            else:
                cmd = [
                    python_executable(), str(paths["trt"]),
                    "--onnx-path", str(onnx_path),
                    "--task", task,
                    "--batch-size", str(trt_batch),
                    "--workspace-mb", str(trt_ws),
                    "--warmup", str(trt_w),
                    "--duration", str(trt_d),
                    "--trtexec-path", args.trtexec_path,
                    "--save-json",
                    "--save-raw",
                ]
                if args.fp16:
                    cmd += ["--fp16"]
                if args.device_id is not None:
                    cmd += ["--device-id", str(args.device_id)]

                run_cmd(cmd, env)

    # Consolidación final
    if args.run_results:
        print("\n============================================================")
        print("[INFO] Ejecutando onnx_results.py ...")
        print("============================================================")
        cmd = [python_executable(), str(paths["results"])]
        run_cmd(cmd, env)

    print("\n[DONE] run_onnx_bench_all.py finalizó correctamente.")


if __name__ == "__main__":
    args = parse_args()
    main()
