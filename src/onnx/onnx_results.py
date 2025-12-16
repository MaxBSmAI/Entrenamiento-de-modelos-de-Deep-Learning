from __future__ import annotations

"""
onnx_results.py

Integra resultados de:
- PyTorch (result/<task>/<model_name>/<device_tag>/<model_name>_metrics.json)
- ONNX Runtime (models/onnx/**/<onnx_stem>_onnxruntime_benchmark.json)
- TensorRT (models/onnx/**/<onnx_stem>_tensorrt_bench.json)

Objetivos:
- Normalizar métricas comparables entre RTX 4080 (local) y A100 (server).
- Generar:
  - CSV global: models/onnx/global/onnx_backends_summary.csv
  - Gráficos por (task, model_name, device_tag):
      - Latencia (ms/batch)
      - Throughput (samples/s)

Métricas normalizadas (prioridad):
- latency_ms_per_batch
- qps_batches
- throughput_samples_s

Retrocompatibilidad:
- Si se detectan campos antiguos:
  - mean_latency_ms -> latency_ms_per_batch
  - fps            -> throughput_samples_s (aprox, si representa samples/s)
  - throughput      -> throughput_samples_s (aprox)
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# JSON HELPERS
# ============================================================

def safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] No se pudo leer JSON {path}: {e}")
        return None


# ============================================================
# ONNX NAME PARSING
# ============================================================

def parse_onnx_name(onnx_path: Path) -> Tuple[str, str]:
    """
    ONNX esperado:
      <model_name>__<device_tag>.onnx
    """
    stem = onnx_path.stem
    if "." in stem:
        stem = stem.split(".", 1)[0]
    if "__" in stem:
        model_name, device_tag = stem.split("__", 1)
    else:
        model_name, device_tag = stem, "UNKNOWN"
    return model_name, device_tag


# ============================================================
# PATH FINDERS
# ============================================================

def find_pytorch_metrics(project_root: Path, task: str, model_name: str, device_tag: str) -> Optional[Dict[str, Any]]:
    base = project_root / "result" / task / model_name
    cand1 = base / device_tag / f"{model_name}_metrics.json"
    cand2 = base / f"{model_name}_metrics.json"

    return safe_load_json(cand1) or safe_load_json(cand2)


def find_onnxruntime_bench(onnx_path: Path) -> Optional[Dict[str, Any]]:
    json_path = onnx_path.with_suffix("")
    json_path = json_path.with_name(json_path.name + "_onnxruntime_benchmark.json")
    return safe_load_json(json_path)


def find_tensorrt_bench(onnx_path: Path) -> Optional[Dict[str, Any]]:
    json_path = onnx_path.with_suffix("")
    json_path = json_path.with_name(json_path.name + "_tensorrt_bench.json")
    return safe_load_json(json_path)


# ============================================================
# NORMALIZATION HELPERS
# ============================================================

def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def normalize_backend_metrics(data: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    Devuelve siempre:
      - latency_ms_per_batch
      - qps_batches
      - throughput_samples_s

    Priorizando el esquema nuevo, con fallback a esquema antiguo.
    """
    # Nuevo esquema (preferido)
    lat = _to_float(data.get("latency_ms_per_batch"))
    qps = _to_float(data.get("qps_batches"))
    thr = _to_float(data.get("throughput_samples_s"))

    if lat is not None or qps is not None or thr is not None:
        return {
            "latency_ms_per_batch": lat,
            "qps_batches": qps,
            "throughput_samples_s": thr,
        }

    # Esquema antiguo (fallback)
    # ORT antiguo: mean_latency_ms, fps, throughput
    lat_old = _to_float(data.get("mean_latency_ms"))
    fps_old = _to_float(data.get("fps"))
    thr_old = _to_float(data.get("throughput"))

    # Interpretación:
    # - lat_old suele ser ms/batch
    # - fps_old y thr_old en tu script antiguo eran "samples/s" (aprox)
    #   (no batches/s).
    # No podemos reconstruir qps_batches sin batch_size; lo dejamos None.
    throughput_samples = fps_old if fps_old is not None else thr_old

    return {
        "latency_ms_per_batch": lat_old,
        "qps_batches": None,
        "throughput_samples_s": throughput_samples,
    }


def extract_env_fields(env: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    env = env or {}
    return {
        "host": env.get("host"),
        "timestamp_utc": env.get("timestamp_utc"),
        "backend_selected": env.get("backend_selected"),
        "gpu_name": env.get("gpu_name"),
        "device_id": env.get("device_id"),
        "cuda_visible_devices": env.get("cuda_visible_devices"),
        "onnxruntime_version": env.get("onnxruntime_version"),
        "tensorrt_version": env.get("tensorrt_version"),
        "providers": env.get("providers"),
    }


# ============================================================
# ROW BUILDERS
# ============================================================

def row_from_pytorch(pytorch_json: Dict[str, Any]) -> Dict[str, Any]:
    model_name = pytorch_json.get("model_name", "unknown_model")
    task = pytorch_json.get("task", "unknown_task")
    dataset = pytorch_json.get("dataset", None)
    num_classes = pytorch_json.get("num_classes", None)
    device_name = pytorch_json.get("device_name", None)
    device_tag = pytorch_json.get("device_tag", "UNKNOWN")

    bench = pytorch_json.get("benchmark", {}) or {}

    # Tu baseline PyTorch típicamente usa:
    # mean_latency_ms, fps, throughput
    # Lo normalizamos:
    lat = _to_float(bench.get("latency_ms_per_batch")) or _to_float(bench.get("mean_latency_ms"))
    qps = _to_float(bench.get("qps_batches"))  # usualmente no existe en PyTorch
    thr = _to_float(bench.get("throughput_samples_s")) or _to_float(bench.get("throughput")) or _to_float(bench.get("fps"))

    batch_size = bench.get("batch_size", None)
    precision = bench.get("precision", "fp32")

    return {
        "task": task,
        "model_name": model_name,
        "backend": "pytorch",
        "dataset": dataset,
        "num_classes": num_classes,
        "device_tag": device_tag,
        "device_name": device_name,
        "batch_size": batch_size,
        "precision": precision,
        "latency_ms_per_batch": lat,
        "qps_batches": qps,
        "throughput_samples_s": thr,
        "vram_used_mb": bench.get("vram_used_mb"),
        "host": None,
        "timestamp_utc": None,
        "gpu_name": None,
        "device_id": None,
        "cuda_visible_devices": None,
        "onnxruntime_version": None,
        "tensorrt_version": None,
        "providers": None,
        "extra_info": "",
        "source_json": str(pytorch_json.get("artifacts", {}).get("best_model_path", "")),
    }


def row_from_onnxruntime(onnx_path: Path, model_name: str, device_tag: str, task_fallback: str, ort_json: Dict[str, Any]) -> Dict[str, Any]:
    task = ort_json.get("task", task_fallback)
    batch_size = ort_json.get("batch_size", None)
    precision = ort_json.get("precision", "fp32")

    m = normalize_backend_metrics(ort_json)
    env_fields = extract_env_fields(ort_json.get("env"))

    extra = ""
    # providers puede estar en env o en campo antiguo "providers"
    providers = env_fields.get("providers") or ort_json.get("providers")
    if providers:
        extra = f"providers={providers}"

    return {
        "task": task,
        "model_name": model_name,
        "backend": "onnxruntime",
        "dataset": None,
        "num_classes": None,
        "device_tag": device_tag,
        "device_name": None,
        "batch_size": batch_size,
        "precision": precision,
        "latency_ms_per_batch": m["latency_ms_per_batch"],
        "qps_batches": m["qps_batches"],
        "throughput_samples_s": m["throughput_samples_s"],
        "vram_used_mb": None,
        "host": env_fields.get("host"),
        "timestamp_utc": env_fields.get("timestamp_utc"),
        "gpu_name": env_fields.get("gpu_name"),
        "device_id": env_fields.get("device_id"),
        "cuda_visible_devices": env_fields.get("cuda_visible_devices"),
        "onnxruntime_version": env_fields.get("onnxruntime_version"),
        "tensorrt_version": None,
        "providers": providers,
        "extra_info": extra,
        "source_json": str(onnx_path.with_suffix("").with_name(onnx_path.with_suffix("").name + "_onnxruntime_benchmark.json")),
    }


def row_from_tensorrt(onnx_path: Path, model_name: str, device_tag: str, task_fallback: str, trt_json: Dict[str, Any]) -> Dict[str, Any]:
    task = trt_json.get("task", task_fallback)
    batch_size = trt_json.get("batch_size", None)
    precision = trt_json.get("precision", trt_json.get("benchmark_tensorrt", {}).get("precision", "fp16/fp32?"))

    m = normalize_backend_metrics(trt_json)
    env_fields = extract_env_fields(trt_json.get("env"))

    extra = ""
    cmd = trt_json.get("trtexec_cmd")
    if cmd:
        extra = "trtexec_cmd=present"

    return {
        "task": task,
        "model_name": model_name,
        "backend": "tensorrt",
        "dataset": None,
        "num_classes": None,
        "device_tag": device_tag,
        "device_name": None,
        "batch_size": batch_size,
        "precision": precision,
        "latency_ms_per_batch": m["latency_ms_per_batch"],
        "qps_batches": m["qps_batches"],
        "throughput_samples_s": m["throughput_samples_s"],
        "vram_used_mb": None,
        "host": env_fields.get("host"),
        "timestamp_utc": env_fields.get("timestamp_utc"),
        "gpu_name": env_fields.get("gpu_name"),
        "device_id": env_fields.get("device_id"),
        "cuda_visible_devices": env_fields.get("cuda_visible_devices"),
        "onnxruntime_version": None,
        "tensorrt_version": env_fields.get("tensorrt_version"),
        "providers": None,
        "extra_info": extra,
        "source_json": str(onnx_path.with_suffix("").with_name(onnx_path.with_suffix("").name + "_tensorrt_bench.json")),
    }


# ============================================================
# PLOTTING
# ============================================================

def plot_backend_comparison(
    rows: List[Dict[str, Any]],
    out_path_latency: Path,
    out_path_throughput: Path,
    title_prefix: str,
) -> None:
    """
    Barras:
      - Latencia ms/batch
      - Throughput samples/s
    """
    backends = [r["backend"] for r in rows]

    lat = [
        float(r["latency_ms_per_batch"]) if r.get("latency_ms_per_batch") is not None else np.nan
        for r in rows
    ]
    thr = [
        float(r["throughput_samples_s"]) if r.get("throughput_samples_s") is not None else np.nan
        for r in rows
    ]

    # Latencia
    plt.figure(figsize=(7, 4))
    x = np.arange(len(backends))
    plt.bar(x, lat)
    plt.xticks(x, backends)
    plt.ylabel("Latencia (ms/batch)")
    plt.title(f"{title_prefix} — Latencia por backend")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    out_path_latency.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path_latency, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Guardado: {out_path_latency}")

    # Throughput
    plt.figure(figsize=(7, 4))
    x = np.arange(len(backends))
    plt.bar(x, thr)
    plt.xticks(x, backends)
    plt.ylabel("Throughput (samples/s)")
    plt.title(f"{title_prefix} — Throughput por backend")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    out_path_throughput.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path_throughput, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Guardado: {out_path_throughput}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    onnx_root = project_root / "models" / "onnx"
    global_dir = onnx_root / "global"
    global_dir.mkdir(parents=True, exist_ok=True)

    onnx_files = list(onnx_root.rglob("*.onnx"))
    if not onnx_files:
        print(f"[WARN] No se encontraron archivos .onnx en {onnx_root}")
        return

    print(f"[INFO] Se encontraron {len(onnx_files)} ONNX.")

    all_rows: List[Dict[str, Any]] = []

    for onnx_path in onnx_files:
        rel = onnx_path.relative_to(project_root)
        print(f"\n[INFO] ONNX: {rel}")

        # task inferido por carpeta: models/onnx/<task>/...
        try:
            parent_task = onnx_path.parent.relative_to(onnx_root).parts[0]
        except Exception:
            parent_task = "unknown"

        model_name, device_tag = parse_onnx_name(onnx_path)
        print(f"       task={parent_task} | model={model_name} | device_tag={device_tag}")

        # ---- PyTorch baseline
        if parent_task in ("classification", "detection"):
            pt = find_pytorch_metrics(project_root, parent_task, model_name, device_tag)
            if pt is not None:
                all_rows.append(row_from_pytorch(pt))
                print("       [OK] PyTorch metrics.")
            else:
                print("       [WARN] PyTorch metrics no encontradas.")

        # ---- ORT
        ort_json = find_onnxruntime_bench(onnx_path)
        if ort_json is not None:
            all_rows.append(row_from_onnxruntime(onnx_path, model_name, device_tag, parent_task, ort_json))
            print("       [OK] ORT benchmark.")
        else:
            print("       [WARN] ORT benchmark no encontrado.")

        # ---- TRT
        trt_json = find_tensorrt_bench(onnx_path)
        if trt_json is not None:
            all_rows.append(row_from_tensorrt(onnx_path, model_name, device_tag, parent_task, trt_json))
            print("       [OK] TensorRT benchmark.")
        else:
            print("       [WARN] TensorRT benchmark no encontrado.")

    if not all_rows:
        print("[WARN] No hay filas para exportar (faltan JSONs).")
        return

    # --------------------------------------------------------
    # CSV global
    # --------------------------------------------------------
    csv_path = global_dir / "onnx_backends_summary.csv"
    fieldnames = [
        "task",
        "model_name",
        "backend",
        "dataset",
        "num_classes",
        "device_tag",
        "device_name",
        "batch_size",
        "precision",
        "latency_ms_per_batch",
        "qps_batches",
        "throughput_samples_s",
        "vram_used_mb",
        "host",
        "timestamp_utc",
        "gpu_name",
        "device_id",
        "cuda_visible_devices",
        "onnxruntime_version",
        "tensorrt_version",
        "providers",
        "extra_info",
        "source_json",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            # asegurar todas las keys
            out = {k: r.get(k) for k in fieldnames}
            w.writerow(out)

    print(f"\n[INFO] CSV guardado en: {csv_path}")

    # --------------------------------------------------------
    # Gráficos por (task, model, device_tag)
    # --------------------------------------------------------
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for r in all_rows:
        key = (str(r.get("task")), str(r.get("model_name")), str(r.get("device_tag")))
        grouped.setdefault(key, []).append(r)

    print("[INFO] Generando gráficos...")

    for (task, model_name, device_tag), rows in grouped.items():
        backends_present = sorted(set(r["backend"] for r in rows if r.get("backend")))
        if len(backends_present) < 2:
            continue  # nada que comparar

        # Orden preferido: pytorch, onnxruntime, tensorrt
        order = {"pytorch": 0, "onnxruntime": 1, "tensorrt": 2}
        rows_sorted = sorted(rows, key=lambda x: order.get(x["backend"], 99))

        title_prefix = f"{task} — {model_name} — {device_tag}"
        safe_model = model_name.replace("/", "_")
        safe_dev = device_tag.replace("/", "_")

        out_lat = global_dir / f"{task}_{safe_model}_{safe_dev}_latency_by_backend.png"
        out_thr = global_dir / f"{task}_{safe_model}_{safe_dev}_throughput_by_backend.png"

        plot_backend_comparison(
            rows=rows_sorted,
            out_path_latency=out_lat,
            out_path_throughput=out_thr,
            title_prefix=title_prefix,
        )

    print("\n[INFO] onnx_results.py completado correctamente.")


if __name__ == "__main__":
    main()
