# src/onnx/onnx_results.py

from __future__ import annotations

"""
onnx_results.py

Integra resultados de:
    - PyTorch (benchmarks incluidos en *_metrics.json)
    - ONNX Runtime (benchmark_onnx_runtime.py)
    - TensorRT (benchmark_tensorrt.py)

Soporta:
    - Clasificación (ResNet-50, ViT-B/16, ...)
    - Detección (Faster R-CNN, RetinaNet, ...)

Estructuras esperadas:

1) Métricas PyTorch
   result/<task>/<model_name>/<device_tag>/<model_name>_metrics.json

   Debe contener algo como:
       {
         "model_name": "...",
         "task": "classification" | "detection",
         "dataset": "...",
         "num_classes": ...,
         "device_name": "...",
         "device_tag": "...",
         "benchmark": {
            "mean_latency_ms": ...,
            "fps": ...,
            "throughput": ...,
            "vram_used_mb": ...,
            ...
         },
         ...
       }

2) ONNX Runtime
   benchmark_onnx_runtime.py guarda:
       <onnx_name>_onnxruntime_benchmark.json

   Ej:
       models/onnx/classification/resnet50_miniimagenet__RTX_4080_onnxruntime_benchmark.json

   Contiene algo como:
       {
         "onnx_model_path": "...",
         "task": "classification",
         "batch_size": ...,
         "num_iterations": ...,
         "mean_latency_ms": ...,
         "fps": ...,
         "throughput": ...,
         "providers": [...],
         "note": "..."
       }

3) TensorRT
   benchmark_tensorrt.py guarda:
       <onnx_name>_tensorrt_bench.json

   Ej:
       models/onnx/classification/resnet50_miniimagenet__RTX_4080_tensorrt_bench.json

   Contiene algo como:
       {
          "onnx_path": "...",
          "task": "classification",
          "batch_size": ...,
          "precision": "fp16" | "fp32",
          "benchmark_tensorrt": {
              "mean_latency_ms": ...,
              "throughput": ...,
              "fps": ...,
              ...
          }
       }

Salida principal:
    - CSV global:
        models/onnx/global/onnx_backends_summary.csv

    - Gráficos comparativos por backend (latencia y FPS) por
      (task, model_name, device_tag) en:
        models/onnx/global/*.png
"""

import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import sys
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Helpers de rutas / lectura
# ============================================================

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    """
    Carga un JSON si existe, devolviendo None en caso de error.
    """
    if not path.is_file():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Error al leer JSON {path}: {e}")
        return None


def parse_onnx_name(onnx_path: Path) -> Tuple[str, str]:
    """
    A partir del nombre del archivo ONNX:
        <model_name>__<device_tag>.onnx

    devuelve:
        model_name, device_tag

    Si no puede parsear, usa heurísticas básicas.
    """
    stem = onnx_path.stem  # sin extensión
    # Quitar posibles sufijos añadidos accidentalmente
    # (ej: si alguien puso .engine o algo raro, nos quedamos con lo antes del primer '.')
    if "." in stem:
        stem = stem.split(".", 1)[0]

    if "__" in stem:
        model_name, device_tag = stem.split("__", 1)
    else:
        # fallback: sin device_tag claro
        model_name = stem
        device_tag = "UNKNOWN"
    return model_name, device_tag


# ============================================================
# Colectar info PyTorch (baseline)
# ============================================================

def find_pytorch_metrics(
    project_root: Path,
    task: str,
    model_name: str,
    device_tag: str,
) -> Optional[Dict[str, Any]]:
    """
    Busca el archivo *_metrics.json correspondiente a PyTorch para un modelo dado.

    Rutas probadas:
        result/<task>/<model_name>/<device_tag>/<model_name>_metrics.json
        result/<task>/<model_name>/<model_name>_metrics.json
    """
    task_root = project_root / "result" / task / model_name
    cand1 = task_root / device_tag / f"{model_name}_metrics.json"
    cand2 = task_root / f"{model_name}_metrics.json"

    data = safe_load_json(cand1)
    if data is not None:
        return data

    data = safe_load_json(cand2)
    if data is not None:
        return data

    return None


# ============================================================
# Colectar info ONNX Runtime / TensorRT
# ============================================================

def find_onnxruntime_bench(onnx_path: Path) -> Optional[Dict[str, Any]]:
    """
    Dado un ONNX, busca el JSON de ONNX Runtime en el mismo directorio:

        <onnx_stem>_onnxruntime_benchmark.json
    """
    json_path = onnx_path.with_suffix("")  # quita .onnx
    json_path = json_path.with_name(json_path.name + "_onnxruntime_benchmark.json")
    return safe_load_json(json_path)


def find_tensorrt_bench(onnx_path: Path) -> Optional[Dict[str, Any]]:
    """
    Dado un ONNX, busca el JSON de TensorRT en el mismo directorio:

        <onnx_stem>_tensorrt_bench.json
    """
    json_path = onnx_path.with_suffix("")  # quita .onnx
    json_path = json_path.with_name(json_path.name + "_tensorrt_bench.json")
    return safe_load_json(json_path)


# ============================================================
# Construir filas para CSV global
# ============================================================

def row_from_pytorch(
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convierte un JSON de métricas PyTorch en una fila estándar de comparación.
    """
    model_name = data.get("model_name", "unknown_model")
    task = data.get("task", "unknown_task")
    dataset = data.get("dataset", "unknown_dataset")
    num_classes = data.get("num_classes", None)
    device_name = data.get("device_name", "unknown_device")
    device_tag = data.get("device_tag", "UNKNOWN")

    benchmark = data.get("benchmark", {}) or {}
    mean_latency_ms = benchmark.get("mean_latency_ms")
    fps = benchmark.get("fps")
    throughput = benchmark.get("throughput")
    vram_used_mb = benchmark.get("vram_used_mb")

    return {
        "task": task,
        "model_name": model_name,
        "backend": "pytorch",
        "dataset": dataset,
        "num_classes": num_classes,
        "device_name": device_name,
        "device_tag": device_tag,
        "batch_size": benchmark.get("batch_size", None),
        "precision": "fp32",  # asumimos fp32
        "mean_latency_ms": mean_latency_ms,
        "fps": fps,
        "throughput": throughput,
        "vram_used_mb": vram_used_mb,
        "extra_info": "",
        "source_json": data.get("artifacts", {}).get("best_model_path", ""),
    }


def row_from_onnxruntime(
    onnx_path: Path,
    model_name: str,
    device_tag: str,
    task_fallback: str,
    ort_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convierte JSON de ONNX Runtime en una fila estándar.
    """
    task = ort_data.get("task", task_fallback)
    batch_size = ort_data.get("batch_size", None)
    mean_latency_ms = ort_data.get("mean_latency_ms", None)
    fps = ort_data.get("fps", None)
    throughput = ort_data.get("throughput", None)
    providers = ort_data.get("providers", [])

    return {
        "task": task,
        "model_name": model_name,
        "backend": "onnxruntime",
        "dataset": None,
        "num_classes": None,
        "device_name": None,
        "device_tag": device_tag,
        "batch_size": batch_size,
        "precision": "fp32",  # por defecto asumimos fp32
        "mean_latency_ms": mean_latency_ms,
        "fps": fps,
        "throughput": throughput,
        "vram_used_mb": None,
        "extra_info": f"providers={providers}",
        "source_json": str(onnx_path.with_suffix("").with_name(
            onnx_path.with_suffix("").name + "_onnxruntime_benchmark.json"
        )),
    }


def row_from_tensorrt(
    onnx_path: Path,
    model_name: str,
    device_tag: str,
    task_fallback: str,
    trt_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convierte JSON de TensorRT en una fila estándar.
    """
    task = trt_data.get("task", task_fallback)
    batch_size = trt_data.get("batch_size", None)
    precision = trt_data.get("precision", "fp16/32?")
    bench = trt_data.get("benchmark_tensorrt", {}) or {}

    mean_latency_ms = bench.get("mean_latency_ms", None)
    fps = bench.get("fps", None)
    throughput = bench.get("throughput", None)

    return {
        "task": task,
        "model_name": model_name,
        "backend": "tensorrt",
        "dataset": None,
        "num_classes": None,
        "device_name": None,
        "device_tag": device_tag,
        "batch_size": batch_size,
        "precision": precision,
        "mean_latency_ms": mean_latency_ms,
        "fps": fps,
        "throughput": throughput,
        "vram_used_mb": None,
        "extra_info": "",
        "source_json": str(onnx_path.with_suffix("").with_name(
            onnx_path.with_suffix("").name + "_tensorrt_bench.json"
        )),
    }


# ============================================================
# Gráficos comparativos por backend
# ============================================================

def plot_backend_comparison(
    rows: List[Dict[str, Any]],
    out_path_latency: Path,
    out_path_fps: Path,
    title_prefix: str,
) -> None:
    """
    Dibuja dos gráficos de barras:
        - Latencia media (ms) por backend
        - FPS por backend
    para un conjunto de filas (todas mismas task/model/device_tag).
    """
    backends = []
    latencies = []
    fps_values = []

    for r in rows:
        b = r["backend"]
        mean_lat = r["mean_latency_ms"]
        fps = r["fps"]

        backends.append(b)
        latencies.append(mean_lat if mean_lat is not None else np.nan)
        fps_values.append(fps if fps is not None else np.nan)

    # Latencia
    plt.figure(figsize=(6, 4))
    x = np.arange(len(backends))
    plt.bar(x, latencies)
    plt.xticks(x, backends)
    plt.ylabel("Latencia media (ms)")
    plt.title(f"{title_prefix} — Latencia por backend")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out_path_latency.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path_latency, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Guardado gráfico de latencia: {out_path_latency}")

    # FPS
    plt.figure(figsize=(6, 4))
    x = np.arange(len(backends))
    plt.bar(x, fps_values)
    plt.xticks(x, backends)
    plt.ylabel("FPS (aprox. samples/s)")
    plt.title(f"{title_prefix} — FPS por backend")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path_fps, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Guardado gráfico de FPS: {out_path_fps}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    onnx_root = project_root / "models" / "onnx"
    global_dir = onnx_root / "global"
    global_dir.mkdir(parents=True, exist_ok=True)

    # 1) Buscar todos los modelos ONNX
    onnx_files = list(onnx_root.rglob("*.onnx"))
    if not onnx_files:
        print(f"[WARN] No se encontraron archivos .onnx en {onnx_root}")
        return

    print(f"[INFO] Se encontraron {len(onnx_files)} modelos ONNX.")

    all_rows: List[Dict[str, Any]] = []

    for onnx_path in onnx_files:
        rel = onnx_path.relative_to(project_root)
        print(f"\n[INFO] Procesando ONNX: {rel}")

        # Inferir task desde carpeta padre (classification/detection)
        try:
            parent_task = onnx_path.parent.relative_to(onnx_root).parts[0]
        except Exception:
            parent_task = "unknown"

        model_name, device_tag = parse_onnx_name(onnx_path)
        print(f"       model_name={model_name} | device_tag={device_tag} | task={parent_task}")

        # --------------------------------------------------------
        # PyTorch (baseline) - si existe
        # --------------------------------------------------------
        if parent_task in ("classification", "detection"):
            pyt_metrics = find_pytorch_metrics(
                project_root=project_root,
                task=parent_task,
                model_name=model_name,
                device_tag=device_tag,
            )
            if pyt_metrics is not None:
                row_pt = row_from_pytorch(pyt_metrics)
                all_rows.append(row_pt)
                print("       [OK] PyTorch metrics encontradas.")
            else:
                print("       [WARN] No se encontraron métricas PyTorch para este modelo.")

        # --------------------------------------------------------
        # ONNX Runtime
        # --------------------------------------------------------
        ort_data = find_onnxruntime_bench(onnx_path)
        if ort_data is not None:
            row_ort = row_from_onnxruntime(
                onnx_path=onnx_path,
                model_name=model_name,
                device_tag=device_tag,
                task_fallback=parent_task,
                ort_data=ort_data,
            )
            all_rows.append(row_ort)
            print("       [OK] ONNX Runtime benchmark encontrado.")
        else:
            print("       [WARN] No se encontró benchmark de ONNX Runtime para este modelo.")

        # --------------------------------------------------------
        # TensorRT
        # --------------------------------------------------------
        trt_data = find_tensorrt_bench(onnx_path)
        if trt_data is not None:
            row_trt = row_from_tensorrt(
                onnx_path=onnx_path,
                model_name=model_name,
                device_tag=device_tag,
                task_fallback=parent_task,
                trt_data=trt_data,
            )
            all_rows.append(row_trt)
            print("       [OK] TensorRT benchmark encontrado.")
        else:
            print("       [WARN] No se encontró benchmark de TensorRT para este modelo.")

    if not all_rows:
        print("[WARN] No se generaron filas de resultados (¿faltan JSONs?).")
        return

    # --------------------------------------------------------
    # Guardar CSV global
    # --------------------------------------------------------
    csv_path = global_dir / "onnx_backends_summary.csv"
    fieldnames = [
        "task",
        "model_name",
        "backend",
        "dataset",
        "num_classes",
        "device_name",
        "device_tag",
        "batch_size",
        "precision",
        "mean_latency_ms",
        "fps",
        "throughput",
        "vram_used_mb",
        "extra_info",
        "source_json",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)

    print(f"\n[INFO] CSV global guardado en: {csv_path}")

    # --------------------------------------------------------
    # Gráficos comparativos por (task, model_name, device_tag)
    # --------------------------------------------------------
    print("[INFO] Generando gráficos comparativos por backend...")

    # Agrupar filas por clave (task, model_name, device_tag)
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for r in all_rows:
        key = (r["task"], r["model_name"], r["device_tag"])
        grouped.setdefault(key, []).append(r)

    for (task, model_name, device_tag), rows in grouped.items():
        # Necesitamos al menos 2 backends para comparar
        backends_present = sorted(set(r["backend"] for r in rows))
        if len(backends_present) < 2:
            continue

        title_prefix = f"{task} — {model_name} — {device_tag}"
        safe_model = model_name.replace("/", "_")
        safe_dev = device_tag.replace("/", "_")

        out_lat = global_dir / f"{task}_{safe_model}_{safe_dev}_latency_by_backend.png"
        out_fps = global_dir / f"{task}_{safe_model}_{safe_dev}_fps_by_backend.png"

        plot_backend_comparison(
            rows=rows,
            out_path_latency=out_lat,
            out_path_fps=out_fps,
            title_prefix=title_prefix,
        )

    print("\n[INFO] onnx_results.py completado correctamente.")


if __name__ == "__main__":
    main()
