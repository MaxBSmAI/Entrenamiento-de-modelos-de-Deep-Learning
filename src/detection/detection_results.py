"""
detection_results.py

Resumen y análisis de TODOS los modelos de detección entrenados:

    - Faster R-CNN (train_Faster_R_CNN.py)
    - YOLOv8s (train_yolov8s_coco.py)
    - RetinaNet (train_retinanet_coco.py)
    - Cualquier otro modelo que guarde *_metrics.json con task="detection"

Procesa estructura:

    result/detection/<model_name>/*_metrics.json

Genera:
    1) summary_detection_runs.csv
    2) Gráficos por dispositivo (mAP50, mAP50-95)
    3) Gráficos comparativos entre dispositivos para cada modelo:
        - FPS
        - Latencia (ms)
        - Throughput
        - VRAM usada (MB)
"""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Dict, Any, List
import sys

import matplotlib.pyplot as plt
import numpy as np

# Añadir src al sys.path
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.utils_plot import plot_benchmark_comparison


# ============================================================
# Búsqueda de archivos
# ============================================================

def find_all_detection_metrics(result_root: Path) -> List[Path]:
    """
    Busca recursivamente todos los *_metrics.json dentro de result/detection/.
    """
    return list(result_root.rglob("*_metrics.json"))


# ============================================================
# Parseo flexible de cada run de detección
# ============================================================

def parse_detection_run(path: Path) -> Dict[str, Any]:
    """
    Carga un JSON de métricas de detección y devuelve un diccionario estándar.
    Compatible con:
        - train_Faster_R_CNN.py
        - train_yolov8s_coco.py
        - train_retinanet_coco.py
        - Otros modelos que respeten la misma convención.
    """
    with open(path, "r") as f:
        data = json.load(f)

    model_name = data.get("model_name", path.parent.name)

    benchmark_data = data.get("benchmark", {}) or {}
    device_name = data.get("device_name", benchmark_data.get("device_name", "unknown"))
    device_tag = data.get("device_tag", "unknown")

    dataset = data.get("dataset", "unknown")
    num_classes = data.get("num_classes", None)
    epochs = data.get("epochs", None)
    best_epoch = data.get("best_epoch", None)

    # Métricas detección
    test_metrics = data.get("test_metrics", {}) or {}
    # Para compatibilidad: algunos scripts pueden usar 'map_50', otros 'mAP50'
    map_50 = (
        test_metrics.get("map_50")
        if "map_50" in test_metrics
        else test_metrics.get("mAP50")
    )
    # Igual para mAP50-95
    map_50_95 = (
        test_metrics.get("map_50_95")
        if "map_50_95" in test_metrics
        else test_metrics.get("mAP50_95")
    )

    # Benchmark
    benchmark = benchmark_data
    fps = benchmark.get("fps")
    latency = benchmark.get("mean_latency_ms")
    throughput = benchmark.get("throughput")
    vram_used = benchmark.get("vram_used_mb")

    return {
        "model_name": model_name,
        "device_name": device_name,
        "device_tag": device_tag,
        "dataset": dataset,
        "num_classes": num_classes,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "map_50": map_50,
        "map_50_95": map_50_95,
        "benchmark": {
            "fps": fps,
            "latency": latency,
            "throughput": throughput,
            "vram_used_mb": vram_used,
        },
        "path": str(path),
    }


# ============================================================
# CSV global
# ============================================================

def write_detection_csv(runs: List[Dict[str, Any]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary_detection_runs.csv"

    fieldnames = [
        "model_name",
        "device_name",
        "device_tag",
        "dataset",
        "num_classes",
        "epochs",
        "best_epoch",
        "map_50",
        "map_50_95",
        "fps",
        "latency",
        "throughput",
        "vram_used_mb",
        "path",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in runs:
            writer.writerow({
                "model_name": r["model_name"],
                "device_name": r["device_name"],
                "device_tag": r["device_tag"],
                "dataset": r["dataset"],
                "num_classes": r["num_classes"],
                "epochs": r["epochs"],
                "best_epoch": r["best_epoch"],
                "map_50": r["map_50"],
                "map_50_95": r["map_50_95"],
                "fps": r["benchmark"]["fps"],
                "latency": r["benchmark"]["latency"],
                "throughput": r["benchmark"]["throughput"],
                "vram_used_mb": r["benchmark"]["vram_used_mb"],
                "path": r["path"],
            })

    print(f"[INFO] CSV detección guardado en: {csv_path}")
    return csv_path


# ============================================================
# Gráficos por dispositivo
# ============================================================

def generate_per_device_metric_plots(runs: List[Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Agrupar por dispositivo
    by_device: Dict[str, List[Dict[str, Any]]] = {}
    for r in runs:
        dev = r["device_name"]
        by_device.setdefault(dev, []).append(r)

    for device_name, dev_runs in by_device.items():
        dev_runs = sorted(dev_runs, key=lambda x: x["model_name"])

        model_names = [r["model_name"] for r in dev_runs]
        map50_vals = [r["map_50"] for r in dev_runs]
        map95_vals = [r["map_50_95"] for r in dev_runs]

        # mAP@0.5
        valid_pairs = [(m, v) for m, v in zip(model_names, map50_vals) if v is not None]
        if valid_pairs:
            names, vals = zip(*valid_pairs)
            plt.figure(figsize=(8, 4))
            idx = np.arange(len(names))
            plt.bar(idx, vals)
            plt.xticks(idx, names, rotation=45, ha="right")
            plt.ylabel("mAP@0.5")
            plt.title(f"mAP@0.5 por modelo — {device_name}")
            plt.grid(axis="y", linestyle="--", alpha=0.5)
            path = out_dir / f"map50_by_model_{device_name.replace(' ', '_')}.png"
            plt.tight_layout()
            plt.savefig(path, bbox_inches="tight")
            plt.close()
            print(f"[INFO] Guardado: {path}")

        # mAP@0.5:0.95
        valid_pairs = [(m, v) for m, v in zip(model_names, map95_vals) if v is not None]
        if valid_pairs:
            names, vals = zip(*valid_pairs)
            plt.figure(figsize=(8, 4))
            idx = np.arange(len(names))
            plt.bar(idx, vals)
            plt.xticks(idx, names, rotation=45, ha="right")
            plt.ylabel("mAP@0.5:0.95")
            plt.title(f"mAP@0.5:0.95 por modelo — {device_name}")
            plt.grid(axis="y", linestyle="--", alpha=0.5)
            path = out_dir / f"map5095_by_model_{device_name.replace(' ', '_')}.png"
            plt.tight_layout()
            plt.savefig(path, bbox_inches="tight")
            plt.close()
            print(f"[INFO] Guardado: {path}")


# ============================================================
# Comparación entre dispositivos por modelo
# ============================================================

def generate_device_comparisons(runs: List[Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Agrupar por modelo
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for r in runs:
        by_model.setdefault(r["model_name"], []).append(r)

    metrics_to_compare = [
        ("fps", "FPS"),
        ("latency", "Latencia (ms)"),
        ("throughput", "Throughput (samples/s)"),
        ("vram_used_mb", "VRAM usada (MB)"),
    ]

    for model_name, model_runs in by_model.items():
        devices_present = sorted(set(r["device_name"] for r in model_runs))
        if len(devices_present) < 2:
            # sin comparación multi-dispositivo
            continue

        benchmarks_by_device = {
            r["device_name"]: r["benchmark"] for r in model_runs
        }

        for key, label in metrics_to_compare:
            non_null_count = sum(
                1 for bm in benchmarks_by_device.values()
                if key in bm and bm[key] is not None
            )
            if non_null_count < 2:
                continue

            save_path = out_dir / f"{model_name}_{key}_by_device.png"

            plot_benchmark_comparison(
                benchmarks_by_device,
                metric_key=key,
                save_path=save_path,
                title=f"{model_name} — {label} por dispositivo",
            )

            print(f"[INFO] Comparación guardada: {save_path}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    detect_root = project_root / "result" / "detection"
    global_dir = detect_root / "global"

    metric_files = find_all_detection_metrics(detect_root)
    if not metric_files:
        print("[WARN] No se encontraron *_metrics.json en result/detection/")
        return

    print(f"[INFO] {len(metric_files)} archivos de métricas encontrados.")

    runs: List[Dict[str, Any]] = []
    for p in metric_files:
        try:
            with open(p, "r") as f:
                dd = json.load(f)
            if dd.get("task", None) != "detection":
                continue

            r = parse_detection_run(p)
            runs.append(r)
        except Exception as e:
            print(f"[WARN] Error procesando {p}: {e}")

    if not runs:
        print("[WARN] No hay corridas válidas de detección.")
        return

    # 1) CSV global
    write_detection_csv(runs, global_dir)

    # 2) Gráficos por dispositivo
    generate_per_device_metric_plots(runs, global_dir)

    # 3) Comparaciones entre dispositivos (FPS, Latencia, etc.)
    generate_device_comparisons(runs, global_dir)

    print("\n[INFO] Análisis de detección completado correctamente.")


if __name__ == "__main__":
    main()
