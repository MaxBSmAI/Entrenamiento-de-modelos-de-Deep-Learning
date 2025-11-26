"""
utils_global.py

Genera un resumen GLOBAL del proyecto mezclando:

- Clasificación
- Detección
- Segmentación

Se generan:
    result/global/summary_all_runs.csv
    result/global/* gráficos comparativos

Ejecución:
    export PYTHONPATH=./src
    python src/global_result.py
"""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

from utils.utils_plot import plot_benchmark_comparison


# ============================================================
# Buscar TODOS los *_metrics.json del proyecto
# ============================================================

def find_all_metrics(result_root: Path) -> List[Path]:
    return list(result_root.rglob("*_metrics.json"))


# ============================================================
# Parseo flexible del JSON
# ============================================================

def parse_any_run(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = json.load(f)

    # Detectar la tarea
    if "classification" in str(path):
        task = "classification"
    elif "detection" in str(path):
        task = "detection"
    elif "segmentation" in str(path):
        task = "segmentation"
    else:
        task = "unknown"

    # Modelo = carpeta padre
    model_name = path.parent.name
    if model_name in ("classification", "detection", "segmentation"):
        model_name = path.stem.replace("_metrics", "")

    test_metrics = data.get("test_metrics", {})
    benchmark = data.get("benchmark", {})

    # Métricas estándar
    accuracy = test_metrics.get("accuracy")
    f1_macro = test_metrics.get("f1_macro")

    map50 = (
        test_metrics.get("map_50")
        or test_metrics.get("mAP_0.5")
        or test_metrics.get("map@0.5")
    )
    map5095 = (
        test_metrics.get("map_50_95")
        or test_metrics.get("mAP_0.5_0.95")
        or test_metrics.get("map@0.5:0.95")
    )

    miou = (
        test_metrics.get("miou")
        or test_metrics.get("mIoU")
        or test_metrics.get("mean_iou")
    )

    dice = (
        test_metrics.get("dice")
        or test_metrics.get("dice_score")
        or test_metrics.get("mean_dice")
    )

    # Benchmark
    device = benchmark.get("device_name", "unknown")
    fps = benchmark.get("fps")
    latency = benchmark.get("mean_latency_ms")
    throughput = benchmark.get("throughput")
    vram = benchmark.get("vram_used_mb")

    return {
        "task": task,
        "model_name": model_name,
        "device_name": device,
        "metrics": {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "map_50": map50,
            "map_50_95": map5095,
            "miou": miou,
            "dice": dice,
        },
        "benchmark": {
            "fps": fps,
            "latency": latency,
            "throughput": throughput,
            "vram_used_mb": vram,
        },
        "path": str(path),
    }


# ============================================================
# CSV Global
# ============================================================

def write_global_csv(runs: List[Dict[str, Any]], out_dir: Path) -> Path:
    csv_path = out_dir / "summary_all_runs.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "task",
        "model_name",
        "device_name",
        "accuracy",
        "f1_macro",
        "map_50",
        "map_50_95",
        "miou",
        "dice",
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
            row = {
                "task": r["task"],
                "model_name": r["model_name"],
                "device_name": r["device_name"],
                "accuracy": r["metrics"]["accuracy"],
                "f1_macro": r["metrics"]["f1_macro"],
                "map_50": r["metrics"]["map_50"],
                "map_50_95": r["metrics"]["map_50_95"],
                "miou": r["metrics"]["miou"],
                "dice": r["metrics"]["dice"],
                "fps": r["benchmark"]["fps"],
                "latency": r["benchmark"]["latency"],
                "throughput": r["benchmark"]["throughput"],
                "vram_used_mb": r["benchmark"]["vram_used_mb"],
                "path": r["path"],
            }
            writer.writerow(row)

    print(f"[INFO] CSV global escrito en: {csv_path}")
    return csv_path


# ============================================================
# Comparación RTX 4080 vs A100 (globally)
# ============================================================

def generate_global_device_comparison(runs: List[Dict[str, Any]], out_dir: Path) -> None:
    models = {}  # {(task, model): {"RTX 4080": run, "A100": run}}

    for r in runs:
        key = (r["task"], r["model_name"])
        models.setdefault(key, {})[r["device_name"]] = r

    metrics_to_compare = [
        ("fps", "FPS"),
        ("latency", "Latencia (ms)"),
        ("throughput", "Throughput"),
        ("vram_used_mb", "VRAM usada (MB)"),
    ]

    for (task, model), devs in models.items():
        if "RTX 4080" not in devs or "A100" not in devs:
            continue

        bm4080 = devs["RTX 4080"]["benchmark"]
        bma100 = devs["A100"]["benchmark"]

        for key, label in metrics_to_compare:
            if bm4080.get(key) is None or bma100.get(key) is None:
                continue

            save_path = out_dir / f"{task}_{model}_{key}_RTX4080_vs_A100.png"

            plot_benchmark_comparison(
                benchmarks={
                    "RTX 4080": bm4080,
                    "A100": bma100,
                },
                metric_key=key,
                save_path=save_path,
                title=f"{task.upper()} — {model}: {label} (RTX 4080 vs A100)",
            )

            print(f"[INFO] Gráfico global guardado: {save_path}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    result_root = project_root / "result"
    global_dir = result_root / "global"

    metric_files = find_all_metrics(result_root)
    if not metric_files:
        print("[WARN] No se encontraron *_metrics.json en result/")
        return

    runs = [parse_any_run(p) for p in metric_files]

    write_global_csv(runs, global_dir)
    generate_global_device_comparison(runs, global_dir)

    print("\n[INFO] Resultados globales generados correctamente.")


if __name__ == "__main__":
    main()
