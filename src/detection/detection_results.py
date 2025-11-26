"""
detection_results.py

Script para:
- Leer todos los *_metrics.json bajo result/detection/.
- Construir un resumen en CSV.
- Generar gráficos comparativos de:
    * Desempeño del modelo (mAP@0.5, mAP@0.5:0.95) entre modelos en un mismo dispositivo.
    * Desempeño computacional (FPS, latencia, throughput, VRAM) entre:
        - modelos en un mismo dispositivo;
        - RTX 4080 vs A100 para cada modelo.

Salida principal en:
    result/detection/

Ejecución recomendada desde la raíz del proyecto:

    export PYTHONPATH=./src
    python src/detection/detection_results.py
"""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import numpy as np

from utils.utils_benchmark import throughput_from_latency
from utils.utils_plot import plot_benchmark_comparison


# ------------------------------------------------------------
# Utilidades de lectura
# ------------------------------------------------------------

def find_detection_metrics(result_root: Path) -> List[Path]:
    """
    Busca todos los *_metrics.json bajo result/detection/.
    """
    det_root = result_root / "detection"
    if not det_root.exists():
        return []
    return list(det_root.rglob("*_metrics.json"))


def parse_run_info(path: Path) -> Dict[str, Any]:
    """
    Parsea un archivo *_metrics.json de DETECCIÓN.

    Estructura esperada (flexible):

    {
        "test_loss": ...,
        "test_metrics": {
            "map_50": ...,
            "map_50_95": ...,
            ...
        },
        "benchmark": {
            "device_name": "RTX 4080",
            "batch_size": ...,
            "mean_latency_ms": ...,
            "fps": ...,
            "vram_used_mb": ...,
            ...
        },
        ...
    }
    """
    with open(path, "r") as f:
        data = json.load(f)

    # Nombre de modelo: usamos la carpeta inmediatamente anterior o el stem
    # Ejemplo: result/detection/detr_coco/detr_coco_metrics.json
    model_name = path.stem.replace("_metrics", "")
    if path.parent.name != "detection":
        model_name = path.parent.name

    test_metrics = data.get("test_metrics", {})
    benchmark = data.get("benchmark", {})

    device_name = (benchmark.get("device_name") or benchmark.get("device") or "unknown").strip()

    mean_latency_ms = benchmark.get("mean_latency_ms")
    fps = benchmark.get("fps")
    vram_used_mb = benchmark.get("vram_used_mb")
    vram_total_mb = benchmark.get("vram_total_mb")
    throughput = benchmark.get("throughput")

    if throughput is None and mean_latency_ms is not None:
        batch_size = data.get("batch_size", benchmark.get("batch_size", 1))
        throughput = throughput_from_latency(batch_size, mean_latency_ms)

    run_info = {
        "path": str(path),
        "task": "detection",
        "model_name": model_name,
        "device_name": device_name,
        "test_loss": data.get("test_loss"),
        "test_metrics": test_metrics,
        "benchmark": {
            "batch_size": data.get("batch_size", benchmark.get("batch_size")),
            "img_size": data.get("img_size", benchmark.get("img_size")),
            "mean_latency_ms": mean_latency_ms,
            "fps": fps,
            "throughput": throughput,
            "vram_used_mb": vram_used_mb,
            "vram_total_mb": vram_total_mb,
            "flops_g": benchmark.get("flops_g"),
            "params_m": benchmark.get("params_m"),
            "power_w": benchmark.get("power_w"),
            "efficiency_fps_w": benchmark.get("efficiency_fps_w"),
        },
    }
    return run_info


# ------------------------------------------------------------
# CSV resumen
# ------------------------------------------------------------

def write_detection_csv(runs: List[Dict[str, Any]], det_dir: Path) -> Path:
    """
    Escribe un CSV con el resumen de DETECCIÓN.
    """
    csv_path = det_dir / "summary_detection_runs.csv"
    det_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model_name",
        "device_name",
        "test_loss",
        "map_50",
        "map_50_95",
        "mean_latency_ms",
        "fps",
        "throughput",
        "vram_used_mb",
        "vram_total_mb",
        "flops_g",
        "params_m",
        "power_w",
        "efficiency_fps_w",
        "path",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in runs:
            tm = r.get("test_metrics", {})
            bm = r.get("benchmark", {})

            # nombres de métricas flexibles: intentamos encontrar mAP si tiene otro nombre
            map_50 = tm.get("map_50", tm.get("mAP_0.5", tm.get("map@0.5")))
            map_50_95 = tm.get(
                "map_50_95",
                tm.get("mAP_0.5_0.95", tm.get("map@0.5:0.95")),
            )

            row = {
                "model_name": r.get("model_name"),
                "device_name": r.get("device_name"),
                "test_loss": r.get("test_loss"),
                "map_50": map_50,
                "map_50_95": map_50_95,
                "mean_latency_ms": bm.get("mean_latency_ms"),
                "fps": bm.get("fps"),
                "throughput": bm.get("throughput"),
                "vram_used_mb": bm.get("vram_used_mb"),
                "vram_total_mb": bm.get("vram_total_mb"),
                "flops_g": bm.get("flops_g"),
                "params_m": bm.get("params_m"),
                "power_w": bm.get("power_w"),
                "efficiency_fps_w": bm.get("efficiency_fps_w"),
                "path": r.get("path"),
            }
            writer.writerow(row)

    print(f"[INFO] Resumen de detección escrito en: {csv_path}")
    return csv_path


# ------------------------------------------------------------
# Agrupar por dispositivo y por modelo
# ------------------------------------------------------------

def group_by_device(runs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Agrupa ejecuciones por device_name.
    """
    by_device: Dict[str, List[Dict[str, Any]]] = {}
    for r in runs:
        dev = str(r["device_name"])
        by_device.setdefault(dev, []).append(r)
    return by_device


def group_by_model_and_device(
    runs: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Agrupa por modelo y, dentro de cada modelo, por dispositivo.

    Retorna:
        {
            "detr_coco": {
                "RTX 4080": run_info,
                "A100": run_info,
            },
            "yolov8s_coco": {
                ...
            }
        }
    """
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in runs:
        m = r["model_name"]
        d = str(r["device_name"])
        out.setdefault(m, {})[d] = r
    return out


# ------------------------------------------------------------
# Gráficos auxiliares
# ------------------------------------------------------------

def plot_metric_across_models(
    runs: List[Dict[str, Any]],
    metric_key: str,
    metric_source: str,
    device_name: str,
    save_path: Path,
    title: str,
) -> None:
    """
    Grafica una métrica (map_50, fps, etc.) entre modelos
    para un mismo dispositivo.

    metric_source:
        - "test_metrics"
        - "benchmark"
    metric_key:
        - p.ej. "map_50", "map_50_95", "fps", "mean_latency_ms", ...
    """
    names: List[str] = []
    values: List[float] = []

    for r in runs:
        if str(r["device_name"]) != device_name:
            continue

        if metric_source == "test_metrics":
            tm = r["test_metrics"]

            # para mAP, aceptamos alias
            if metric_key == "map_50":
                v = tm.get("map_50", tm.get("mAP_0.5", tm.get("map@0.5")))
            elif metric_key == "map_50_95":
                v = tm.get("map_50_95", tm.get("mAP_0.5_0.95", tm.get("map@0.5:0.95")))
            else:
                v = tm.get(metric_key)
        else:
            v = r["benchmark"].get(metric_key)

        if v is None:
            continue

        names.append(r["model_name"])
        values.append(float(v))

    if not names:
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4))
    idx = np.arange(len(names))
    plt.bar(idx, values)
    plt.xticks(idx, names, rotation=45, ha="right")
    plt.ylabel(metric_key)
    plt.title(f"{title} ({device_name})")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Gráfico guardado: {save_path}")


def generate_model_vs_model_plots(
    runs: List[Dict[str, Any]],
    det_dir: Path,
) -> None:
    """
    Genera gráficos de:
      - mAP@0.5 entre modelos en cada dispositivo
      - mAP@0.5:0.95 entre modelos en cada dispositivo
      - FPS entre modelos en cada dispositivo
      - Latencia entre modelos en cada dispositivo
    """
    by_device = group_by_device(runs)

    for device_name, runs_dev in by_device.items():
        # desempeño (mAP)
        plot_metric_across_models(
            runs_dev,
            metric_key="map_50",
            metric_source="test_metrics",
            device_name=device_name,
            save_path=det_dir / f"detection_map50_{device_name}.png",
            title="mAP@0.5 por modelo",
        )

        plot_metric_across_models(
            runs_dev,
            metric_key="map_50_95",
            metric_source="test_metrics",
            device_name=device_name,
            save_path=det_dir / f"detection_map50_95_{device_name}.png",
            title="mAP@0.5:0.95 por modelo",
        )

        # computacional
        plot_metric_across_models(
            runs_dev,
            metric_key="fps",
            metric_source="benchmark",
            device_name=device_name,
            save_path=det_dir / f"detection_benchmark_fps_{device_name}.png",
            title="FPS por modelo",
        )

        plot_metric_across_models(
            runs_dev,
            metric_key="mean_latency_ms",
            metric_source="benchmark",
            device_name=device_name,
            save_path=det_dir / f"detection_benchmark_latency_ms_{device_name}.png",
            title="Latencia (ms) por modelo",
        )


def generate_4080_vs_a100_plots(
    runs: List[Dict[str, Any]],
    det_dir: Path,
    device_a: str = "RTX 4080",
    device_b: str = "A100",
) -> None:
    """
    Para cada modelo, genera comparación RTX 4080 vs A100
    para FPS, latencia, throughput y VRAM.
    """
    models = group_by_model_and_device(runs)

    metrics_to_compare = [
        ("fps", "FPS"),
        ("mean_latency_ms", "Latencia (ms)"),
        ("throughput", "Throughput (samples/s)"),
        ("vram_used_mb", "VRAM usada (MB)"),
    ]

    for model_name, by_dev in models.items():
        if device_a not in by_dev or device_b not in by_dev:
            continue

        bm_a = by_dev[device_a]["benchmark"]
        bm_b = by_dev[device_b]["benchmark"]

        benchmarks_dict = {
            device_a: bm_a,
            device_b: bm_b,
        }

        for metric_key, label in metrics_to_compare:
            if bm_a.get(metric_key) is None or bm_b.get(metric_key) is None:
                continue

            filename = f"{model_name}_{metric_key}_RTX4080_vs_A100.png"
            save_path = det_dir / filename

            plot_benchmark_comparison(
                benchmarks=benchmarks_dict,
                metric_key=metric_key,
                save_path=save_path,
                title=f"{label}: {model_name} (RTX 4080 vs A100)",
            )

            print(f"[INFO] Gráfico guardado: {save_path}")


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main() -> None:
    # src/detection/detection_results.py → project_root 2 niveles arriba
    project_root = Path(__file__).resolve().parents[2]
    result_root = project_root / "result"
    det_dir = result_root / "detection"

    metric_files = find_detection_metrics(result_root)
    if not metric_files:
        print("[WARN] No se encontraron *_metrics.json en result/detection/")
        return

    runs = [parse_run_info(p) for p in metric_files]

    # 1) CSV resumen
    write_detection_csv(runs, det_dir)

    # 2) Comparaciones entre modelos (mismo dispositivo)
    generate_model_vs_model_plots(runs, det_dir)

    # 3) Comparaciones RTX 4080 vs A100 para cada modelo
    generate_4080_vs_a100_plots(runs, det_dir, device_a="RTX 4080", device_b="A100")


if __name__ == "__main__":
    main()
