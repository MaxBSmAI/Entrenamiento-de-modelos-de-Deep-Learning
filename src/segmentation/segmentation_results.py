"""
segmentation_results.py

Script para:
- Leer todos los *_metrics.json bajo result/segmentation/.
- Construir un resumen en CSV.
- Generar gráficos comparativos de:
    * Desempeño del modelo (mIoU, Dice) entre modelos en un mismo dispositivo.
    * Desempeño computacional (FPS, latencia, throughput, VRAM) entre:
        - modelos en un mismo dispositivo;
        - RTX 4080 vs A100 para cada modelo.

Salida principal en:
    result/segmentation/

Ejecución recomendada desde la raíz del proyecto:

    export PYTHONPATH=./src
    python src/segmentation/segmentation_results.py
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

def find_segmentation_metrics(result_root: Path) -> List[Path]:
    """
    Busca todos los *_metrics.json bajo result/segmentation/.
    """
    seg_root = result_root / "segmentation"
    if not seg_root.exists():
        return []
    return list(seg_root.rglob("*_metrics.json"))


def _get_miou(test_metrics: Dict[str, Any]) -> Any:
    """
    Intenta extraer una métrica de mIoU con nombres flexibles.
    """
    return (
        test_metrics.get("miou")
        or test_metrics.get("mIoU")
        or test_metrics.get("mean_iou")
        or test_metrics.get("iou_mean")
        or test_metrics.get("mIoU_val")
    )


def _get_dice(test_metrics: Dict[str, Any]) -> Any:
    """
    Intenta extraer una métrica de Dice con nombres flexibles.
    """
    return (
        test_metrics.get("dice")
        or test_metrics.get("dice_coef")
        or test_metrics.get("dice_score")
        or test_metrics.get("mean_dice")
    )


def parse_run_info(path: Path) -> Dict[str, Any]:
    """
    Parsea un archivo *_metrics.json de SEGMENTACIÓN.

    Estructura esperada (flexible):

    {
        "test_loss": ...,
        "test_metrics": {
            "miou": ...,
            "dice": ...,
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

    # Nombre de modelo: carpeta inmediatamente anterior o stem
    # Ejemplo: result/segmentation/deeplabv3_vine/deeplabv3_vine_metrics.json
    model_name = path.stem.replace("_metrics", "")
    if path.parent.name != "segmentation":
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
        "task": "segmentation",
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

def write_segmentation_csv(runs: List[Dict[str, Any]], seg_dir: Path) -> Path:
    """
    Escribe un CSV con el resumen de SEGMENTACIÓN.
    """
    csv_path = seg_dir / "summary_segmentation_runs.csv"
    seg_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model_name",
        "device_name",
        "test_loss",
        "miou",
        "dice",
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

            miou = _get_miou(tm)
            dice = _get_dice(tm)

            row = {
                "model_name": r.get("model_name"),
                "device_name": r.get("device_name"),
                "test_loss": r.get("test_loss"),
                "miou": miou,
                "dice": dice,
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

    print(f"[INFO] Resumen de segmentación escrito en: {csv_path}")
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
            "deeplabv3_vine": {
                "RTX 4080": run_info,
                "A100": run_info,
            },
            "segformer_vine": {
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
    metric_name: str,
    device_name: str,
    save_path: Path,
    title: str,
) -> None:
    """
    Grafica una métrica de desempeño (mIoU, Dice) entre modelos
    para un mismo dispositivo.
    """
    names: List[str] = []
    values: List[float] = []

    for r in runs:
        if str(r["device_name"]) != device_name:
            continue

        tm = r["test_metrics"]

        if metric_name == "miou":
            v = _get_miou(tm)
        elif metric_name == "dice":
            v = _get_dice(tm)
        else:
            v = tm.get(metric_name)

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
    plt.ylabel(metric_name)
    plt.title(f"{title} ({device_name})")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Gráfico guardado: {save_path}")


def plot_benchmark_across_models(
    runs: List[Dict[str, Any]],
    metric_key: str,
    device_name: str,
    save_path: Path,
    title: str,
) -> None:
    """
    Grafica una métrica de benchmark (fps, latencia, etc.) entre modelos
    para un mismo dispositivo.
    """
    names: List[str] = []
    values: List[float] = []

    for r in runs:
        if str(r["device_name"]) != device_name:
            continue

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
    seg_dir: Path,
) -> None:
    """
    Genera gráficos de:
      - mIoU entre modelos en cada dispositivo
      - Dice entre modelos en cada dispositivo
      - FPS entre modelos en cada dispositivo
      - Latencia entre modelos en cada dispositivo
    """
    by_device = group_by_device(runs)

    for device_name, runs_dev in by_device.items():
        # desempeño
        plot_metric_across_models(
            runs_dev,
            metric_name="miou",
            device_name=device_name,
            save_path=seg_dir / f"segmentation_miou_{device_name}.png",
            title="mIoU por modelo",
        )

        plot_metric_across_models(
            runs_dev,
            metric_name="dice",
            device_name=device_name,
            save_path=seg_dir / f"segmentation_dice_{device_name}.png",
            title="Dice por modelo",
        )

        # computacional
        plot_benchmark_across_models(
            runs_dev,
            metric_key="fps",
            device_name=device_name,
            save_path=seg_dir / f"segmentation_benchmark_fps_{device_name}.png",
            title="FPS por modelo",
        )

        plot_benchmark_across_models(
            runs_dev,
            metric_key="mean_latency_ms",
            device_name=device_name,
            save_path=seg_dir / f"segmentation_benchmark_latency_ms_{device_name}.png",
            title="Latencia (ms) por modelo",
        )


def generate_4080_vs_a100_plots(
    runs: List[Dict[str, Any]],
    seg_dir: Path,
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
            save_path = seg_dir / filename

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
    # src/segmentation/segmentation_results.py → project_root 2 niveles arriba
    project_root = Path(__file__).resolve().parents[2]
    result_root = project_root / "result"
    seg_dir = result_root / "segmentation"

    metric_files = find_segmentation_metrics(result_root)
    if not metric_files:
        print("[WARN] No se encontraron *_metrics.json en result/segmentation/")
        return

    runs = [parse_run_info(p) for p in metric_files]

    # 1) CSV resumen
    write_segmentation_csv(runs, seg_dir)

    # 2) Comparaciones entre modelos (mismo dispositivo)
    generate_model_vs_model_plots(runs, seg_dir)

    # 3) Comparaciones RTX 4080 vs A100 para cada modelo
    generate_4080_vs_a100_plots(runs, seg_dir, device_a="RTX 4080", device_b="A100")


if __name__ == "__main__":
    main()
