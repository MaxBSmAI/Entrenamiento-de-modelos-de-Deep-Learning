"""
classification_results.py

Resumen y análisis de TODOS los modelos de clasificación entrenados.

- Busca recursivamente todos los *_metrics.json bajo:
    result/classification/

- Espera el formato de JSON generado por:
    train_resnet50_imagenet.py
    train_vit_b16_imagenet.py

- Genera:
    1) CSV global de resultados de clasificación:
        result/classification/summary_classification_runs.csv

    2) Gráficos por dispositivo:
        - Accuracy por modelo
        - F1-macro por modelo

    3) Gráficos de comparación entre dispositivos (por modelo):
        - FPS
        - Latencia media (ms)
        - Throughput
        - VRAM usada (MB)

Ejecución recomendada (desde la raíz del proyecto):

    cd /workspace
    export PYTHONPATH=./src
    python src/classification/classification_results.py
"""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Dict, Any, List

import sys

# ----------------------------------------------------------------------
# AÑADIR src al sys.path para que funcionen los imports (utils, etc.)
# ----------------------------------------------------------------------
SRC_ROOT = Path(__file__).resolve().parents[1]  # .../src
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
# ----------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

from utils.utils_plot import plot_benchmark_comparison


# ============================================================
# Búsqueda de archivos *_metrics.json
# ============================================================

def find_all_classification_metrics(result_root: Path) -> List[Path]:
    """
    Busca recursivamente todos los *_metrics.json dentro de result/classification/.
    """
    return list(result_root.rglob("*_metrics.json"))


# ============================================================
# Parseo flexible de cada run de clasificación
# ============================================================

def parse_classification_run(path: Path) -> Dict[str, Any]:
    """
    Carga un JSON de métricas de clasificación y devuelve un diccionario estándar.

    Espera el formato generado por train_resnet50_imagenet.py y train_vit_b16_imagenet.py.
    """
    with open(path, "r") as f:
        data = json.load(f)

    model_name = data.get("model_name", path.parent.name)
    device_name = data.get("device_name", "unknown")
    device_tag = data.get("device_tag", "unknown")
    dataset = data.get("dataset", "unknown")
    num_classes = data.get("num_classes", None)
    epochs = data.get("epochs", None)
    best_epoch = data.get("best_epoch", None)

    test_loss = data.get("test_loss", None)
    test_metrics = data.get("test_metrics", {}) or {}
    benchmark = data.get("benchmark", {}) or {}

    accuracy = test_metrics.get("accuracy")
    f1_macro = test_metrics.get("f1_macro")
    precision_macro = test_metrics.get("precision_macro")
    recall_macro = test_metrics.get("recall_macro")

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
        "test_loss": test_loss,
        "metrics": {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
        },
        "benchmark": {
            "fps": fps,
            "latency": latency,
            "throughput": throughput,
            "vram_used_mb": vram_used,
        },
        "path": str(path),
    }


# ============================================================
# Escritura de CSV global
# ============================================================

def write_classification_csv(runs: List[Dict[str, Any]], out_dir: Path) -> Path:
    """
    Escribe un CSV resumen con todas las corridas de clasificación.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary_classification_runs.csv"

    fieldnames = [
        "model_name",
        "device_name",
        "device_tag",
        "dataset",
        "num_classes",
        "epochs",
        "best_epoch",
        "test_loss",
        "accuracy",
        "f1_macro",
        "precision_macro",
        "recall_macro",
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
                "model_name": r["model_name"],
                "device_name": r["device_name"],
                "device_tag": r["device_tag"],
                "dataset": r["dataset"],
                "num_classes": r["num_classes"],
                "epochs": r["epochs"],
                "best_epoch": r["best_epoch"],
                "test_loss": r["test_loss"],
                "accuracy": r["metrics"]["accuracy"],
                "f1_macro": r["metrics"]["f1_macro"],
                "precision_macro": r["metrics"]["precision_macro"],
                "recall_macro": r["metrics"]["recall_macro"],
                "fps": r["benchmark"]["fps"],
                "latency": r["benchmark"]["latency"],
                "throughput": r["benchmark"]["throughput"],
                "vram_used_mb": r["benchmark"]["vram_used_mb"],
                "path": r["path"],
            }
            writer.writerow(row)

    print(f"[INFO] CSV de clasificación escrito en: {csv_path}")
    return csv_path


# ============================================================
# Gráficos por dispositivo (accuracy y F1 por modelo)
# ============================================================

def generate_per_device_metric_plots(
    runs: List[Dict[str, Any]],
    out_dir: Path,
) -> None:
    """
    Para cada dispositivo, genera gráficos de:

        - Accuracy por modelo
        - F1-macro por modelo

    Usando todos los modelos de clasificación entrenados en ese dispositivo.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Agrupar por dispositivo
    by_device: Dict[str, List[Dict[str, Any]]] = {}
    for r in runs:
        dev = r["device_name"]
        by_device.setdefault(dev, []).append(r)

    for device_name, dev_runs in by_device.items():
        # Ordenar por model_name para gráficos más legibles
        dev_runs = sorted(dev_runs, key=lambda x: x["model_name"])

        model_names = [r["model_name"] for r in dev_runs]
        accuracies = [r["metrics"]["accuracy"] for r in dev_runs]
        f1_macros = [r["metrics"]["f1_macro"] for r in dev_runs]

        # Filtrar None para evitar errores
        acc_pairs = [(m, a) for m, a in zip(model_names, accuracies) if a is not None]
        f1_pairs = [(m, f) for m, f in zip(model_names, f1_macros) if f is not None]

        # Accuracy por modelo
        if acc_pairs:
            names_acc, vals_acc = zip(*acc_pairs)

            plt.figure(figsize=(8, 4))
            idx = np.arange(len(names_acc))
            plt.bar(idx, vals_acc)
            plt.xticks(idx, names_acc, rotation=45, ha="right")
            plt.ylabel("Accuracy")
            plt.title(f"Accuracy por modelo — {device_name}")
            plt.grid(axis="y", linestyle="--", alpha=0.5)

            acc_path = out_dir / f"accuracy_by_model_{device_name.replace(' ', '_')}.png"
            plt.tight_layout()
            plt.savefig(acc_path, bbox_inches="tight")
            plt.close()

            print(f"[INFO] Gráfico accuracy por modelo guardado: {acc_path}")

        # F1-macro por modelo
        if f1_pairs:
            names_f1, vals_f1 = zip(*f1_pairs)

            plt.figure(figsize=(8, 4))
            idx = np.arange(len(names_f1))
            plt.bar(idx, vals_f1)
            plt.xticks(idx, names_f1, rotation=45, ha="right")
            plt.ylabel("F1-macro")
            plt.title(f"F1-macro por modelo — {device_name}")
            plt.grid(axis="y", linestyle="--", alpha=0.5)

            f1_path = out_dir / f"f1macro_by_model_{device_name.replace(' ', '_')}.png"
            plt.tight_layout()
            plt.savefig(f1_path, bbox_inches="tight")
            plt.close()

            print(f"[INFO] Gráfico F1-macro por modelo guardado: {f1_path}")


# ============================================================
# Comparación entre dispositivos (por modelo)
# ============================================================

def generate_device_comparisons(
    runs: List[Dict[str, Any]],
    out_dir: Path,
) -> None:
    """
    Para cada modelo, si existe al menos 2 dispositivos distintos (ej. RTX 4080, A100),
    genera comparaciones de:

        - FPS
        - Latencia (ms)
        - Throughput
        - VRAM usada (MB)

    usando plot_benchmark_comparison de utils_plot.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Agrupar por modelo
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for r in runs:
        mname = r["model_name"]
        by_model.setdefault(mname, []).append(r)

    metrics_to_compare = [
        ("fps", "FPS"),
        ("latency", "Latencia (ms)"),
        ("throughput", "Throughput (samples/s)"),
        ("vram_used_mb", "VRAM usada (MB)"),
    ]

    for model_name, model_runs in by_model.items():
        # Necesitamos al menos dos dispositivos distintos para comparar
        devices_present = sorted(set(r["device_name"] for r in model_runs))
        if len(devices_present) < 2:
            continue

        # Construimos un diccionario {device_name: benchmark_dict}
        benchmarks_by_device: Dict[str, Dict[str, Any]] = {}
        for r in model_runs:
            dev = r["device_name"]
            benchmarks_by_device[dev] = r["benchmark"]

        for key, label in metrics_to_compare:
            # Comprobar que al menos dos dispositivos tienen la métrica
            non_null_count = sum(
                1 for bm in benchmarks_by_device.values()
                if key in bm and bm[key] is not None
            )
            if non_null_count < 2:
                continue

            save_path = out_dir / f"{model_name}_{key}_by_device.png"

            plot_benchmark_comparison(
                benchmarks=benchmarks_by_device,
                metric_key=key,
                save_path=save_path,
                title=f"{model_name} — {label} por dispositivo",
            )

            print(f"[INFO] Comparación entre dispositivos guardada: {save_path}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    class_result_root = project_root / "result" / "classification"
    global_dir = class_result_root / "global"

    metric_files = find_all_classification_metrics(class_result_root)
    if not metric_files:
        print("[WARN] No se encontraron *_metrics.json en result/classification/")
        return

    print(f"[INFO] Encontrados {len(metric_files)} archivos de métricas de clasificación.")

    runs: List[Dict[str, Any]] = []
    for p in metric_files:
        try:
            r = parse_classification_run(p)
            # Filtrar por task == classification si existe (por seguridad)
            task = None
            try:
                with open(p, "r") as f:
                    d = json.load(f)
                    task = d.get("task", None)
            except Exception:
                pass

            if task is not None and task != "classification":
                continue

            runs.append(r)
        except Exception as e:
            print(f"[WARN] Error al parsear {p}: {e}")

    if not runs:
        print("[WARN] No hay corridas de clasificación válidas.")
        return

    # 1) CSV global
    write_classification_csv(runs, global_dir)

    # 2) Gráficos por dispositivo (accuracy y F1-macro por modelo)
    generate_per_device_metric_plots(runs, global_dir)

    # 3) Comparación entre dispositivos (FPS, latencia, throughput, VRAM)
    generate_device_comparisons(runs, global_dir)

    print("\n[INFO] Resultados de clasificación generados correctamente.")


if __name__ == "__main__":
    main()
