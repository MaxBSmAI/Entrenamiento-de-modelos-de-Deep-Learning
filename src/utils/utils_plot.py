"""
utils_plot.py

Funciones para graficar:

- Curvas de pérdida (loss) y accuracy
- Métricas por clase (precision/recall/F1)
- Matriz de confusión
- Métricas de benchmark computacional (latencia, FPS, throughput, VRAM, etc.)

Usa únicamente matplotlib para mantener compatibilidad en cualquier entorno.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np


# ======================================================
#  Curvas de entrenamiento (Loss, Accuracy)
# ======================================================

def plot_loss(
    losses: List[float],
    save_path: Path,
    title: str = "Loss",
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
) -> None:
    """
    Grafica una curva de pérdida simple.

    Pensado para usarse como:
        plot_loss(train_losses, result_dir / "train_loss.png")
        plot_loss(val_losses, result_dir / "val_loss.png")
    """
    epochs = range(1, len(losses) + 1)

    plt.figure()
    plt.plot(epochs, losses, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_accuracy(
    accuracies: List[float],
    save_path: Path,
    title: str = "Validation Accuracy",
    xlabel: str = "Epoch",
    ylabel: str = "Accuracy",
) -> None:
    """
    Grafica una curva de accuracy (por ejemplo, en validación).
    """
    epochs = range(1, len(accuracies) + 1)

    plt.figure()
    plt.plot(epochs, accuracies, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_train_val_loss(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Path,
    title: str = "Train vs Val Loss",
) -> None:
    """
    Versión combinada para mostrar train y val en una misma figura (opcional).
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, marker="o", label="Train")
    plt.plot(epochs, val_losses, marker="s", label="Validation")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


# ======================================================
#  Métricas por clase y matriz de confusión
# ======================================================

def plot_per_class_metrics(
    per_class: Dict[int, Dict[str, float]],
    save_path: Path,
    metric: str = "f1",
    title: Optional[str] = None,
) -> None:
    """
    Grafica una métrica por clase (precision, recall o f1).

    Parámetros
    ----------
    per_class : dict
        Salida de utils_metrics.per_class_metrics
        {clase: {"precision": p, "recall": r, "f1": f, "support": n}}
    save_path : Path
        Ruta donde se guarda la figura.
    metric : str
        'precision', 'recall' o 'f1'.
    """
    classes = sorted(per_class.keys())
    values = [per_class[c][metric] for c in classes]

    plt.figure(figsize=(10, 4))
    plt.bar(range(len(classes)), values)
    plt.xticks(range(len(classes)), classes, rotation=90)
    ttl = title if title is not None else f"{metric.upper()} por clase"
    plt.title(ttl)
    plt.xlabel("Clase")
    plt.ylabel(metric.upper())
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(
    cm_dict: Dict[str, Any],
    save_path: Path,
    title: str = "Matriz de confusión",
    cmap: str = "Blues",
) -> None:
    """
    Grafica una matriz de confusión.

    Parámetros
    ----------
    cm_dict : dict
        Salida de utils_metrics.confusion_matrix_metrics:
        {
            "confusion_matrix": [[...], [...], ...],
            "labels": [0,1,2,...]
        }
    save_path : Path
        Ruta de guardado de la figura.
    """
    cm = np.array(cm_dict["confusion_matrix"])
    labels = cm_dict["labels"]

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)

    # anotaciones
    thresh = cm.max() / 2.0 if cm.size > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                f"{cm[i, j]:.2f}" if cm.dtype == float else int(cm[i, j]),
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


# ======================================================
#  Gráficos de benchmark computacional
# ======================================================

def plot_benchmark_metrics(
    benchmark: Dict[str, Any],
    save_path: Path,
    title: str = "Benchmark computacional",
) -> None:
    """
    Grafica un resumen de métricas de utils_benchmark para UN modelo.

    Espera un diccionario del tipo:
        {
            "mean_latency_ms": ...,
            "std_latency_ms": ...,
            "fps": ...,
            "throughput": ...,
            "vram_used_mb": ...,
            "vram_total_mb": ...,
            "flops_g": ... (opcional),
            "params_m": ... (opcional),
            "power_w": ... (opcional),
            "efficiency_fps_w": ... (opcional),
        }
    """
    # Filtramos solo métricas numéricas relevantes
    keys = []
    values = []

    def _add_if_present(k: str, label: str) -> None:
        v = benchmark.get(k, None)
        if v is not None:
            keys.append(label)
            values.append(float(v))

    _add_if_present("mean_latency_ms", "Latencia (ms)")
    _add_if_present("fps", "FPS")
    _add_if_present("throughput", "Throughput (samples/s)")
    _add_if_present("vram_used_mb", "VRAM usada (MB)")
    _add_if_present("flops_g", "FLOPs (GFLOPs)")
    _add_if_present("params_m", "Parámetros (M)")
    _add_if_present("power_w", "Potencia (W)")
    _add_if_present("efficiency_fps_w", "FPS/W")

    if not keys:
        # nada que graficar
        return

    plt.figure(figsize=(8, 4))
    idx = np.arange(len(keys))
    plt.bar(idx, values)
    plt.xticks(idx, keys, rotation=45, ha="right")
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_benchmark_comparison(
    benchmarks: Dict[str, Dict[str, Any]],
    metric_key: str,
    save_path: Path,
    title: Optional[str] = None,
) -> None:
    """
    Compara un metric_key entre varios modelos/dispositivos.

    Parámetros
    ----------
    benchmarks : dict
        {nombre_modelo: benchmark_dict}
    metric_key : str
        Clave dentro de cada benchmark_dict (por ej. 'fps', 'mean_latency_ms').
    save_path : Path
        Ruta donde se guarda el gráfico.
    """
    names = []
    values = []

    for name, bm in benchmarks.items():
        if metric_key in bm and bm[metric_key] is not None:
            names.append(name)
            values.append(float(bm[metric_key]))

    if not names:
        return

    plt.figure(figsize=(8, 4))
    idx = np.arange(len(names))
    plt.bar(idx, values)
    plt.xticks(idx, names, rotation=45, ha="right")
    ttl = title if title is not None else f"Comparación de {metric_key}"
    plt.title(ttl)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
