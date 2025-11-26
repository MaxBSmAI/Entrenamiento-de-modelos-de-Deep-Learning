"""
utils_metrics.py

Funciones de métricas para:
- Clasificación (top-1 accuracy, precision, recall, F1)
- Métricas por clase
- Matriz de confusión (opcionalmente normalizada)

Pensado para usarse con:
    from utils.utils_metrics import (
        classification_metrics,
        per_class_metrics,
        confusion_matrix_metrics,
    )
"""

from typing import Dict, Any, Optional, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
) -> Dict[str, Any]:
    """
    Calcula métricas globales de clasificación.

    Parámetros
    ----------
    y_true : array-like
        Etiquetas verdaderas (enteros).
    y_pred : array-like
        Etiquetas predichas (enteros).
    average : str
        Tipo de promedio para precision/recall/F1.
        Valores típicos: 'macro', 'micro', 'weighted'.

    Retorna
    -------
    dict con:
        - accuracy
        - precision_<average>
        - recall_<average>
        - f1_<average>
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=average,
        zero_division=0,
    )

    metrics = {
        "accuracy": float(acc),
        f"precision_{average}": float(precision),
        f"recall_{average}": float(recall),
        f"f1_{average}": float(f1),
    }
    return metrics


def per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[int]] = None,
) -> Dict[int, Dict[str, float]]:
    """
    Calcula precision, recall y F1 POR CLASE.

    Parámetros
    ----------
    y_true : array-like
        Etiquetas verdaderas.
    y_pred : array-like
        Etiquetas predichas.
    labels : lista de int, opcional
        Lista de clases a considerar. Si es None, se infiere de y_true.

    Retorna
    -------
    dict: {clase: {"precision": p, "recall": r, "f1": f, "support": n}}
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if labels is None:
        labels = sorted(np.unique(y_true))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    per_class = {}
    for i, cls in enumerate(labels):
        per_class[int(cls)] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    return per_class


def confusion_matrix_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[int]] = None,
    normalize: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calcula la matriz de confusión (normalizada o no).

    Parámetros
    ----------
    y_true : array-like
        Etiquetas verdaderas.
    y_pred : array-like
        Etiquetas predichas.
    labels : lista de int, opcional
        Orden de las clases en la matriz.
    normalize : {'true', 'pred', 'all', None}
        Forma de normalización:
            - None  : cuentas absolutas
            - 'true': cada fila suma 1
            - 'pred': cada columna suma 1
            - 'all' : toda la matriz suma 1

    Retorna
    -------
    dict con:
        - "confusion_matrix": lista de listas (para ser serializable a JSON)
        - "labels": lista de etiquetas en el orden usado
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if labels is None:
        labels = sorted(np.unique(y_true))

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    cm_list = cm.tolist()

    return {
        "confusion_matrix": cm_list,
        "labels": [int(l) for l in labels],
    }


# Si quieres hacer una prueba rápida ejecutando este archivo directamente:
if __name__ == "__main__":
    # Ejemplo pequeño
    y_true = np.array([0, 1, 2, 2, 1, 0])
    y_pred = np.array([0, 2, 2, 2, 1, 0])

    global_metrics = classification_metrics(y_true, y_pred, average="macro")
    print("Métricas globales:", global_metrics)

    pc_metrics = per_class_metrics(y_true, y_pred)
    print("Métricas por clase:", pc_metrics)

    cm = confusion_matrix_metrics(y_true, y_pred)
    print("Matriz de confusión:", cm)
