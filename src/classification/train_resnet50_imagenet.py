"""
train_resnet50_imagenet.py

Entrena un modelo ResNet-50 en Mini-ImageNet (streaming desde HuggingFace)
y guarda:

- El mejor modelo según val_loss (early stopping).
- Un archivo JSON con:
    * test_loss
    * test_metrics: accuracy, f1_macro, precision_macro, recall_macro
    * benchmark: latencia, fps, throughput, VRAM, etc.

Salida principal en:
    result/classification/resnet50_miniimagenet/
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List

import sys

# --- AÑADIR src al sys.path para que funcionen los imports relativos ---
SRC_ROOT = Path(__file__).resolve().parents[1]  # .../src
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
# ----------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from data.dataloaders import get_miniimagenet_dataloaders
from utils.utils_benchmark import throughput_from_latency


# ------------------------------------------------------------
# Utilidades generales
# ------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Crea un ResNet-50 y reemplaza la capa final para num_classes.
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


# ------------------------------------------------------------
# Loop de entrenamiento y evaluación
# ------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Entrena una época completa.

    IMPORTANTE: soporta IterableDataset → NO usa len(dataloader.dataset).
    En su lugar, va contando explícitamente cuántos ejemplos se han visto.
    """
    model.train()
    running_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []
    n_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_targets.extend(labels.detach().cpu().tolist())

    # si por alguna razón no hubo datos, evitamos división por 0
    if n_samples == 0:
        epoch_loss = 0.0
        epoch_acc = 0.0
    else:
        epoch_loss = running_loss / n_samples
        epoch_acc = accuracy_score(all_targets, all_targets if not all_preds else all_preds)

    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """
    Evalúa en validación o test.

    También soporta IterableDataset (cuenta n_samples manualmente).
    """
    model.eval()
    running_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []
    n_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_targets.extend(labels.detach().cpu().tolist())

    if n_samples == 0:
        epoch_loss = 0.0
        acc = f1_macro = prec_macro = rec_macro = 0.0
    else:
        epoch_loss = running_loss / n_samples

        acc = accuracy_score(all_targets, all_preds)
        f1_macro = f1_score(all_targets, all_preds, average="macro")
        prec_macro = precision_score(all_targets, all_preds, average="macro", zero_division=0)
        rec_macro = recall_score(all_targets, all_preds, average="macro", zero_division=0)

    metrics = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
    }
    return epoch_loss, metrics


# ------------------------------------------------------------
# Early Stopping sencillo
# ------------------------------------------------------------

class EarlyStopping:
    """
    Early stopping basado en la métrica de validación (p.ej. val_loss).

    Para este script, usamos val_loss (minimizar).
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_value: float | None = None
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> None:
        if self.best_value is None or (self.best_value - value) > self.min_delta:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


# ------------------------------------------------------------
# Benchmark de inferencia
# ------------------------------------------------------------

@torch.no_grad()
def benchmark_classification_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int = 20,
) -> Dict[str, Any]:
    """
    Mide latencia promedio por batch, FPS y VRAM de forma sencilla.
    """
    model.eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        device_name = torch.cuda.get_device_name(0)
    else:
        device_name = "cpu"

    total_images = 0
    total_time = 0.0
    num_batches = 0

    for i, (images, _) in enumerate(dataloader):
        if i >= max_batches:
            break

        images = images.to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()

        _ = model(images)

        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.time()

        batch_time = end - start
        total_time += batch_time
        total_images += images.size(0)
        num_batches += 1

    if num_batches == 0:
        return {
            "device_name": device_name,
            "mean_latency_ms": None,
            "fps": None,
            "throughput": None,
            "vram_used_mb": None,
            "vram_total_mb": None,
            "flops_g": None,
            "params_m": None,
            "power_w": None,
            "efficiency_fps_w": None,
            "batch_size": getattr(dataloader, "batch_size", None),
        }

    mean_latency_ms = (total_time / num_batches) * 1000.0
    fps = total_images / total_time if total_time > 0 else None
    throughput = fps  # para clasificación, fps ~ samples/s

    if device.type == "cuda":
        vram_used_mb = torch.cuda.max_memory_allocated() / (1024**2)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    else:
        vram_used_mb = None
        total_mem = None

    # Parámetros del modelo
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    flops_g = None  # lo dejamos como None (se puede integrar ptflops más adelante)

    benchmark = {
        "device_name": device_name,
        "batch_size": dataloader.batch_size,
        "mean_latency_ms": mean_latency_ms,
        "fps": fps,
        "throughput": throughput,
        "vram_used_mb": vram_used_mb,
        "vram_total_mb": total_mem,
        "flops_g": flops_g,
        "params_m": params_m,
        "power_w": None,
        "efficiency_fps_w": None,
    }
    return benchmark


# ------------------------------------------------------------
# (el resto del archivo: parse_args, main, etc.)
# ------------------------------------------------------------

# TODO: deja tal cual la parte de parse_args(), main(), etc. que ya tienes.
#       Solo necesitábamos modificar train_one_epoch y evaluate.
