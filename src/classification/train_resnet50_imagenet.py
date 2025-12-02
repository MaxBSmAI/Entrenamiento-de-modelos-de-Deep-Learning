"""
train_resnet50_imagenet.py

Entrena un modelo ResNet-50 en Mini-ImageNet y guarda:

- El mejor modelo según val_loss (early stopping).
- Un archivo JSON con:
    * test_loss
    * test_metrics: accuracy, f1_macro, precision_macro, recall_macro
    * benchmark: latencia, fps, throughput, VRAM, etc.

Soporta:
    --streaming True/False   para usar o no streaming desde HuggingFace.

Salida principal en:
    result/classification/resnet50_miniimagenet/

Ejecución recomendada desde la raíz del proyecto:

    cd /workspace
    export PYTHONPATH=./src
    python src/classification/train_resnet50_imagenet.py --streaming True
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List

import sys

# ----------------------------------------------------------------------
# AÑADIR src al sys.path para que funcionen los imports (data, utils, ...)
# ----------------------------------------------------------------------
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
from utils.utils_benchmark import throughput_from_latency  # por si lo usas luego


# ============================================================
# Utilidades generales
# ============================================================

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


# ============================================================
# Loop de entrenamiento y evaluación (compatible IterableDataset)
# ============================================================

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

    if n_samples == 0:
        epoch_loss = 0.0
        epoch_acc = 0.0
    else:
        epoch_loss = running_loss / n_samples
        epoch_acc = accuracy_score(all_targets, all_preds)

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


# ============================================================
# Early Stopping
# ============================================================

class EarlyStopping:
    """
    Early stopping basado en la métrica de validación (val_loss).

    Se detiene cuando val_loss deja de mejorar después de 'patience' épocas.
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


# ============================================================
# Benchmark de inferencia
# ============================================================

@torch.no_grad()
def benchmark_classification_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int = 20,
) -> Dict[str, Any]:
    """
    Mide latencia promedio por batch, FPS, throughput y VRAM.

    Usa hasta 'max_batches' batches del dataloader para no tardar demasiado.
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
            "batch_size": getattr(dataloader, "batch_size", None),
            "mean_latency_ms": None,
            "fps": None,
            "throughput": None,
            "vram_used_mb": None,
            "vram_total_mb": None,
            "flops_g": None,
            "params_m": None,
            "power_w": None,
            "efficiency_fps_w": None,
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
    flops_g = None  # se puede integrar ptflops más adelante

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


# ============================================================
# Parser: streaming con type=bool (robusto vía str2bool)
# ============================================================

def str2bool(v) -> bool:
    """
    Convierte cadenas tipo 'true'/'false', '1'/'0', 'yes'/'no' a bool.

    Ejemplos válidos:
        --streaming True
        --streaming False
        --streaming 1
        --streaming 0
        --streaming yes
        --streaming no
    """
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in ("yes", "y", "true", "t", "1"):
        return True
    if v in ("no", "n", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Valor booleano inválido: {v}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet-50 on Mini-ImageNet")

    parser.add_argument("--epochs", type=int, default=30, help="Número máximo de épocas")
    parser.add_argument("--batch_size", type=int, default=8, help="Tamaño de batch")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=5, help="Patience de early stopping")
    parser.add_argument("--min_delta", type=float, default=0.0, help="Mejora mínima para resetear patience")
    parser.add_argument("--run_name", type=str, default="resnet50_miniimagenet", help="Nombre de la corrida")
    parser.add_argument("--no_pretrained", action="store_true", help="No usar pesos preentrenados")

    # Aquí controlas el default: True o False
    parser.add_argument(
        "--streaming",
        type=str2bool,
        default=True,  # <-- puedes cambiar a False si quieres
        help="True=usa streaming desde HuggingFace (IterableDataset); False=usa dataset local/descargado",
    )

    return parser.parse_args()


# ============================================================
# main
# ============================================================

def main() -> None:
    args = parse_args()
    set_seed(42)

    device = get_device()
    print(f"[INFO] Usando dispositivo: {device}")
    print(f"[INFO] Parámetro streaming: {args.streaming}")

    # Rutas
    project_root = Path(__file__).resolve().parents[2]  # .../workspace
    result_root = project_root / "result" / "classification"
    run_dir = result_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / f"{args.run_name}_best.pth"
    metrics_path = run_dir / f"{args.run_name}_metrics.json"

    # DataLoaders
    print(f"[INFO] Cargando Mini-ImageNet (streaming={args.streaming})...")
    train_loader, val_loader, test_loader, num_classes = get_miniimagenet_dataloaders(
        batch_size=args.batch_size,
        streaming=args.streaming,
    )
    print(f"[INFO] Número de clases: {num_classes}")

    # Modelo
    model = build_model(num_classes=num_classes, pretrained=not args.no_pretrained)
    model.to(device)

    # Optimización
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)

    best_val_loss = float("inf")
    best_epoch = -1

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    # --------------------------------------------------------
    # Loop de entrenamiento
    # --------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        print(f"\n[Epoch {epoch}/{args.epochs}]")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

        val_acc = val_metrics["accuracy"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        scheduler.step()

        # Early stopping basado en val_loss
        if val_loss < best_val_loss - args.min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            early_stopping.counter = 0

            torch.save(model.state_dict(), model_path)
            print(f"  [INFO] Mejor val_loss hasta ahora. Modelo guardado en: {model_path}")
        else:
            early_stopping.step(val_loss)
            print(f"  [INFO] EarlyStopping counter: {early_stopping.counter}/{early_stopping.patience}")

        if early_stopping.should_stop:
            print("[INFO] Early stopping activado. Fin del entrenamiento.")
            break

    print(f"\n[INFO] Entrenamiento finalizado. Mejor época: {best_epoch}, mejor val_loss: {best_val_loss:.4f}")

    # --------------------------------------------------------
    # Cargar mejor modelo y evaluar en test
    # --------------------------------------------------------
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[INFO] Mejor modelo cargado desde: {model_path}")

    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
    print(
        f"[TEST] Loss: {test_loss:.4f} | "
        f"Acc: {test_metrics['accuracy']:.4f} | "
        f"F1-macro: {test_metrics['f1_macro']:.4f}"
    )

    # --------------------------------------------------------
    # Benchmark de inferencia
    # --------------------------------------------------------
    print("\n[INFO] Ejecutando benchmark de inferencia...")
    benchmark = benchmark_classification_model(model, test_loader, device, max_batches=20)
    if benchmark["fps"] is not None:
        print(
            f"[BENCHMARK] Device: {benchmark['device_name']} | "
            f"Latencia media (ms): {benchmark['mean_latency_ms']:.3f} | "
            f"FPS: {benchmark['fps']:.2f}"
        )
    else:
        print("[BENCHMARK] sin datos")

    # --------------------------------------------------------
    # Guardar métricas en JSON
    # --------------------------------------------------------
    metrics_dict: Dict[str, Any] = {
        "model_name": args.run_name,
        "task": "classification",
        "dataset": "mini-imagenet",
        "num_classes": num_classes,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "train_history": history,
        "test_loss": float(test_loss),
        "test_metrics": {
            "accuracy": float(test_metrics["accuracy"]),
            "f1_macro": float(test_metrics["f1_macro"]),
            "precision_macro": float(test_metrics["precision_macro"]),
            "recall_macro": float(test_metrics["recall_macro"]),
        },
        "benchmark": benchmark,
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)

    print(f"[INFO] Métricas guardadas en: {metrics_path}")


if __name__ == "__main__":
    main()
