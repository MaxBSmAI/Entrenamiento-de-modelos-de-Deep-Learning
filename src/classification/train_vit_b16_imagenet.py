"""
train_vit_b16_imagenet.py

Entrena Vision Transformer ViT-B/16 en Mini-ImageNet.

NUEVO:
- max_steps_per_epoch (igual que Faster R-CNN, RetinaNet y ResNet-50)
- Logging por batch en tiempo real
- Compatibilidad con streaming (IterableDataset)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import sys

# ----------------------------------------------------------------------
# Añadir src al sys.path
# ----------------------------------------------------------------------
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
# ----------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

import numpy as np
import timm

from data.dataloaders import get_miniimagenet_dataloaders
from utils.utils_metrics import per_class_metrics, confusion_matrix_metrics
from utils.utils_plot import (
    plot_train_val_loss,
    plot_accuracy,
    plot_confusion_matrix,
    plot_per_class_metrics,
    plot_benchmark_metrics,
)


# ============================================================
# Utilidades generales
# ============================================================

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device_and_tags():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        raw = torch.cuda.get_device_name(0).lower()

        if "4080" in raw:
            return device, "RTX 4080", "RTX_4080"
        if "4060" in raw:
            return device, "RTX 4060", "RTX_4060"
        if "a100" in raw:
            return device, "A100", "A100"

        return device, raw, raw.replace(" ", "_")
    else:
        return torch.device("cpu"), "CPU", "CPU"


def build_model(num_classes: int, pretrained: bool = True):
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model


# ============================================================
# ENTRENAMIENTO — con max_steps_per_epoch + logging
# ============================================================

def train_one_epoch(
    model,
    dataloader: DataLoader,
    criterion,
    optimizer,
    device,
    max_steps_per_epoch: Optional[int] = None,
):

    LOG_INTERVAL = 10

    model.train()
    running_loss = 0.0
    total_pred, total_true = [], []
    n_samples = 0

    # Para soportar streaming/IterableDataset
    try:
        total_batches = len(dataloader)
    except:
        total_batches = None

    data_iter = iter(dataloader)
    step = 0

    while True:

        if max_steps_per_epoch is not None and step >= max_steps_per_epoch:
            break

        try:
            images, labels = next(data_iter)
        except StopIteration:
            break

        step += 1

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
        total_pred.extend(preds.cpu().tolist())
        total_true.extend(labels.cpu().tolist())

        # Logging por batch
        if (
            step == 1
            or (LOG_INTERVAL and step % LOG_INTERVAL == 0)
            or (max_steps_per_epoch and step == max_steps_per_epoch)
        ):
            if max_steps_per_epoch:
                print(f"    [train] step {step}/{max_steps_per_epoch} - loss={loss.item():.4f}")
            elif total_batches:
                print(f"    [train] step {step}/{total_batches} - loss={loss.item():.4f}")
            else:
                print(f"    [train] step {step} - loss={loss.item():.4f}")

    if n_samples == 0:
        return 0.0, 0.0

    epoch_loss = running_loss / n_samples
    epoch_acc = accuracy_score(total_true, total_pred)

    return epoch_loss, epoch_acc


# ============================================================
# EVALUACIÓN
# ============================================================

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_pred, total_true = [], []
    n_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        bs = images.size(0)
        running_loss += loss.item() * bs
        n_samples += bs

        preds = outputs.argmax(dim=1)
        total_pred.extend(preds.cpu().tolist())
        total_true.extend(labels.cpu().tolist())

    if n_samples == 0:
        return 0.0, {
            "accuracy": 0.0,
            "f1_macro": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
        }

    avg_loss = running_loss / n_samples

    metrics = {
        "accuracy": accuracy_score(total_true, total_pred),
        "f1_macro": f1_score(total_true, total_pred, average="macro"),
        "precision_macro": precision_score(total_true, total_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(total_true, total_pred, average="macro", zero_division=0),
    }
    return avg_loss, metrics


@torch.no_grad()
def get_predictions(model, dataloader, device):
    model.eval()
    y_true, y_pred = [], []
    for images, labels in dataloader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        y_true.extend(labels.tolist())
        y_pred.extend(preds.cpu().tolist())
    return y_true, y_pred


# ============================================================
# EARLY STOPPING
# ============================================================

class EarlyStopping:

    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.counter = 0
        self.should_stop = False

    def step(self, value):
        if self.best is None or (self.best - value) > self.min_delta:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


# ============================================================
# BENCHMARK
# ============================================================

@torch.no_grad()
def benchmark_classification_model(model, loader, device, device_name, max_batches=20):

    model.eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    total_time = 0.0
    total_images = 0
    nb = 0

    for i, (images, _) in enumerate(loader):
        if i >= max_batches:
            break

        images = images.to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        _ = model(images)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()

        total_time += (t1 - t0)
        total_images += images.size(0)
        nb += 1

    if nb == 0:
        return {
            "device_name": device_name,
            "mean_latency_ms": None,
            "fps": None,
            "throughput": None,
            "vram_used_mb": None,
            "vram_total_mb": None,
        }

    latency = (total_time / nb) * 1000.0
    fps = total_images / total_time

    if device.type == "cuda":
        used = torch.cuda.max_memory_allocated() / (1024**2)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    else:
        used = None
        total = None

    return {
        "device_name": device_name,
        "batch_size": loader.batch_size,
        "mean_latency_ms": latency,
        "fps": fps,
        "throughput": fps,
        "vram_used_mb": used,
        "vram_total_mb": total,
        "params_m": sum(p.numel() for p in model.parameters()) / 1e6,
    }


# ============================================================
# PARSER
# ============================================================

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in ("yes", "y", "true", "t", "1"):
        return True
    if v in ("no", "n", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Valor inválido: {v}")


def parse_args():

    p = argparse.ArgumentParser("Train ViT-B/16 on Mini-ImageNet")

    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min_delta", type=float, default=0.0)
    p.add_argument("--run_name", type=str, default="vit_b16_miniimagenet")
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--streaming", type=str2bool, default=True)

    # NUEVO
    p.add_argument(
        "--max_steps_per_epoch",
        type=int,
        default=1000,
        help="Número máximo de batches por época. None = época completa.",
    )

    return p.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()
    set_seed(42)

    device, device_name, device_tag = get_device_and_tags()
    print(f"[INFO] Device: {device_name}")
    print(f"[INFO] max_steps_per_epoch={args.max_steps_per_epoch}")

    project_root = Path(__file__).resolve().parents[2]
    result_root = project_root / "result" / "classification"

    run_dir = result_root / args.run_name / device_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / f"{args.run_name}_best.pth"
    metrics_path = run_dir / f"{args.run_name}_metrics.json"

    # Dataloaders
    print(f"[INFO] Loading Mini-ImageNet (streaming={args.streaming})...")
    train_loader, val_loader, test_loader, num_classes = get_miniimagenet_dataloaders(
        batch_size=args.batch_size,
        streaming=args.streaming,
    )

    # Modelo
    model = build_model(num_classes, pretrained=not args.no_pretrained)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    early = EarlyStopping(args.patience, args.min_delta)

    best_val = float("inf")
    best_epoch = -1

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # =======================================================
    # Training Loop
    # =======================================================
    for epoch in range(1, args.epochs + 1):

        print(f"\n[Epoch {epoch}/{args.epochs}]")

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            max_steps_per_epoch=args.max_steps_per_epoch,
        )
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_metrics["accuracy"])

        print(
            f"  TrainLoss={train_loss:.4f} | TrainAcc={train_acc:.4f} | "
            f"ValLoss={val_loss:.4f} | ValAcc={val_metrics['accuracy']:.4f}"
        )

        scheduler.step()

        # Early stopping
        if val_loss < best_val - args.min_delta:
            best_val = val_loss
            best_epoch = epoch
            early.counter = 0
            torch.save(model.state_dict(), model_path)
            print(f"[INFO] Guardado mejor modelo → {model_path}")
        else:
            early.step(val_loss)
            print(f"[INFO] EarlyStopping {early.counter}/{early.patience}")

        if early.should_stop:
            print("[INFO] Early stopping activado.")
            break

    print(f"[INFO] Mejor época = {best_epoch}, mejor val_loss = {best_val:.4f}")

    # =======================================================
    # Test Final
    # =======================================================
    model.load_state_dict(torch.load(model_path, map_location=device))
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)

    print(
        f"[TEST] Loss={test_loss:.4f} | Acc={test_metrics['accuracy']:.4f} | "
        f"F1={test_metrics['f1_macro']:.4f}"
    )

    # =======================================================
    # Artefactos visuales
    # =======================================================
    plot_train_val_loss(history["train_loss"], history["val_loss"], run_dir / "train_val_loss.png")
    plot_accuracy(history["val_acc"], run_dir / "val_accuracy.png")

    y_true, y_pred = get_predictions(model, test_loader, device)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    per_class = per_class_metrics(y_true, y_pred)
    cm_raw = confusion_matrix_metrics(y_true, y_pred, normalize=None)
    cm_norm = confusion_matrix_metrics(y_true, y_pred, normalize="true")

    json.dump(per_class, open(run_dir / "per_class_metrics.json", "w"), indent=4)
    json.dump(cm_raw, open(run_dir / "confusion_matrix_raw.json", "w"), indent=4)
    json.dump(cm_norm, open(run_dir / "confusion_matrix_normalized.json", "w"), indent=4)

    plot_confusion_matrix(cm_raw, run_dir / "confusion_matrix_raw.png")
    plot_confusion_matrix(cm_norm, run_dir / "confusion_matrix_normalized.png")
    plot_per_class_metrics(per_class, run_dir / "per_class_f1.png")

    # Benchmark
    benchmark = benchmark_classification_model(model, test_loader, device, device_name)
    plot_benchmark_metrics(benchmark, run_dir / "benchmark_summary.png")

    # =======================================================
    # Guardar JSON final
    # =======================================================
    final_metrics = {
        "model_name": args.run_name,
        "task": "classification",
        "dataset": "mini-imagenet",
        "num_classes": num_classes,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "device_name": device_name,
        "device_tag": device_tag,
        "train_history": history,
        "test_loss": float(test_loss),
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "benchmark": benchmark,
    }

    json.dump(final_metrics, open(metrics_path, "w"), indent=4)

    print(f"[INFO] Métricas guardadas en {metrics_path}")
    print(f"[INFO] Artefactos guardados en {run_dir}")


if __name__ == "__main__":
    main()
