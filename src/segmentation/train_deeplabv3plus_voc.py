"""
train_deeplabv3plus_voc.py

Entrena un modelo DeepLabV3+ (sin preentrenado) en VOC Segmentation y guarda:

- El mejor modelo según val_loss (early stopping).
- Un archivo JSON con:
    * test_loss
    * test_metrics: miou, dice
    * benchmark: latencia, fps, throughput, VRAM, etc.

Soporta:
    --streaming True/False  (para coherencia con el resto; lo maneja el dataloader)

Salida principal en:
    result/segmentation/deeplabv3plus_voc/

Ejecución recomendada desde la raíz del proyecto:

    cd /workspace
    export PYTHONPATH=./src
    python src/segmentation/train_deeplabv3plus_voc.py --streaming True
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

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

import timm  # DeepLabV3+ de timm

from data.dataloaders import get_voc_segmentation_dataloaders


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


def build_model(num_classes: int) -> nn.Module:
    """
    Crea un DeepLabV3+ SIN PREENTRENADO (pretrained=False) usando timm.
    Ajusta num_classes para la cabeza de segmentación.
    """
    model = timm.create_model(
        "deeplabv3plus_resnet50",
        pretrained=False,        # 🔴 SIN PREENTRENADO, como pediste
        num_classes=num_classes  # número de clases de VOC (ej. 21 con fondo)
    )
    return model


# ============================================================
# Métricas de segmentación: mIoU y Dice
# ============================================================

def _fast_hist(true: torch.Tensor, pred: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Matriz de confusión [num_classes, num_classes]
    true, pred: tensores 1D de etiquetas válidas (sin ignore_index).
    """
    k = (true >= 0) & (true < num_classes)
    inds = num_classes * true[k] + pred[k]
    hist = torch.bincount(
        inds,
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes).float()
    return hist


def compute_segmentation_metrics_from_hist(
    hist: torch.Tensor,
) -> Dict[str, float]:
    """
    A partir de la matriz de confusión acumulada, computa mIoU y Dice.
    """
    # intersección por clase = diagonal
    intersection = torch.diag(hist)
    # union = sum filas + sum columnas - diagonal
    union = hist.sum(1) + hist.sum(0) - intersection
    iou = intersection / union.clamp(min=1e-6)

    # ignoramos clases que nunca aparecieron (union=0)
    valid = union > 0
    if valid.sum() == 0:
        miou = 0.0
    else:
        miou = float(iou[valid].mean().item())

    # Dice por clase: 2*inter / (sum filas + sum columnas)
    dice = 2 * intersection / (hist.sum(1) + hist.sum(0)).clamp(min=1e-6)
    if valid.sum() == 0:
        dice_mean = 0.0
    else:
        dice_mean = float(dice[valid].mean().item())

    return {
        "miou": miou,
        "dice": dice_mean,
    }


# ============================================================
# Loops de entrenamiento y evaluación
# ============================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_classes: int,
    ignore_index: int = 255,
) -> Tuple[float, Dict[str, float]]:
    """
    Entrena una época de segmentación.

    Devuelve:
        - loss promedio
        - métricas (miou, dice) aproximadas sobre el conjunto de entrenamiento.
    """
    model.train()
    running_loss = 0.0
    n_batches = 0

    hist = torch.zeros(num_classes, num_classes, device=device)

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)  # [B, C, H, W] o dict; asumimos tensor
        if isinstance(outputs, dict) and "out" in outputs:
            outputs = outputs["out"]

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

        # Métricas: mIoU/Dice aproximada en train
        preds = outputs.argmax(dim=1)  # [B, H, W]
        valid = masks != ignore_index
        if valid.any():
            true = masks[valid].view(-1)
            pred = preds[valid].view(-1)
            hist += _fast_hist(true, pred, num_classes)

    if n_batches == 0:
        avg_loss = 0.0
        metrics = {"miou": 0.0, "dice": 0.0}
    else:
        avg_loss = running_loss / n_batches
        metrics = compute_segmentation_metrics_from_hist(hist)

    return avg_loss, metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    ignore_index: int = 255,
) -> Tuple[float, Dict[str, float]]:
    """
    Evalúa en validación o test:
        - loss promedio
        - métricas mIoU y Dice
    """
    model.eval()
    running_loss = 0.0
    n_batches = 0

    hist = torch.zeros(num_classes, num_classes, device=device)

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        if isinstance(outputs, dict) and "out" in outputs:
            outputs = outputs["out"]

        loss = criterion(outputs, masks)
        running_loss += loss.item()
        n_batches += 1

        preds = outputs.argmax(dim=1)
        valid = masks != ignore_index
        if valid.any():
            true = masks[valid].view(-1)
            pred = preds[valid].view(-1)
            hist += _fast_hist(true, pred, num_classes)

    if n_batches == 0:
        avg_loss = 0.0
        metrics = {"miou": 0.0, "dice": 0.0}
    else:
        avg_loss = running_loss / n_batches
        metrics = compute_segmentation_metrics_from_hist(hist)

    return avg_loss, metrics


# ============================================================
# Early Stopping
# ============================================================

class EarlyStopping:
    """
    Early stopping basado en la métrica de validación (val_loss).
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
def benchmark_segmentation_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int = 20,
) -> Dict[str, Any]:
    """
    Mide latencia promedio, FPS (imágenes/s), throughput y VRAM
    usando hasta 'max_batches' del dataloader.
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

        outputs = model(images)
        if isinstance(outputs, dict) and "out" in outputs:
            _ = outputs["out"]

        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.time()

        batch_time = end - start
        total_time += batch_time
        total_images += images.size(0)
        num_batches += 1

    if num_batches == 0 or total_time == 0.0:
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
    fps = total_images / total_time
    throughput = fps

    if device.type == "cuda":
        vram_used_mb = torch.cuda.max_memory_allocated() / (1024**2)
        total_mem_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    else:
        vram_used_mb = None
        total_mem_mb = None

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    flops_g = None  # se puede integrar ptflops más adelante

    return {
        "device_name": device_name,
        "batch_size": dataloader.batch_size,
        "mean_latency_ms": mean_latency_ms,
        "fps": fps,
        "throughput": throughput,
        "vram_used_mb": vram_used_mb,
        "vram_total_mb": total_mem_mb,
        "flops_g": flops_g,
        "params_m": params_m,
        "power_w": None,
        "efficiency_fps_w": None,
    }


# ============================================================
# Parser (incluye streaming, coherente con otros scripts)
# ============================================================

def str2bool(v) -> bool:
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in ("yes", "y", "true", "t", "1"):
        return True
    if v in ("no", "n", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Valor booleano inválido: {v}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DeepLabV3+ on VOC (sin preentrenado)")

    parser.add_argument("--epochs", type=int, default=50, help="Número máximo de épocas")
    parser.add_argument("--batch_size", type=int, default=8, help="Tamaño de batch")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=5, help="Patience de early stopping")
    parser.add_argument("--min_delta", type=float, default=0.0, help="Mejora mínima en val_loss")
    parser.add_argument("--run_name", type=str, default="deeplabv3plus_voc", help="Nombre de la corrida")

    parser.add_argument(
        "--streaming",
        type=str2bool,
        default=True,
        help="True=usa streaming desde HuggingFace (si el dataloader lo soporta); False=dataset local",
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
    print(f"[INFO] streaming={args.streaming}")

    # Rutas
    project_root = Path(__file__).resolve().parents[2]  # .../workspace
    result_root = project_root / "result" / "segmentation"
    run_dir = result_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / f"{args.run_name}_best.pth"
    metrics_path = run_dir / f"{args.run_name}_metrics.json"

    # DataLoaders (VOC)
    print(f"[INFO] Cargando VOC Segmentation (streaming={args.streaming})...")
    train_loader, val_loader, test_loader, num_classes = get_voc_segmentation_dataloaders(
        batch_size=args.batch_size,
        streaming=args.streaming,
    )
    print(f"[INFO] Número de clases: {num_classes}")

    # Modelo DeepLabV3+ SIN preentrenado
    model = build_model(num_classes=num_classes)
    model.to(device)

    # Loss: CrossEntropy con ignore_index típico de VOC (=255)
    ignore_index = 255
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    optimizer = optim.AdamW(
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
        "train_miou": [],
        "train_dice": [],
        "val_loss": [],
        "val_miou": [],
        "val_dice": [],
    }

    # --------------------------------------------------------
    # Loop de entrenamiento
    # --------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        print(f"\n[Epoch {epoch}/{args.epochs}]")

        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, num_classes, ignore_index
        )
        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, device, num_classes, ignore_index
        )

        history["train_loss"].append(train_loss)
        history["train_miou"].append(train_metrics["miou"])
        history["train_dice"].append(train_metrics["dice"])
        history["val_loss"].append(val_loss)
        history["val_miou"].append(val_metrics["miou"])
        history["val_dice"].append(val_metrics["dice"])

        print(
            f"  Train Loss: {train_loss:.4f} | mIoU: {train_metrics['miou']:.4f} | Dice: {train_metrics['dice']:.4f}"
        )
        print(
            f"  Val   Loss: {val_loss:.4f} | mIoU: {val_metrics['miou']:.4f} | Dice: {val_metrics['dice']:.4f}"
        )

        scheduler.step()

        # Early stopping por val_loss
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

    test_loss, test_metrics = evaluate(
        model, test_loader, criterion, device, num_classes, ignore_index
    )

    print(
        f"[TEST] Loss: {test_loss:.4f} | mIoU: {test_metrics['miou']:.4f} | "
        f"Dice: {test_metrics['dice']:.4f}"
    )

    # --------------------------------------------------------
    # Benchmark de inferencia
    # --------------------------------------------------------
    print("\n[INFO] Ejecutando benchmark de inferencia...")
    benchmark = benchmark_segmentation_model(
        model, test_loader, device, max_batches=20
    )
    if benchmark["fps"] is not None:
        print(
            f"[BENCHMARK] Device: {benchmark['device_name']} | "
            f"Latencia media (ms): {benchmark['mean_latency_ms']:.3f} | "
            f"FPS: {benchmark['fps']:.2f}"
        )
    else:
        print("[BENCHMARK] sin datos")

    # --------------------------------------------------------
    # Guardar métricas en JSON (compatible con segmentation_results.py)
    # --------------------------------------------------------
    metrics_dict: Dict[str, Any] = {
        "model_name": args.run_name,
        "task": "segmentation",
        "dataset": "voc_segmentation",
        "num_classes": num_classes,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "train_history": history,
        "test_loss": float(test_loss),
        "test_metrics": {
            "miou": float(test_metrics["miou"]),
            "dice": float(test_metrics["dice"]),
        },
        "benchmark": benchmark,
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)

    print(f"[INFO] Métricas guardadas en: {metrics_path}")


if __name__ == "__main__":
    main()
