"""
train_Faster_R_CNN.py

Entrena Faster R-CNN (ResNet50-FPN) en COCO Detection usando el dataloader
get_coco_detection_dataloaders.

Salida principal en:
    result/detection/<run_name>/

Guarda:
- Mejor modelo según val_loss (early stopping) → <run_name>_best.pth
- JSON con:
    * test_loss
    * test_metrics: map_50
    * benchmark: latencia, fps, throughput, VRAM, params_m, etc.
- Figura PNG con curva Train vs Val Loss
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any
import sys

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Añadir src al path
SRC_ROOT = Path(__file__).resolve().parents[1]  # .../src
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Modelos torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Dataloaders COCO
from data.dataloaders import get_coco_detection_dataloaders

# Utilidades de gráficos (para guardar curvas de loss)
from utils.utils_plot import plot_train_val_loss


# ============================================================
# utilidades varias
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


def str2bool(v) -> bool:
    """
    Conversor robusto para argumentos booleanos (como en los scripts de clasificación).

    Permite:
        --streaming True
        --streaming False
        --streaming 1 / 0
        --streaming yes / no
    """
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in ("yes", "y", "true", "t", "1"):
        return True
    if v in ("no", "n", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Valor booleano inválido: {v}")


def build_model(num_classes: int, pretrained: bool = True) -> torch.nn.Module:
    """
    Crea Faster R-CNN ResNet50-FPN con num_classes clases.

    - Si pretrained=True: carga pesos COCO_V1 (91 clases) y reemplaza
      la cabeza final por una nueva con num_classes (p.ej. 80).
    - Si pretrained=False: modelo desde cero con num_classes.
    """
    if pretrained:
        weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        model = fasterrcnn_resnet50_fpn(weights=weights)  # NO pasamos num_classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    else:
        model = fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)

    return model


def to_device_detection_batch(images, targets, device):
    imgs = [i.to(device) for i in images]
    tgts = [{k: v.to(device) for k, v in t.items()} for t in targets]
    return imgs, tgts


# ============================================================
# mAP@0.5 simple
# ============================================================

def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.size(0), boxes2.size(0)))

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter

    return inter / union.clamp(min=1e-6)


def compute_map_50(
    preds: list[Dict[str, torch.Tensor]],
    gts: list[Dict[str, torch.Tensor]],
    num_classes: int,
    iou_thresh: float = 0.5,
) -> float:
    """
    Implementación simple de mAP@0.5 (promedio sobre clases).
    """
    aps = []

    for cls in range(1, num_classes):  # asumimos background=0
        cls_scores = []
        cls_tp = []
        cls_fp = []
        gt_total = 0

        for pred, gt in zip(preds, gts):
            # pred
            p_boxes = pred["boxes"]
            p_scores = pred["scores"]
            p_labels = pred["labels"]

            # gt
            g_boxes = gt["boxes"]
            g_labels = gt["labels"]

            p_mask = p_labels == cls
            g_mask = g_labels == cls

            p_boxes_cls = p_boxes[p_mask]
            p_scores_cls = p_scores[p_mask]
            g_boxes_cls = g_boxes[g_mask]

            gt_total += g_boxes_cls.size(0)

            if p_boxes_cls.numel() == 0:
                continue

            if g_boxes_cls.numel() == 0:
                cls_scores.extend(p_scores_cls.tolist())
                cls_tp.extend([0] * p_boxes_cls.size(0))
                cls_fp.extend([1] * p_boxes_cls.size(0))
                continue

            ious = box_iou(p_boxes_cls, g_boxes_cls)
            assigned_gt = set()

            for score, ious_row in sorted(
                zip(p_scores_cls.tolist(), ious), key=lambda x: x[0], reverse=True
            ):
                best_iou, best_gt_idx = ious_row.max(dim=0)
                if best_iou >= iou_thresh and int(best_gt_idx) not in assigned_gt:
                    cls_scores.append(score)
                    cls_tp.append(1)
                    cls_fp.append(0)
                    assigned_gt.add(int(best_gt_idx))
                else:
                    cls_scores.append(score)
                    cls_tp.append(0)
                    cls_fp.append(1)

        if gt_total == 0 or len(cls_scores) == 0:
            aps.append(0.0)
            continue

        # ordenar por score descendente
        indices = sorted(range(len(cls_scores)), key=lambda i: cls_scores[i], reverse=True)
        tp = torch.tensor([cls_tp[i] for i in indices], dtype=torch.float32)
        fp = torch.tensor([cls_fp[i] for i in indices], dtype=torch.float32)

        cum_tp = torch.cumsum(tp, 0)
        cum_fp = torch.cumsum(fp, 0)

        recall = cum_tp / gt_total
        precision = cum_tp / (cum_tp + cum_fp).clamp(min=1e-6)

        ap = 0.0
        prev_r = 0.0
        for p, r in zip(precision, recall):
            ap += float(p) * max(float(r) - prev_r, 0)
            prev_r = float(r)

        aps.append(ap)

    return float(sum(aps) / len(aps)) if aps else 0.0


# ============================================================
# loops de entrenamiento / evaluación
# ============================================================

def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    loss_sum = 0.0
    batches = 0

    for images, targets in loader:
        imgs, tgts = to_device_detection_batch(images, targets, device)

        losses = model(imgs, tgts)
        loss = sum(losses.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        batches += 1

    return loss_sum / max(1, batches)


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    # En detection, el modo "eval" desactiva gradientes, pero
    # necesitamos la rama de entrenamiento para obtener losses
    model.train()
    total = 0.0
    batches = 0
    for images, targets in loader:
        imgs, tgts = to_device_detection_batch(images, targets, device)
        loss = sum(model(imgs, tgts).values())
        total += loss.item()
        batches += 1
    return total / max(1, batches)


@torch.no_grad()
def evaluate_map(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    max_batches: int = 50,
) -> Dict[str, float]:
    model.eval()
    preds, gts = [], []

    for i, (images, targets) in enumerate(loader):
        if i >= max_batches:
            break

        outputs = model([img.to(device) for img in images])

        for o, t in zip(outputs, targets):
            preds.append(
                {
                    "boxes": o["boxes"].cpu(),
                    "scores": o["scores"].cpu(),
                    "labels": o["labels"].cpu(),
                }
            )
            gts.append(
                {
                    "boxes": t["boxes"].cpu(),
                    "labels": t["labels"].cpu(),
                }
            )

    return {"map_50": compute_map_50(preds, gts, num_classes)}


# ============================================================
# early stopping
# ============================================================

class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.counter = 0
        self.should_stop = False

    def step(self, val: float) -> None:
        if self.best is None or (self.best - val) > self.min_delta:
            self.best = val
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


# ============================================================
# benchmark
# ============================================================

@torch.no_grad()
def benchmark_detection_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 20,
) -> Dict[str, Any]:
    model.eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        dev_name = torch.cuda.get_device_name(0)
    else:
        dev_name = "cpu"

    total_time = 0.0
    total_images = 0
    batches = 0

    for i, (images, _) in enumerate(loader):
        if i >= max_batches:
            break

        imgs = [img.to(device) for img in images]

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        _ = model(imgs)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()

        total_time += (t1 - t0)
        total_images += len(imgs)
        batches += 1

    if batches == 0 or total_time == 0.0:
        return {
            "device_name": dev_name,
            "batch_size": loader.batch_size,
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

    latency = (total_time / batches) * 1000.0
    fps = total_images / total_time
    throughput = fps

    if device.type == "cuda":
        vram_used = torch.cuda.max_memory_allocated() / (1024**2)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    else:
        vram_used = None
        total_vram = None

    params_m = sum(p.numel() for p in model.parameters()) / 1e6

    return {
        "device_name": dev_name,
        "batch_size": loader.batch_size,
        "mean_latency_ms": latency,
        "fps": fps,
        "throughput": throughput,
        "vram_used_mb": vram_used,
        "vram_total_mb": total_vram,
        "flops_g": None,
        "params_m": params_m,
        "power_w": None,
        "efficiency_fps_w": None,
    }


# ============================================================
# parser
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train Faster R-CNN on COCO")

    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--run_name", type=str, default="fasterrcnn_coco")
    parser.add_argument("--no_pretrained", action="store_true")

    parser.add_argument(
        "--streaming",
        type=str2bool,
        default=True,
        help="True=usa streaming HuggingFace; False=dataset local/descargado",
    )

    return parser.parse_args()


# ============================================================
# main
# ============================================================

def main() -> None:
    args = parse_args()
    set_seed(42)
    device = get_device()

    print(f"[INFO] Dispositivo: {device}")
    print(f"[INFO] streaming={args.streaming}")

    # carpetas de salida
    project_root = Path(__file__).resolve().parents[2]  # .../workspace
    result_root = project_root / "result" / "detection"
    run_dir = result_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / f"{args.run_name}_best.pth"
    metrics_path = run_dir / f"{args.run_name}_metrics.json"
    loss_fig_path = run_dir / f"{args.run_name}_loss.png"

    # dataloaders
    train_loader, val_loader, num_classes = get_coco_detection_dataloaders(
        batch_size=args.batch_size,
        streaming=args.streaming,
    )

    # test_loader = validación (no hay test real separado)
    test_loader = val_loader

    print(f"[INFO] Número de clases COCO: {num_classes}")

    # modelo
    model = build_model(num_classes, pretrained=not args.no_pretrained)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    early = EarlyStopping(args.patience, args.min_delta)

    best_val_loss = float("inf")
    best_epoch = -1
    history: Dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    # entrenamiento
    for epoch in range(1, args.epochs + 1):
        print(f"\n[Epoch {epoch}/{args.epochs}]")

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate_loss(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        scheduler.step()

        if val_loss < best_val_loss - args.min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            early.counter = 0
            torch.save(model.state_dict(), model_path)
            print(f"  [INFO] Mejor val_loss hasta ahora. Modelo guardado en: {model_path}")
        else:
            early.step(val_loss)
            print(
                f"  [INFO] EarlyStopping counter: "
                f"{early.counter}/{early.patience}"
            )

        if early.should_stop:
            print("[INFO] Early stopping activado.")
            break

    print(
        f"\n[INFO] Entrenamiento finalizado. "
        f"Mejor época: {best_epoch}, mejor val_loss: {best_val_loss:.4f}"
    )

    # cargar mejor modelo
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[INFO] Mejor modelo cargado desde: {model_path}")

    # evaluación test
    test_loss = evaluate_loss(model, test_loader, device)
    test_metrics = evaluate_map(model, test_loader, device, num_classes)

    print(
        f"[TEST] Loss: {test_loss:.4f} | "
        f"mAP@0.5: {test_metrics['map_50']:.4f}"
    )

    # benchmark
    print("\n[INFO] Ejecutando benchmark de inferencia...")
    benchmark = benchmark_detection_model(model, test_loader, device)
    if benchmark["fps"] is not None:
        print(
            f"[BENCHMARK] Device: {benchmark['device_name']} | "
            f"Latencia media (ms): {benchmark['mean_latency_ms']:.3f} | "
            f"FPS: {benchmark['fps']:.2f}"
        )
    else:
        print("[BENCHMARK] sin datos")

    # guardar curva de loss (imagen)
    plot_train_val_loss(
        history["train_loss"],
        history["val_loss"],
        save_path=loss_fig_path,
        title=f"Train vs Val Loss - {args.run_name}",
    )
    print(f"[INFO] Curva de loss guardada en: {loss_fig_path}")

    # guardar JSON
    metrics: Dict[str, Any] = {
        "model_name": args.run_name,
        "task": "detection",
        "dataset": "coco_detection",
        "num_classes": num_classes,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "train_history": history,
        "test_loss": float(test_loss),
        "test_metrics": test_metrics,
        "benchmark": benchmark,
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"[OK] Métricas guardadas en: {metrics_path}")


if __name__ == "__main__":
    main()
