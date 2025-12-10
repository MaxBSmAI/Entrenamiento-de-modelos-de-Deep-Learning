"""
train_retinanet_coco.py

Entrena RetinaNet (ResNet50-FPN v2) en COCO Detection usando HuggingFace.

Incluye:
- max_steps_per_epoch (Estrategia B)
- iterador global para entrenamiento parcial
- Early stopping
- mAP@0.5 simplificado
- Benchmark de inferencia
- Guardado json con device_name, metrics, benchmark

Ejecución recomendada:
    cd /workspace
    export PYTHONPATH=./src
    python src/detection/train_retinanet_coco.py --streaming True --max_steps_per_epoch 100
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Iterator, Optional, Tuple
import sys

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# ------------------------------------------------------------
# Añadir src al sys.path
# ------------------------------------------------------------
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Modelos torchvision (detección)
from torchvision.models.detection import (
    retinanet_resnet50_fpn_v2,
    RetinaNet_ResNet50_FPN_V2_Weights,
)

# Dataloaders COCO (HuggingFace)
from data.dataloaders import get_coco_detection_dataloaders


# ============================================================
# Utils
# ============================================================

def set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in ("yes","y","true","t","1"):
        return True
    if v in ("no","n","false","f","0"):
        return False
    raise argparse.ArgumentTypeError(f"Valor booleano inválido: {v}")


def build_model(num_classes: int, pretrained: bool = True):
    """
    RetinaNet ResNet50-FPN v2.
    """
    if pretrained:
        weights = RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
        model = retinanet_resnet50_fpn_v2(weights=weights)   # num_classes=91
    else:
        model = retinanet_resnet50_fpn_v2(weights=None, num_classes=num_classes)
    return model


def to_device_detection_batch(images, targets, device):
    imgs = [i.to(device) for i in images]
    tgts = [{k: v.to(device) for k, v in t.items()} for t in targets]
    return imgs, tgts


# ============================================================
# mAP@0.5
# ============================================================

def box_iou(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.size(0), boxes2.size(0)))

    area1 = (boxes1[:,2] - boxes1[:,0]) * (boxes1[:,3] - boxes1[:,1])
    area2 = (boxes2[:,2] - boxes2[:,0]) * (boxes2[:,3] - boxes2[:,1])

    lt = torch.max(boxes1[:,None,:2], boxes2[:,:2])
    rb = torch.min(boxes1[:,None,2:], boxes2[:,2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:,:,0] * wh[:,:,1]
    union = area1[:,None] + area2 - inter

    return inter / union.clamp(min=1e-6)


def compute_map_50(predictions, ground_truths, num_classes:int):
    aps = []

    for cls in range(num_classes):
        cls_preds = []
        cls_gts = {}

        for img_idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            mask_p = pred["labels"] == cls
            if mask_p.any():
                p_boxes = pred["boxes"][mask_p]
                p_scores = pred["scores"][mask_p]
                for b, s in zip(p_boxes, p_scores):
                    cls_preds.append({
                        "img_id": img_idx,
                        "box": b,
                        "score": float(s),
                    })

            mask_g = gt["labels"] == cls
            if mask_g.any():
                boxes = gt["boxes"][mask_g]
                cls_gts.setdefault(img_idx, [])
                for b in boxes:
                    cls_gts[img_idx].append({"box": b, "matched": False})

        if len(cls_preds) == 0:
            continue

        cls_preds.sort(key=lambda x: x["score"], reverse=True)

        tp = []
        fp = []

        for pred in cls_preds:
            img_id = pred["img_id"]
            pred_box = pred["box"].unsqueeze(0)

            gts = cls_gts.get(img_id, [])
            if not gts:
                tp.append(0.0)
                fp.append(1.0)
                continue

            gt_boxes = torch.stack([g["box"] for g in gts], dim=0)
            ious = box_iou(pred_box, gt_boxes).squeeze(0)

            best_iou, best_idx = ious.max(0)
            if best_iou >= 0.5 and not gts[best_idx]["matched"]:
                gts[best_idx]["matched"] = True
                tp.append(1.0)
                fp.append(0.0)
            else:
                tp.append(0.0)
                fp.append(1.0)

        tp = torch.tensor(tp)
        fp = torch.tensor(fp)

        gt_total = sum(len(v) for v in cls_gts.values())
        if gt_total == 0:
            continue

        cum_tp = torch.cumsum(tp,0)
        cum_fp = torch.cumsum(fp,0)

        recall = cum_tp / gt_total
        precision = cum_tp / (cum_tp + cum_fp).clamp(min=1e-6)

        ap = 0.0
        prev_r = 0.0
        for p,r in zip(precision, recall):
            ap += float(p) * max(float(r)-prev_r,0)
            prev_r = float(r)

        aps.append(ap)

    return float(sum(aps)/len(aps)) if aps else 0.0


# ============================================================
# TRAINING — ESTRATEGIA B
# ============================================================

def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer,
    device,
    max_steps_per_epoch: Optional[int] = None,
    train_iter: Optional[Iterator] = None
):
    """
    - max_steps_per_epoch = None → usa todo el loader (modo clásico)
    - max_steps_per_epoch = N    → consume N batches usando iterador global
    """
    model.train()
    loss_sum = 0.0

    # Caso clásico → recorre todo el DataLoader
    if max_steps_per_epoch is None:
        batches = 0
        for images, targets in loader:
            imgs, tgts = to_device_detection_batch(images, targets, device)

            losses_dict = model(imgs, tgts)
            loss = sum(losses_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            batches += 1

        return loss_sum / max(1,batches), None

    # Estrategia B → N batches por época
    if train_iter is None:
        train_iter = iter(loader)

    steps = 0
    while steps < max_steps_per_epoch:
        try:
            images, targets = next(train_iter)
        except StopIteration:
            train_iter = iter(loader)
            images, targets = next(train_iter)

        imgs, tgts = to_device_detection_batch(images, targets, device)

        losses_dict = model(imgs, tgts)
        loss = sum(losses_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        steps += 1

    return loss_sum / max(1, steps), train_iter


# ============================================================
# evaluación
# ============================================================

@torch.no_grad()
def evaluate_loss(model, loader: DataLoader, device: torch.device):
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
def evaluate_map(model, loader: DataLoader, device: torch.device, num_classes:int, max_batches=50):
    model.eval()
    preds, gts = [], []

    for i, (images, targets) in enumerate(loader):
        if i >= max_batches:
            break

        outputs = model([img.to(device) for img in images])

        for o,t in zip(outputs, targets):
            preds.append({
                "boxes": o["boxes"].cpu(),
                "scores": o["scores"].cpu(),
                "labels": o["labels"].cpu(),
            })
            gts.append({
                "boxes": t["boxes"].cpu(),
                "labels": t["labels"].cpu(),
            })

    return {"map_50": compute_map_50(preds, gts, num_classes)}


# ============================================================
# early stopping
# ============================================================

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.counter = 0
        self.should_stop = False

    def step(self, val):
        if self.best is None or (self.best - val) > self.min_delta:
            self.best = val
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


# ============================================================
# benchmark detección
# ============================================================

@torch.no_grad()
def benchmark_detection_model(
    model,
    loader: DataLoader,
    device: torch.device,
    max_batches=20
):
    model.eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        dev_name = torch.cuda.get_device_name(0)
    else:
        dev_name = "CPU"

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

    if batches == 0 or total_time == 0:
        return {
            "device_name": dev_name,
            "batch_size": loader.batch_size,
            "mean_latency_ms": None,
            "fps": None,
            "throughput": None,
            "vram_used_mb": None,
            "vram_total_mb": None,
        }

    latency = (total_time / batches) * 1000.0
    fps = total_images / total_time

    vram_used = torch.cuda.max_memory_allocated()/(1024**2) if device.type=="cuda" else None
    total_vram = torch.cuda.get_device_properties(0).total_memory/(1024**2) if device.type=="cuda" else None

    params_m = sum(p.numel() for p in model.parameters())/1e6

    return {
        "device_name": dev_name,
        "batch_size": loader.batch_size,
        "mean_latency_ms": latency,
        "fps": fps,
        "throughput": fps,
        "vram_used_mb": vram_used,
        "vram_total_mb": total_vram,
        "params_m": params_m,
    }


# ============================================================
# parser
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser("Train RetinaNet on COCO")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--run_name", type=str, default="retinanet_coco")
    parser.add_argument("--no_pretrained", action="store_true")

    parser.add_argument(
        "--streaming",
        type=str2bool,
        default=True,
        help="True = HuggingFace streaming; False = map-style",
    )

    parser.add_argument(
        "--max_steps_per_epoch",
        type=int,
        default=1000,
        help="Número de batches por época lógica (<=0 usa todo el DataLoader)",
    )
    return parser.parse_args()


# ============================================================
# main
# ============================================================

def main():
    args = parse_args()
    set_seed(42)
    device = get_device()

    print(f"[INFO] Dispositivo: {device}")
    print(f"[INFO] streaming={args.streaming}")
    print(f"[INFO] max_steps_per_epoch={args.max_steps_per_epoch}")

    if args.max_steps_per_epoch is not None and args.max_steps_per_epoch <= 0:
        args.max_steps_per_epoch = None

    # carpetas
    project_root = Path(__file__).resolve().parents[2]
    result_root = project_root / "result" / "detection"
    run_dir = result_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / f"{args.run_name}_best.pth"
    metrics_path = run_dir / f"{args.run_name}_metrics.json"

    # dataloaders
    train_loader, val_loader, num_classes = get_coco_detection_dataloaders(
        batch_size=args.batch_size,
        streaming=args.streaming,
    )

    print(f"[INFO] Número de clases COCO: {num_classes}")

    # modelo
    model = build_model(num_classes, pretrained=not args.no_pretrained)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    early = EarlyStopping(args.patience, args.min_delta)

    best_val_loss = float("inf")
    best_epoch = -1

    history = {"train_loss": [], "val_loss": []}

    # Estrategia B → iterador global
    train_iter: Optional[Iterator] = None
    if args.max_steps_per_epoch is not None:
        train_iter = iter(train_loader)

    # entrenamiento
    for epoch in range(1, args.epochs + 1):
        print(f"\n[Epoch {epoch}/{args.epochs}]")

        if args.max_steps_per_epoch is None:
            train_loss, _ = train_one_epoch(
                model, train_loader, optimizer, device,
                max_steps_per_epoch=None,
                train_iter=None
            )
        else:
            train_loss, train_iter = train_one_epoch(
                model, train_loader, optimizer, device,
                max_steps_per_epoch=args.max_steps_per_epoch,
                train_iter=train_iter
            )

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
            print(f"[INFO] Mejor modelo guardado en {model_path}")
        else:
            early.step(val_loss)
            print(f"[INFO] EarlyStopping {early.counter}/{args.patience}")

        if early.should_stop:
            print("[INFO] Early stopping activado.")
            break

    # cargar mejor modelo
    model.load_state_dict(torch.load(model_path, map_location=device))

    # test
    test_loss = evaluate_loss(model, val_loader, device)
    test_metrics = evaluate_map(model, val_loader, device, num_classes)

    # benchmark
    benchmark = benchmark_detection_model(model, val_loader, device)

    # guardar JSON
    metrics = {
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
        "device_name": benchmark.get("device_name"),
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"[OK] Métricas guardadas en {metrics_path}")


if __name__ == "__main__":
    main()
