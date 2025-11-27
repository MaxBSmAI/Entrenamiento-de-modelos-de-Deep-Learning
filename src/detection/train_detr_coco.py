"""
train_detr_coco.py

Entrena un modelo DETR-ResNet50 en COCO Detection y guarda:

- El mejor modelo según val_loss (early stopping).
- Un archivo JSON con:
    * test_loss
    * test_metrics: map_50, map_50_95 (placeholder simple)
    * benchmark: latencia, fps, throughput, VRAM, etc.

Soporta:
    --streaming True/False   para usar o no streaming desde HuggingFace.

Salida principal en:
    result/detection/detr_coco/

Ejecución recomendada desde la raíz del proyecto:

    cd /workspace
    export PYTHONPATH=./src
    python src/detection/train_detr_coco.py --streaming True
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import sys

# ----------------------------------------------------------------------
# Añadir src al sys.path para que funcionen los imports (data, utils, ...)
# ----------------------------------------------------------------------
SRC_ROOT = Path(__file__).resolve().parents[1]  # .../src
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
# ----------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision.models.detection import detr_resnet50, Detr_ResNet50_Weights

from data.dataloaders import get_coco_detection_dataloaders


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


def build_model(num_classes: int, pretrained: bool = True) -> torch.nn.Module:
    """
    Crea un DETR-ResNet50 con num_classes.
    """
    weights = Detr_ResNet50_Weights.COCO_V1 if pretrained else None
    model = detr_resnet50(weights=weights, num_classes=num_classes)
    return model


# ============================================================
# Funciones auxiliares para detección
# ============================================================

def to_device_detection_batch(
    images: torch.Tensor,
    targets: List[Dict[str, torch.Tensor]],
    device: torch.device,
) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    """
    Convierte batch de imágenes (tensor) + targets (lista de dict)
    al formato esperado por modelos de detección (lista de tensores).
    """
    images_list = [img.to(device) for img in images]
    targets_list: List[Dict[str, torch.Tensor]] = []
    for t in targets:
        targets_list.append(
            {k: v.to(device) for k, v in t.items()}
        )
    return images_list, targets_list


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    IoU entre dos conjuntos de cajas [N,4], [M,4] en formato [x1,y1,x2,y2].
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.size(0), boxes2.size(0)), dtype=torch.float32)

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter
    iou = inter / union.clamp(min=1e-6)
    return iou


def compute_map_at_05_simple(
    preds: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    num_classes: int,
) -> float:
    """
    Cálculo simplificado de mAP@0.5 IoU.
    No es la implementación COCO oficial, pero sirve como métrica comparativa.

    preds / targets: listas de longitud N_imágenes
      - cada elemento:
        {
          "boxes": [Ni, 4],
          "scores": [Ni],
          "labels": [Ni]
        }
    """
    # Agrupamos por clase
    aps: List[float] = []

    for cls in range(1, num_classes + 1):  # asumimos labels en [1, num_classes]
        cls_pred_boxes = []
        cls_pred_scores = []
        cls_gt_boxes = 0

        per_image_data = []

        for p, t in zip(preds, targets):
            # ground truth de esta clase
            t_mask = (t["labels"] == cls)
            gt_boxes = t["boxes"][t_mask]
            num_gt = gt_boxes.size(0)
            cls_gt_boxes += num_gt

            # predicciones de esta clase
            if "scores" not in p:
                # si no hay scores (caso extraño), saltamos
                continue
            p_mask = (p["labels"] == cls)
            pred_boxes = p["boxes"][p_mask]
            pred_scores = p["scores"][p_mask]

            per_image_data.append((pred_boxes, pred_scores, gt_boxes))

        if cls_gt_boxes == 0:
            # No hay GT de esta clase -> no cuenta
            continue

        # Concatenamos predicciones de todas las imágenes
        all_boxes = []
        all_scores = []
        image_indices = []
        img_idx = 0
        for pred_boxes, pred_scores, gt_boxes in per_image_data:
            all_boxes.append(pred_boxes)
            all_scores.append(pred_scores)
            image_indices.extend([img_idx] * pred_boxes.size(0))
            img_idx += 1

        if len(all_boxes) == 0:
            continue

        all_boxes = torch.cat(all_boxes, dim=0)
        all_scores = torch.cat(all_scores, dim=0)
        image_indices = torch.tensor(image_indices, dtype=torch.long)

        # Ordenamos por score descendente
        scores_sorted, order = all_scores.sort(descending=True)
        all_boxes = all_boxes[order]
        image_indices = image_indices[order]

        tp = torch.zeros(len(all_boxes))
        fp = torch.zeros(len(all_boxes))

        # Por cada GT, marcamos si ya fue emparejada
        matched = {i: torch.zeros(per_image_data[i][2].size(0), dtype=torch.bool) for i in range(len(per_image_data))}

        for i in range(len(all_boxes)):
            box = all_boxes[i]
            img_i = int(image_indices[i])
            gt_boxes = per_image_data[img_i][2]

            if gt_boxes.numel() == 0:
                fp[i] = 1.0
                continue

            ious = box_iou(box.unsqueeze(0), gt_boxes).squeeze(0)  # [num_gt]
            max_iou, max_idx = ious.max(dim=0)

            if max_iou >= 0.5 and not matched[img_i][max_idx]:
                tp[i] = 1.0
                matched[img_i][max_idx] = True
            else:
                fp[i] = 1.0

        # Cálculo de AP
        cum_tp = torch.cumsum(tp, dim=0)
        cum_fp = torch.cumsum(fp, dim=0)
        recalls = cum_tp / float(cls_gt_boxes)
        precisions = cum_tp / torch.clamp(cum_tp + cum_fp, min=1e-6)

        # AP como suma de (r_{k} - r_{k-1}) * p_k
        ap = 0.0
        prev_r = 0.0
        for p, r in zip(precisions, recalls):
            ap += float(p) * max(float(r) - prev_r, 0.0)
            prev_r = float(r)
        aps.append(ap)

    if len(aps) == 0:
        return 0.0
    return float(sum(aps) / len(aps))


# ============================================================
# Loops de entrenamiento y evaluación
# ============================================================

def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Entrena una época para DETR.
    Usa las losses que retorna el modelo.
    """
    model.train()
    running_loss = 0.0
    n_batches = 0

    for images, targets in dataloader:
        images_list, targets_list = to_device_detection_batch(images, targets, device)

        loss_dict = model(images_list, targets_list)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

    if n_batches == 0:
        return 0.0
    return running_loss / n_batches


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """
    Evalúa la loss promedio en validación usando las losses del modelo.
    """
    model.train()  # los modelos de detección calculan loss en modo train
    running_loss = 0.0
    n_batches = 0

    for images, targets in dataloader:
        images_list, targets_list = to_device_detection_batch(images, targets, device)
        loss_dict = model(images_list, targets_list)
        loss = sum(loss_dict.values())

        running_loss += loss.item()
        n_batches += 1

    if n_batches == 0:
        return 0.0
    return running_loss / n_batches


@torch.no_grad()
def evaluate_map(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    max_batches: int | None = None,
) -> Dict[str, float]:
    """
    Evalúa mAP @0.5 usando un cálculo simplificado.
    """
    model.eval()

    all_preds: List[Dict[str, torch.Tensor]] = []
    all_targets: List[Dict[str, torch.Tensor]] = []

    for i, (images, targets) in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break

        images_list = [img.to(device) for img in images]
        outputs = model(images_list)  # lista de dicts

        # Pasamos todo a CPU para métrica
        for out, tgt in zip(outputs, targets):
            pred = {
                "boxes": out["boxes"].detach().cpu(),
                "scores": out["scores"].detach().cpu(),
                "labels": out["labels"].detach().cpu(),
            }
            gt = {
                "boxes": tgt["boxes"].detach().cpu(),
                "labels": tgt["labels"].detach().cpu(),
            }
            all_preds.append(pred)
            all_targets.append(gt)

    if len(all_preds) == 0:
        return {"map_50": 0.0, "map_50_95": 0.0}

    map_50 = compute_map_at_05_simple(all_preds, all_targets, num_classes=num_classes)
    # Placeholder: en esta implementación simple no calculamos 0.5:0.95
    map_50_95 = None

    return {
        "map_50": map_50,
        "map_50_95": map_50_95,
    }


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
def benchmark_detection_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int = 20,
) -> Dict[str, Any]:
    """
    Mide latencia promedio por batch, FPS, throughput y VRAM.

    Usa hasta 'max_batches' batches del dataloader.
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

        images_list = [img.to(device) for img in images]

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()

        _ = model(images_list)

        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.time()

        batch_time = end - start
        total_time += batch_time
        total_images += len(images_list)
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
    throughput = fps  # detección: fps ~ imágenes/s

    if device.type == "cuda":
        vram_used_mb = torch.cuda.max_memory_allocated() / (1024**2)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    else:
        vram_used_mb = None
        total_mem = None

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    flops_g = None  # se puede integrar ptflops después

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
# Parser: streaming (str2bool)
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
    parser = argparse.ArgumentParser(description="Train DETR-ResNet50 on COCO Detection")

    parser.add_argument("--epochs", type=int, default=25, help="Número máximo de épocas")
    parser.add_argument("--batch_size", type=int, default=8, help="Tamaño de batch")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=5, help="Patience de early stopping")
    parser.add_argument("--min_delta", type=float, default=0.0, help="Mejora mínima para resetear patience")
    parser.add_argument("--run_name", type=str, default="detr_coco", help="Nombre de la corrida")
    parser.add_argument("--no_pretrained", action="store_true", help="No usar pesos preentrenados")

    parser.add_argument(
        "--streaming",
        type=str2bool,
        default=True,
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
    project_root = Path(__file__).resolve().parents[2]
    result_root = project_root / "result" / "detection"
    run_dir = result_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / f"{args.run_name}_best.pth"
    metrics_path = run_dir / f"{args.run_name}_metrics.json"

    # DataLoaders
    print(f"[INFO] Cargando COCO Detection (streaming={args.streaming})...")
    train_loader, val_loader, test_loader, num_classes = get_coco_detection_dataloaders(
        batch_size=args.batch_size,
        streaming=args.streaming,
    )
    print(f"[INFO] Número de clases: {num_classes}")

    # Modelo
    model = build_model(num_classes=num_classes, pretrained=not args.no_pretrained)
    model.to(device)

    # Optimización
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)

    best_val_loss = float("inf")
    best_epoch = -1

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
    }

    # --------------------------------------------------------
    # Loop de entrenamiento
    # --------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        print(f"\n[Epoch {epoch}/{args.epochs}]")

        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate_loss(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"  Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        scheduler.step()

        # Early stopping
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
    # Cargar mejor modelo y evaluar en test (mAP simple)
    # --------------------------------------------------------
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[INFO] Mejor modelo cargado desde: {model_path}")

    test_loss = evaluate_loss(model, test_loader, device)
    test_metrics = evaluate_map(model, test_loader, device, num_classes=num_classes, max_batches=50)

    print(
        f"[TEST] Loss: {test_loss:.4f} | "
        f"mAP@0.5: {test_metrics['map_50']:.4f}"
    )

    # --------------------------------------------------------
    # Benchmark de inferencia
    # --------------------------------------------------------
    print("\n[INFO] Ejecutando benchmark de inferencia...")
    benchmark = benchmark_detection_model(model, test_loader, device, max_batches=20)
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
        "task": "detection",
        "dataset": "coco_detection",
        "num_classes": num_classes,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "train_history": history,
        "test_loss": float(test_loss),
        "test_metrics": {
            "map_50": float(test_metrics["map_50"]),
            "map_50_95": (
                float(test_metrics["map_50_95"])
                if test_metrics["map_50_95"] is not None
                else None
            ),
        },
        "benchmark": benchmark,
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)

    print(f"[INFO] Métricas guardadas en: {metrics_path}")


if __name__ == "__main__":
    main()
