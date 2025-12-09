# src/data/dataloaders/coco_detection.py

"""
Dataloader para el dataset:
    detection-datasets/coco   (HuggingFace)

Características clave:
- NO necesitas descargar COCO manualmente en tu máquina local.
- Por defecto usa "streaming=True" para NO almacenar el dataset completo en disco.

El formato de salida es compatible con:
- Faster R-CNN / RetinaNet / Mask R-CNN de torchvision:
    imágenes: list[Tensor[C,H,W]]
    targets:  list[dict(boxes, labels)]
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Any

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
from datasets import load_dataset
from torchvision import transforms
from PIL import Image


# ============================================================
#  FUNCIONES AUXILIARES
# ============================================================

def _sanitize_boxes_and_labels(
    bboxes: Any,
    labels: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Recibe bboxes y labels (listas o tensores) y:

    - Convierte a tensores.
    - Asegura shape (N, 4).
    - Filtra cajas degeneradas:
        * x_max > x_min
        * y_max > y_min
        * valores finitos (no NaN / Inf)
    """
    boxes = torch.as_tensor(bboxes, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.int64)

    if boxes.ndim == 1:
        # Caso raro: una sola caja [4] -> [1, 4]
        boxes = boxes.unsqueeze(0)

    if boxes.numel() == 0:
        # Sin cajas, devolvemos tensores vacíos consistentes
        return boxes.reshape(0, 4), labels.reshape(0)

    # Se asume formato [x_min, y_min, x_max, y_max]
    x_min = boxes[:, 0]
    y_min = boxes[:, 1]
    x_max = boxes[:, 2]
    y_max = boxes[:, 3]

    valid = (x_max > x_min) & (y_max > y_min)

    # Por seguridad, filtramos también cajas con valores no finitos
    finite_mask = torch.isfinite(boxes).all(dim=1)
    valid = valid & finite_mask

    boxes = boxes[valid]
    labels = labels[valid]

    return boxes, labels


# ============================================================
#  DATASETS AUXILIARES
# ============================================================


class CocoDetectionMapDataset(Dataset):
    """
    Versión MAP-STYLE:
    - Requiere cachear los datos dentro del contenedor Docker.
    - NO se descargan en el notebook local.
    """
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int):
        ex = self.hf_dataset[idx]

        img = ex["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        bboxes = ex["objects"]["bbox"]      # [N, 4]
        labels = ex["objects"]["category"]  # [N]

        # --- NUEVO: limpieza de cajas inválidas ---
        boxes, labels = _sanitize_boxes_and_labels(bboxes, labels)

        target = {"boxes": boxes, "labels": labels}
        return img, target


class CocoDetectionIterableDataset(IterableDataset):
    """
    Versión STREAMING:
    - NO guarda el dataset completo en disco.
    - Ideal para tu caso con HuggingFace y contenedores.
    """
    def __init__(self, hf_iterable, transform=None):
        self.hf_iterable = hf_iterable
        self.transform = transform

    def __iter__(self):
        for ex in self.hf_iterable:
            img = ex["image"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            img = img.convert("RGB")

            if self.transform:
                img = self.transform(img)

            bboxes = ex["objects"]["bbox"]
            labels = ex["objects"]["category"]

            # --- NUEVO: limpieza de cajas inválidas ---
            boxes, labels = _sanitize_boxes_and_labels(bboxes, labels)

            target = {"boxes": boxes, "labels": labels}
            yield img, target


# ============================================================
#  TRANSFORMACIONES Y COLLATE
# ============================================================


def build_default_transform(img_size: int = 640):
    """
    Transformación estándar para detección.

    Opción A (actual): solo ToTensor → deja tamaños originales (lo más típico
    para Faster R-CNN, que maneja tamaños variables).

    Si quisieras forzar 640x640 en todo, podrías usar:
        transforms.Resize((img_size, img_size))
    antes del ToTensor.
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )


def collate_fn_coco(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]):
    """
    collate_fn para detección (torchvision):

    - imágenes: lista de tensores [3, H_i, W_i]
    - targets:  lista de diccionarios

    NO se hace torch.stack, porque las imágenes pueden tener tamaños distintos.
    """
    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    return images, targets


# ============================================================
#  FUNCIÓN PRINCIPAL DEL DATALOADER
# ============================================================


def get_coco_detection_dataloaders(
    batch_size: int = 4,
    img_size: int = 640,
    streaming: bool = True,
    num_workers: int = 2,
):
    """
    Retorna:
        train_loader, val_loader, num_classes

    batch_size:
        - Para Faster R-CNN en COCO, valores típicos son 2–4 por GPU dependiendo de VRAM.
    """
    transform = build_default_transform(img_size=img_size)

    if streaming:
        # -------------------------
        # STREAMING MODE
        # -------------------------
        hf_train = load_dataset(
            "detection-datasets/coco", split="train", streaming=True
        )
        hf_val = load_dataset(
            "detection-datasets/coco", split="val", streaming=True
        )

        train_dataset = CocoDetectionIterableDataset(hf_train, transform)
        val_dataset = CocoDetectionIterableDataset(hf_val, transform)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=0,        # IterableDataset → num_workers=0
            collate_fn=collate_fn_coco,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=0,
            collate_fn=collate_fn_coco,
        )
    else:
        # -------------------------
        # MAP-STYLE MODE
        # -------------------------
        hf_train = load_dataset(
            "detection-datasets/coco", split="train", streaming=False
        )
        hf_val = load_dataset(
            "detection-datasets/coco", split="val", streaming=False
        )

        train_dataset = CocoDetectionMapDataset(hf_train, transform)
        val_dataset = CocoDetectionMapDataset(hf_val, transform)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            collate_fn=collate_fn_coco,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            collate_fn=collate_fn_coco,
        )

    # COCO estándar: 80 clases
    num_classes = 80

    return train_loader, val_loader, num_classes


if __name__ == "__main__":
    # Pequeña prueba rápida
    train_loader, val_loader, num_classes = get_coco_detection_dataloaders(
        batch_size=2,
        img_size=640,
        streaming=True,
    )

    print("Número de clases:", num_classes)
    images, targets = next(iter(train_loader))
    print("Len batch imágenes:", len(images))
    print("Tamaño imagen 0:", images[0].shape)
    print("Ejemplo target keys:", targets[0].keys())
