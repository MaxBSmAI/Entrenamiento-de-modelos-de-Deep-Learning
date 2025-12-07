# src/data/dataloaders/coco_detection.py

"""
Dataloader para el dataset:
    detection-datasets/coco   (HuggingFace)

Características clave:
- NO necesitas descargar COCO manualmente en tu máquina local.
- Por defecto usa "streaming=True" para NO almacenar el dataset completo en disco.
- Permite usar el mismo batch_size = 16 para comparar rendimiento entre GPUs.

El formato de salida es compatible con:
- Faster R-CNN / RetinaNet / Mask R-CNN
- Otros modelos de detección de torchvision que esperen:
    imágenes: lista[Tensor[C,H,W]]
    targets:  lista[dict(boxes, labels)]
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Any

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
from datasets import load_dataset
from torchvision import transforms
from PIL import Image


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

        # En detection-datasets/coco, bbox ya está en formato [x_min, y_min, x_max, y_max]
        bboxes = ex["objects"]["bbox"]      # [N, 4]
        labels = ex["objects"]["category"]  # [N]

        boxes = torch.tensor(bboxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

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

            boxes = torch.tensor(bboxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

            target = {"boxes": boxes, "labels": labels}
            yield img, target


# ============================================================
#  TRANSFORMACIONES Y COLLATE
# ============================================================


def build_default_transform():
    """
    Transformación estándar para detección.

    OJO:
    - No redimensionamos aquí para no desalinear las bounding boxes.
    - Faster R-CNN y otros modelos de torchvision aceptan tamaños variables.
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )


def collate_fn_coco(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]):
    """
    collate_fn para detección:

    - imágenes: Tensor [B, 3, H, W]
    - targets:  lista de diccionarios
    """
    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    images = torch.stack(images, dim=0)
    return images, targets


# ============================================================
#  FUNCIÓN PRINCIPAL DEL DATALOADER
# ============================================================


def get_coco_detection_dataloaders(
    batch_size: int = 16,
    streaming: bool = True,
    num_workers: int = 2,
):
    """
    Retorna:
        train_loader, val_loader, num_classes

    Parámetros
    ----------
    batch_size : int
        Tamaño de batch para todos los modelos de detección.
    streaming : bool
        True  → Usa IterableDataset desde HuggingFace (no descarga todo).
        False → Usa Dataset map-style (requiere almacenamiento en el contenedor).
    num_workers : int
        Número de workers para DataLoader en modo no streaming.
    """
    transform = build_default_transform()

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
            num_workers=0,
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
        batch_size=8,
        streaming=True,
    )

    print("Número de clases:", num_classes)
    images, targets = next(iter(train_loader))
    print("Batch imágenes:", images.shape)
    print("Ejemplo target keys:", targets[0].keys())
    print("Boxes shape:", targets[0]["boxes"].shape)
    print("Labels shape:", targets[0]["labels"].shape)
