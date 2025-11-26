# src/data/dataloaders/coco_detection.py

"""
Dataloader para el dataset:
    detection-datasets/coco   (HuggingFace)

Características clave:
- NO necesitas descargar COCO manualmente en tu notebook.
- Por defecto usa "streaming=True" para NO almacenar el dataset completo en disco.
- Permite usar el mismo batch size = 8 para comparar el rendimiento entre
  RTX 4080 y la A100 del supercomputador.

El formato de salida es compatible con:
- DETR
- YOLO (previa conversión de cajas si se necesita)
- Faster R-CNN / RetinaNet / Mask R-CNN
"""

from datasets import load_dataset
from torch.utils.data import Dataset, IterableDataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch


# ============================================================
#  DATASETS AUXILIARES
# ============================================================

class CocoDetectionMapDataset(Dataset):
    """
    Versión MAP-STYLE:
    - Requiere cachear los datos dentro del contenedor Docker.
    - NO se descargan en tu notebook.
    """
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        ex = self.hf_dataset[idx]

        img = ex["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        bboxes = ex["objects"]["bbox"]        # [N, 4]
        labels = ex["objects"]["category"]    # [N]

        boxes = torch.tensor(bboxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}
        return img, target


class CocoDetectionIterableDataset(IterableDataset):
    """
    Versión STREAMING:
    - NO guarda el dataset completo en disco.
    - Es ideal para tu caso (no descargar COCO al notebook).
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

def build_default_transform(img_size: int = 640):
    """
    Transformación estándar para detección.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])


def collate_fn_coco(batch):
    """
    collate_fn para detección:
      - imágenes: Tensor [B, 3, H, W]
      - targets: lista de diccionarios
    """
    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    images = torch.stack(images, dim=0)
    return images, targets


# ============================================================
#  FUNCIÓN PRINCIPAL DEL DATALOADER
# ============================================================

def get_coco_detection_dataloaders(
    batch_size: int = 8,      # <--- BATCH UNIVERSAL
    img_size: int = 640,
    num_workers: int = 4,
    streaming: bool = True,   # <--- NO DOWNLOAD en notebook
):
    """
    Retorna:
        train_loader, val_loader, num_classes

    Parámetros:
    -----------
    streaming=True:
        NO descarga el dataset completo.
        Usará IterableDataset → num_workers=0.
    """
    transform = build_default_transform(img_size=img_size)

    if streaming:
        # -------------------------
        # STREAMING MODE
        # -------------------------
        hf_train = load_dataset("detection-datasets/coco", split="train", streaming=True)
        hf_val   = load_dataset("detection-datasets/coco", split="val",   streaming=True)

        train_dataset = CocoDetectionIterableDataset(hf_train, transform)
        val_dataset   = CocoDetectionIterableDataset(hf_val,   transform)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, num_workers=0,
            collate_fn=collate_fn_coco
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, num_workers=0,
            collate_fn=collate_fn_coco
        )

    else:
        # -------------------------
        # MAP-STYLE (cache en contenedor)
        # -------------------------
        hf_train = load_dataset("detection-datasets/coco", split="train")
        hf_val   = load_dataset("detection-datasets/coco", split="val")

        train_dataset = CocoDetectionMapDataset(hf_train, transform)
        val_dataset   = CocoDetectionMapDataset(hf_val,   transform)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn_coco,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn_coco,
        )

    # COCO tiene 80 clases "thing"
    num_classes = 80

    return train_loader, val_loader, num_classes


# ============================================================
#  PRUEBA INTERNA DEL DATALOADER
# ============================================================

if __name__ == "__main__":
    train_loader, val_loader, num_classes = get_coco_detection_dataloaders(
        batch_size=8,
        img_size=640,
        streaming=True
    )

    print("Número de clases:", num_classes)
    images, targets = next(iter(train_loader))
    print("Batch imágenes:", images.shape)
    print("Ejemplo target keys:", targets[0].keys())
    print("Boxes shape:", targets[0]["boxes"].shape)
    print("Labels shape:", targets[0]["labels"].shape)
