# src/data/dataloaders/mini_imagenet.py

"""
Dataloader oficial para el dataset:
    timm/mini-imagenet  (disponible en HuggingFace)

Características:
- NO necesitas descargar el dataset manualmente.
- Funciona en modo STREAMING (no guarda el dataset completo en disco).
- Permite entrenar en RTX 4080 y A100 con el mismo batch para comparaciones HPC.

Parámetros principales:
- batch_size = 16 (universal y estandarizado)
- streaming = True (default)
"""

from datasets import load_dataset
from torch.utils.data import Dataset, IterableDataset, DataLoader
from torchvision import transforms
from PIL import Image


# ============================================================
#  MODELOS AUXILIARES: MAP-STYLE + STREAMING
# ============================================================

class MiniImagenetMapDataset(Dataset):
    """
    Forma tradicional (dataset completo en disco del contenedor).
    *Nota*: No se descarga en tu notebook, solo en el contenedor Docker.
    """
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        ex = self.hf_dataset[idx]
        img = ex["image"]

        # Asegurar que sea PIL
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = ex["label"]
        return img, label


class MiniImagenetIterableDataset(IterableDataset):
    """
    Forma streaming: NO guarda el dataset completo en disco.
    Lee los ejemplos "al vuelo" desde HuggingFace.
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

            label = ex["label"]
            yield img, label


# ============================================================
#  TRANSFORMACIONES
# ============================================================

def build_default_transform(img_size: int = 224):
    """
    Transformaciones estándar para modelos CNN/Transformer modernos.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])


# ============================================================
#  FUNCIÓN PRINCIPAL DEL DATALOADER
# ============================================================

def get_miniimagenet_dataloaders(
    batch_size: int = 16,     # <--- BATCH ESTÁNDAR UNIVERSAL
    img_size: int = 224,
    num_workers: int = 4,
    streaming: bool = True,  # <--- NO DESCARGA NADA EN EL NOTEBOOK
):
    """
    Retorna:
        train_loader, val_loader, test_loader, num_classes

    Modo streaming:
        - NO descarga el dataset completo
        - NO utiliza cache local en tu notebook
        - Almacena temporalmente solo lo mínimo dentro del contenedor Docker
    """
    transform = build_default_transform(img_size=img_size)

    if streaming:
        # -----------------------------
        # STREAMING (RECOMENDADO)
        # -----------------------------
        hf_train = load_dataset("timm/mini-imagenet", split="train", streaming=True)
        hf_val   = load_dataset("timm/mini-imagenet", split="validation", streaming=True)
        hf_test  = load_dataset("timm/mini-imagenet", split="test", streaming=True)

        train_dataset = MiniImagenetIterableDataset(hf_train, transform)
        val_dataset   = MiniImagenetIterableDataset(hf_val, transform)
        test_dataset  = MiniImagenetIterableDataset(hf_test, transform)

        # num_workers=0 porque IterableDataset NO soporta multiprocessing
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size, num_workers=0)
        test_loader  = DataLoader(test_dataset,  batch_size=batch_size, num_workers=0)

    else:
        # -----------------------------
        # MAP-STYLE (cachea en el contenedor)
        # -----------------------------
        hf_train = load_dataset("timm/mini-imagenet", split="train")
        hf_val   = load_dataset("timm/mini-imagenet", split="validation")
        hf_test  = load_dataset("timm/mini-imagenet", split="test")

        train_dataset = MiniImagenetMapDataset(hf_train, transform)
        val_dataset   = MiniImagenetMapDataset(hf_val, transform)
        test_dataset  = MiniImagenetMapDataset(hf_test, transform)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

    num_classes = 100  # mini-ImageNet tiene 100 clases
    return train_loader, val_loader, test_loader, num_classes


# ============================================================
#  TEST INTERNO (OPCIONAL)
# ============================================================

if __name__ == "__main__":
    train_loader, val_loader, test_loader, num_classes = \
        get_miniimagenet_dataloaders(batch_size=16, streaming=True)

    print("Número de clases:", num_classes)
    batch = next(iter(train_loader))
    images, labels = batch
    print("Batch imágenes:", images.shape)
    print("Batch labels:", labels.shape)
