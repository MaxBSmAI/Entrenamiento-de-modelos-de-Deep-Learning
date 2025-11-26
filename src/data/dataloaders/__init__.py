# src/data/dataloaders/__init__.py
"""
Inicializador de los dataloaders del proyecto.

Permite importar los dataloaders con una sintaxis limpia:
    from data.dataloaders import (
        get_miniimagenet_dataloaders,
        get_coco_detection_dataloaders,
    )

Al agregar nuevos dataloaders (ej. segmentation),
solo debes incorporarlos aquí en __all__.
"""

from .mini_imagenet import get_miniimagenet_dataloaders
from .coco_detection import get_coco_detection_dataloaders

__all__ = [
    "get_miniimagenet_dataloaders",
    "get_coco_detection_dataloaders",
]
