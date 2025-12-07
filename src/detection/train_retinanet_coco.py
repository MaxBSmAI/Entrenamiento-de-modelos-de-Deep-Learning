# src/onnx/export_to_onnx.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Optional

import sys
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    retinanet_resnet50_fpn_v2,
)


# ============================================================
# Añadir src al sys.path
# ============================================================
SRC_ROOT = Path(__file__).resolve().parents[1]  # .../src
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


# ============================================================
# Helpers de dispositivo
# ============================================================

def get_device_and_tags() -> Tuple[torch.device, str, str]:
    """
    Devuelve:
        device      : torch.device ('cuda' o 'cpu')
        device_name : nombre legible (RTX 4080, A100, CPU, etc.)
        device_tag  : versión "amigable" para carpetas (RTX_4080, A100, CPU)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        raw_name = torch.cuda.get_device_name(0)
        raw_lower = raw_name.lower()

        if "4080" in raw_lower:
            device_name = "RTX 4080"
            device_tag = "RTX_4080"
        elif "4060" in raw_lower:
            device_name = "RTX 4060"
            device_tag = "RTX_4060"
        elif "a100" in raw_lower:
            device_name = "A100"
            device_tag = "A100"
        else:
            device_name = raw_name.strip()
            device_tag = raw_name.replace(" ", "_").replace("-", "_")
    else:
        device = torch.device("cpu")
        device_name = "CPU"
        device_tag = "CPU"

    return device, device_name, device_tag


# ============================================================
# MODELOS DE CLASIFICACIÓN
# ============================================================

def build_classification_model(model_name: str, num_classes: int) -> nn.Module:
    """
    Reconstruye el modelo de clasificación según 'model_name'.

    Soporta:
        - 'resnet50_miniimagenet'
        - 'vit_b16_miniimagenet'
    """
    model_name = model_name.lower()

    if model_name == "resnet50_miniimagenet":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    elif model_name == "vit_b16_miniimagenet":
        import timm

        model = timm.create_model(
            "vit_base_patch16_224.augreg_in21k_ft_in1k",
            pretrained=False,
            num_classes=num_classes,
        )
        return model

    raise ValueError(f"Modelo de clasificación no soportado: {model_name}")


# ============================================================
# MODELOS DE DETECCIÓN
# ============================================================

def build_detection_model(model_name: str) -> nn.Module:
    """
    Reconstruye el modelo de detección según 'model_name'.

    Soporta:
        - 'fasterrcnn_coco'
        - 'retinanet_coco'

    IMPORTANTE:
    - No se usan pesos preentrenados aquí (weights=None) para evitar descargas.
    - Las arquitecturas coinciden con las usadas en el entrenamiento, por lo
      que al cargar el state_dict se sobrescriben correctamente los pesos.
    """
    model_name = model_name.lower()

    if model_name == "fasterrcnn_coco":
        # Misma arquitectura que train_Faster_R_CNN.py
        model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
        return model

    if model_name == "retinanet_coco":
        # Misma arquitectura que train_retinanet_coco.py
        model = retinanet_resnet50_fpn_v2(weights=None, weights_backbone=None)
        return model

    raise ValueError(f"Modelo de detección no soportado: {model_name}")


class DetectionWrapper(nn.Module):
    """
    Wrapper para hacer al modelo de torchvision más amigable con ONNX.

    Entrada:
        images: Tensor [B, 3, H, W]

    Salida:
        boxes : Tensor [N_det, 4]
        scores: Tensor [N_det]
        labels: Tensor [N_det]

    Por simplicidad, se exporta pensando en B=1 (un batch). El tamaño N_det
    es dinámico en ONNX.
    """

    def __init__(self, detector: nn.Module):
        super().__init__()
        self.detector = detector

    def forward(self, images: torch.Tensor):
        # images: [B,3,H,W]  → lista de Tensors [3,H,W]
        image_list = [img for img in images]
        outputs = self.detector(image_list)  # lista de dicts

        # Aquí usamos solo el primer elemento del batch (B=1)
        out0 = outputs[0]
        boxes = out0["boxes"]
        scores = out0["scores"]
        labels = out0["labels"]
        return boxes, scores, labels


# ============================================================
# Utilidades de paths
# ============================================================

def find_checkpoint(
    project_root: Path,
    task: str,
    model_name: str,
    device_tag: str,
    checkpoint: Optional[str] = None,
) -> Path:
    """
    Si 'checkpoint' no es None, se usa directamente.
    Si no, se busca en:
        result/<task>/<model_name>/<device_tag>/<model_name>_best.pth
    y luego en:
        result/<task>/<model_name>/<model_name>_best.pth
    """
    if checkpoint is not None:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint explícito no existe: {ckpt_path}")
        return ckpt_path

    task_root = project_root / "result" / task / model_name

    cand1 = task_root / device_tag / f"{model_name}_best.pth"
    cand2 = task_root / f"{model_name}_best.pth"

    if cand1.exists():
        return cand1
    if cand2.exists():
        return cand2

    raise FileNotFoundError(
        f"No se encontró checkpoint para {model_name}. "
        f"Probados: {cand1} y {cand2}"
    )


def find_metrics_json(ckpt_path: Path) -> Optional[Path]:
    """
    Intenta encontrar el JSON de métricas en la misma carpeta que el .pth.
    """
    json_path = ckpt_path.with_name(ckpt_path.stem.replace("_best", "_metrics") + ".json")
    return json_path if json_path.exists() else None


# ============================================================
# Parser
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exportar modelos entrenados a ONNX "
            "(ResNet-50, ViT-B/16, Faster R-CNN, RetinaNet)"
        ),
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["classification", "detection"],
        help="Tarea del modelo a exportar.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "Nombre lógico del modelo:\n"
            "  classification: resnet50_miniimagenet, vit_b16_miniimagenet\n"
            "  detection     : fasterrcnn_coco, retinanet_coco"
        ),
    )
    parser.add_argument(
        "--device_tag",
        type=str,
        default=None,
        help=(
            "Tag del dispositivo usado al entrenar (RTX_4060, RTX_4080, A100...). "
            "Si no se especifica, se infiere del GPU actual."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Ruta explícita al .pth. Si se omite, se busca automáticamente en result/<task>/...",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=None,
        help=(
            "Tamaño de imagen (lado) para el dummy input. "
            "Por defecto: 224 (clasificación), 640 (detección)."
        ),
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="Versión de opset ONNX (>= 12 recomendado).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Directorio base de salida para .onnx.\n"
            "Por defecto: models/onnx/<task>/"
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[2]
    device, device_name, auto_device_tag = get_device_and_tags()

    device_tag = args.device_tag or auto_device_tag
    print(f"[INFO] Device actual: {device} ({device_name}) → device_tag={device_tag}")

    # ------------------------------------------------------------------
    # Localizar checkpoint
    # ------------------------------------------------------------------
    ckpt_path = find_checkpoint(
        project_root=project_root,
        task=args.task,
        model_name=args.model,
        device_tag=device_tag,
        checkpoint=args.checkpoint,
    )

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {ckpt_path}")

    print(f"[INFO] Usando checkpoint: {ckpt_path}")

    # ------------------------------------------------------------------
    # Leer num_classes desde JSON (si aplica)
    # ------------------------------------------------------------------
    num_classes = 100 if args.task == "classification" else None
    metrics_json = find_metrics_json(ckpt_path)

    if metrics_json is not None:
        try:
            import json
            with open(metrics_json, "r") as f:
                data = json.load(f)
            if args.task == "classification":
                num_classes = int(data.get("num_classes", num_classes))
                print(f"[INFO] num_classes (clasificación) leído desde {metrics_json}: {num_classes}")
            else:
                # Para detección es solo informativo
                nc = data.get("num_classes", None)
                print(f"[INFO] num_classes (detección) en métricas: {nc}")
        except Exception as e:
            print(f"[WARN] No se pudo leer métricas desde {metrics_json}: {e}")
    else:
        if args.task == "classification":
            print(f"[INFO] No se encontró JSON de métricas; usando num_classes={num_classes}")

    # ------------------------------------------------------------------
    # Construir modelo
    # ------------------------------------------------------------------
    if args.task == "classification":
        if num_classes is None:
            raise ValueError("num_classes es None para clasificación.")
        model = build_classification_model(args.model, num_classes=num_classes)
        default_img_size = 224
    else:
        model = build_detection_model(args.model)
        default_img_size = 640

    # Cargar pesos
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Si es detección, envolvemos el modelo
    if args.task == "detection":
        model = DetectionWrapper(model).to(device).eval()

    # ------------------------------------------------------------------
    # Dummy input
    # ------------------------------------------------------------------
    img_size = args.img_size or default_img_size
    if args.task == "classification":
        dummy_input = torch.randn(1, 3, img_size, img_size, device=device)
        print(f"[INFO] Dummy input (clasificación) shape: {tuple(dummy_input.shape)}")
    else:
        # Para detección usamos B=1
        dummy_input = torch.randn(1, 3, img_size, img_size, device=device)
        print(f"[INFO] Dummy input (detección) shape: {tuple(dummy_input.shape)}")

    # ------------------------------------------------------------------
    # Directorio de salida
    # ------------------------------------------------------------------
    if args.output_dir is not None:
        out_base = Path(args.output_dir)
    else:
        out_base = project_root / "models" / "onnx" / args.task

    out_base.mkdir(parents=True, exist_ok=True)

    onnx_name = f"{args.model}__{device_tag}.onnx"
    onnx_path = out_base / onnx_name

    # ------------------------------------------------------------------
    # Exportar a ONNX
    # ------------------------------------------------------------------
    print(f"[INFO] Exportando a ONNX → {onnx_path}")

    if args.task == "classification":
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
        )
    else:
        # detección: DetectionWrapper devuelve (boxes, scores, labels)
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["boxes", "scores", "labels"],
            dynamic_axes={
                "images": {0: "batch_size"},
                "boxes": {0: "num_detections"},
                "scores": {0: "num_detections"},
                "labels": {0: "num_detections"},
            },
        )

    print("\n[OK] Exportación completada.")
    print(f"[OK] Modelo ONNX guardado en: {onnx_path}")
    print(
        "[INFO] Ahora puedes usar benchmark_onnx_runtime.py y benchmark_tensorrt.py "
        "para comparar rendimiento en RTX 4080 y A100."
    )


if __name__ == "__main__":
    main()
