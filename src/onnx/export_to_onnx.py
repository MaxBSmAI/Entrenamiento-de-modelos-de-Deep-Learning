# src/onnx/export_to_onnx.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import sys
import torch
import torch.nn as nn
from torchvision import models


# ============================================================
# Añadir src al sys.path
# ============================================================
SRC_ROOT = Path(__file__).resolve().parents[1]  # .../src
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


# ============================================================
# Device helpers
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
# Construcción de modelos de CLASIFICACIÓN
# ============================================================

def build_classification_model(model_name: str, num_classes: int) -> nn.Module:
    """
    Reconstruye el modelo de clasificación según 'model_name'.

    Por ahora soporta:
        - 'resnet50_miniimagenet'
        - 'vit_b16_miniimagenet'
    """
    model_name = model_name.lower()

    if model_name == "resnet50_miniimagenet":
        # ResNet-50 estándar, reemplazando la FC final
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    elif model_name == "vit_b16_miniimagenet":
        # ViT-B/16 desde timm
        import timm

        # Debe coincidir con el usado en train_vit_b16_imagenet.py
        model = timm.create_model(
            "vit_base_patch16_224.augreg_in21k_ft_in1k",
            pretrained=False,
            num_classes=num_classes,
        )
        return model

    else:
        raise ValueError(f"Modelo de clasificación no soportado: {model_name}")


# ============================================================
# Utilidades de paths
# ============================================================

def find_checkpoint(
    project_root: Path,
    task: str,
    model_name: str,
    device_tag: str,
) -> Path:
    """
    Busca el .pth del mejor modelo en:
        result/<task>/<model_name>/<device_tag>/<model_name>_best.pth
    y si no existe, prueba:
        result/<task>/<model_name>/<model_name>_best.pth
    """
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


def find_metrics_json(ckpt_path: Path) -> Path | None:
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
        description="Exportar modelos entrenados a ONNX para interoperabilidad HPC",
    )

    parser.add_argument(
        "--task",
        type=str,
        default="classification",
        choices=["classification"],   # por ahora solo clasificación
        help="Tarea a exportar (classification; detection se puede agregar luego).",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Nombre lógico del modelo (ej: resnet50_miniimagenet, vit_b16_miniimagenet).",
    )
    parser.add_argument(
        "--device_tag",
        type=str,
        default=None,
        help="Tag del dispositivo usado en el entrenamiento (RTX_4060, RTX_4080, A100...). "
             "Si no se especifica, se infiere del GPU actual.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Ruta explícita al .pth. Si se omite, se busca en result/<task>/<model>/<device_tag>/.",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Tamaño de imagen cuadrada para el dummy input (clasificación).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="Versión de opset ONNX.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directorio base de salida para .onnx (por defecto models/onnx/<task>/).",
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
    if args.checkpoint is not None:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = find_checkpoint(
            project_root=project_root,
            task=args.task,
            model_name=args.model,
            device_tag=device_tag,
        )

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint no encontrado: {ckpt_path}")

    print(f"[INFO] Usando checkpoint: {ckpt_path}")

    # ------------------------------------------------------------------
    # Leer num_classes desde el JSON si está disponible
    # ------------------------------------------------------------------
    num_classes = 100  # valor por defecto para mini-ImageNet
    metrics_json = find_metrics_json(ckpt_path)
    if metrics_json is not None:
        try:
            import json
            with open(metrics_json, "r") as f:
                data = json.load(f)
            num_classes = int(data.get("num_classes", num_classes))
            print(f"[INFO] num_classes obtenido desde {metrics_json}: {num_classes}")
        except Exception as e:
            print(f"[WARN] No se pudo leer num_classes desde {metrics_json}: {e}")
    else:
        print(f"[INFO] No se encontró JSON de métricas; usando num_classes={num_classes}")

    # ------------------------------------------------------------------
    # Construir modelo según la tarea
    # ------------------------------------------------------------------
    if args.task == "classification":
        model = build_classification_model(args.model, num_classes=num_classes)
    else:
        raise NotImplementedError("Por ahora solo se soporta 'classification' en export_to_onnx.py")

    # Cargar pesos
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # ------------------------------------------------------------------
    # Crear dummy input
    # ------------------------------------------------------------------
    img_size = args.img_size
    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)
    print(f"[INFO] Dummy input shape: {tuple(dummy_input.shape)}")

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

    print("\n[OK] Exportación completada.")
    print(f"[OK] Modelo ONNX guardado en: {onnx_path}")
    print(f"[INFO] Ahora puedes usar benchmark_onnx_runtime.py y benchmark_tensorrt.py "
          f"para medir latencia/FPS en RTX 4080 y A100.")


if __name__ == "__main__":
    main()
