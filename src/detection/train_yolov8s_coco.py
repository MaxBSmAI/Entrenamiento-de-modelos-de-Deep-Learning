"""
train_yolov8s_coco.py

Entrena YOLOv8s en COCO Detection. Compatible con tu estructura result/detection/.

- Por defecto entrena SIN PREENTRENADO (yolov8s.yaml).
- Si se usa --pretrained → carga pesos COCO (yolov8s.pt).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any
import sys

import torch
from ultralytics import YOLO  # pip install ultralytics


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


def str2bool(v) -> bool:
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in ("yes", "y", "true", "t", "1"):
        return True
    if v in ("no", "n", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Valor booleano inválido: {v}")


# ============================================================
# Benchmark inferencia sintética
# ============================================================

@torch.no_grad()
def benchmark_yolov8_model(
    model: YOLO,
    device: torch.device,
    img_size: int = 640,
    batch_size: int = 8,
    max_batches: int = 20,
) -> Dict[str, Any]:

    backend = model.model.to(device).eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        device_name = torch.cuda.get_device_name(0)
    else:
        device_name = "cpu"

    total_time, total_images = 0.0, 0

    for _ in range(max_batches):
        x = torch.randn(batch_size, 3, img_size, img_size, device=device)

        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.time()

        _ = backend(x)

        if device.type == "cuda": torch.cuda.synchronize()
        t1 = time.time()

        total_time += (t1 - t0)
        total_images += batch_size

    if total_time == 0:
        return {
            "device_name": device_name,
            "batch_size": batch_size,
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

    mean_latency_ms = (total_time / max_batches) * 1000
    fps = total_images / total_time
    throughput = fps

    if device.type == "cuda":
        vram_used = torch.cuda.max_memory_allocated() / (1024 ** 2)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    else:
        vram_used = None
        total_vram = None

    params_m = sum(p.numel() for p in backend.parameters()) / 1e6

    return {
        "device_name": device_name,
        "batch_size": batch_size,
        "mean_latency_ms": mean_latency_ms,
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
# Argumentos
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8s on COCO")

    parser.add_argument("--data", type=str, default="coco.yaml",
                        help="YAML de datos (ej: coco.yaml)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--run_name", type=str, default="yolov8s_coco")

    # 🔥 AQUÍ ESTÁ TU MODIFICACIÓN
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=False,
        help="Si se especifica, usa pesos preentrenados (yolov8s.pt). Por defecto entrena desde cero."
    )

    parser.add_argument(
        "--streaming",
        type=str2bool,
        default=True,
        help="Decorativo: YOLOv8 no usa este parámetro."
    )

    return parser.parse_args()


# ============================================================
# main
# ============================================================

def main() -> None:
    args = parse_args()
    set_seed(42)

    device = get_device()
    print(f"[INFO] Device: {device}")
    print(f"[INFO] pretrained={args.pretrained}")
    print(f"[INFO] streaming={args.streaming} (no afecta a YOLOv8)")

    # Estructura result/detection/<run_name>/
    project_root = Path(__file__).resolve().parents[2]
    result_dir = project_root / "result" / "detection" / args.run_name
    result_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = result_dir / f"{args.run_name}_metrics.json"

    # --------------------------------------------------------
    # Cargar modelo
    # --------------------------------------------------------
    if args.pretrained:
        print("[INFO] Cargando YOLOv8s preentrenado (yolov8s.pt)...")
        model = YOLO("yolov8s.pt")
    else:
        print("[INFO] Entrenando YOLOv8s desde cero (yolov8s.yaml)...")
        model = YOLO("yolov8s.yaml")

    # --------------------------------------------------------
    # Entrenamiento
    # --------------------------------------------------------
    device_str = "0" if device.type == "cuda" else "cpu"

    print("\n[INFO] Iniciando entrenamiento...")
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        lr0=args.lr,
        device=device_str,
        project=str(result_dir),
        name="run",
        exist_ok=True,
        verbose=True,
    )

    # --------------------------------------------------------
    # Evaluación
    # --------------------------------------------------------
    print("\n[INFO] Evaluando en conjunto de validación...")
    metrics = model.val(
        data=args.data,
        imgsz=args.img_size,
        batch=args.batch_size,
        device=device_str,
    )

    box_metrics = getattr(metrics, "box", metrics)

    map50 = float(getattr(box_metrics, "map50", 0.0))
    map5095 = float(getattr(box_metrics, "map", 0.0))

    print(f"[VAL] mAP@0.5={map50:.4f} | mAP@0.5:0.95={map5095:.4f}")

    # --------------------------------------------------------
    # Benchmark sintético
    # --------------------------------------------------------
    print("\n[INFO] Ejecutando benchmark de inferencia...")
    benchmark = benchmark_yolov8_model(
        model=model,
        device=device,
        img_size=args.img_size,
        batch_size=args.batch_size,
        max_batches=20,
    )

    # --------------------------------------------------------
    # Guardar métricas
    # --------------------------------------------------------
    metrics_dict = {
        "model_name": args.run_name,
        "task": "detection",
        "dataset": "coco_yolo",
        "epochs": args.epochs,
        "num_classes": getattr(metrics, "nc", None),
        "test_loss": None,  # YOLO no expone test_loss como un único valor
        "test_metrics": {
            "map_50": map50,
            "map_50_95": map5095,
        },
        "benchmark": benchmark,
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)

    print(f"[INFO] Métricas guardadas en: {metrics_path}")


if __name__ == "__main__":
    main()
