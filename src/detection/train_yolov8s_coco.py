"""
train_yolov8s_coco.py

Entrena un modelo YOLOv8s en COCO Detection usando Ultralytics.

Salida principal en:
    result/detection/yolov8s_coco/<DEVICE_TAG>/

Guarda:
- Pesos del mejor modelo (best.pt) en el directorio interno de Ultralytics
  y referencia en el JSON principal.
- JSON con:
    * test_metrics: map_50, map_50_95
    * benchmark: latencia, fps, throughput, VRAM, params_m, etc.
- Figura PNG con resumen de benchmark computacional.

Incluye EARLY STOPPING vía el parámetro:
    --patience N  → se pasa directo a model.train(patience=N)

NOTA:
Ultralytics ya genera:
    - Curvas de loss
    - PR curves
    - Confusion matrix
en su propio subdirectorio (runs). Aquí nos centramos en unificar
las métricas numéricas y el benchmark en un JSON homogéneo con el resto
del proyecto.

Ejecución recomendada desde la raíz del proyecto:

    cd /workspace
    export PYTHONPATH=./src
    python src/detection/train_yolov8s_coco.py --data coco.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple
import sys

# ----------------------------------------------------------------------
# AÑADIR src al sys.path para que funcionen los imports
# ----------------------------------------------------------------------
SRC_ROOT = Path(__file__).resolve().parents[1]  # .../src
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
# ----------------------------------------------------------------------

import torch
from ultralytics import YOLO

from utils.utils_plot import plot_benchmark_metrics


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


def get_device_and_tags() -> Tuple[torch.device, str, str]:
    """
    Retorna:
        - device       : torch.device (cuda o cpu)
        - device_name  : nombre "amigable" para JSON (p.ej. 'RTX 4080', 'A100', 'CPU')
        - device_tag   : versión para carpeta (p.ej. 'RTX_4080', 'A100', 'CPU')
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


def str2bool(v) -> bool:
    """
    Conversor robusto para argumentos booleanos.
    Permite: true/false, 1/0, yes/no...
    """
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in ("yes", "y", "true", "t", "1"):
        return True
    if v in ("no", "n", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Valor booleano inválido: {v}")


# ============================================================
# Benchmark de inferencia para YOLO
# ============================================================

@torch.no_grad()
def benchmark_yolo_model(
    model: YOLO,
    device: torch.device,
    device_name: str,
    imgsz: int = 640,
    batch_size: int = 16,
    warmup: int = 10,
    runs: int = 50,
) -> Dict[str, Any]:
    """
    Mide latencia promedio, FPS y VRAM para YOLOv8 en un input sintético.
    """
    model.to(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    dummy = torch.randn(batch_size, 3, imgsz, imgsz, device=device)

    # Warmup
    for _ in range(warmup):
        _ = model(dummy, verbose=False)
        if device.type == "cuda":
            torch.cuda.synchronize()

    total_time = 0.0
    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        _ = model(dummy, verbose=False)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        total_time += (t1 - t0)

    if runs == 0 or total_time == 0.0:
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

    mean_latency_ms = (total_time / runs) * 1000.0
    fps = (batch_size * runs) / total_time
    throughput = fps  # samples/s

    if device.type == "cuda":
        vram_used_mb = torch.cuda.max_memory_allocated() / (1024**2)
        vram_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    else:
        vram_used_mb = None
        vram_total_mb = None

    # Parámetros del modelo
    params_m = None
    try:
        params_m = sum(p.numel() for p in model.model.parameters()) / 1e6
    except Exception:
        params_m = None

    return {
        "device_name": device_name,
        "batch_size": batch_size,
        "mean_latency_ms": mean_latency_ms,
        "fps": fps,
        "throughput": throughput,
        "vram_used_mb": vram_used_mb,
        "vram_total_mb": vram_total_mb,
        "flops_g": None,
        "params_m": params_m,
        "power_w": None,
        "efficiency_fps_w": None,
    }


# ============================================================
# Parser
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train YOLOv8s on COCO")

    parser.add_argument("--epochs", type=int, default=100, help="Número máximo de épocas")
    parser.add_argument("--batch_size", type=int, default=16, help="Tamaño de batch")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamaño de imagen")
    parser.add_argument("--data", type=str, default="coco.yaml", help="YAML de datos de Ultralytics")
    parser.add_argument("--run_name", type=str, default="yolov8s_coco", help="Nombre lógico del modelo/corrida")
    parser.add_argument("--no_pretrained", action="store_true", help="Si se usa, entrena desde cero (yolov8s.yaml)")

    parser.add_argument(
        "--rect",
        type=str2bool,
        default=False,
        help="Usar rect training (True/False)",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Épocas sin mejora antes de activar early stopping en YOLOv8.",
    )

    return parser.parse_args()


# ============================================================
# main
# ============================================================

def main() -> None:
    args = parse_args()
    set_seed(42)

    device, device_name, device_tag = get_device_and_tags()
    print(f"[INFO] Dispositivo: {device} ({device_name}), tag: {device_tag}")
    print(f"[INFO] Data YAML: {args.data}")
    print(f"[INFO] rect={args.rect}, patience={args.patience}")

    # Carpetas principales
    project_root = Path(__file__).resolve().parents[2]  # .../workspace
    result_root = project_root / "result" / "detection"

    model_name = args.run_name
    run_root = result_root / model_name / device_tag
    run_root.mkdir(parents=True, exist_ok=True)

    # Ultralytics usará este directorio como "project" y "train" como "name"
    # De esta forma sus artefactos quedarán en:
    #   result/detection/yolov8s_coco/<DEVICE_TAG>/train/
    yolo_project = run_root
    yolo_name = "train"

    metrics_path = run_root / f"{model_name}_metrics.json"
    benchmark_fig_path = run_root / "benchmark_summary.png"

    # Cargar modelo YOLOv8s
    if args.no_pretrained:
        print("[INFO] Cargando modelo YOLOv8s desde config (sin pesos preentrenados)...")
        model = YOLO("yolov8s.yaml")
    else:
        print("[INFO] Cargando modelo YOLOv8s preentrenado (COCO)...")
        model = YOLO("yolov8s.pt")

    # Seleccionar dispositivo para Ultralytics (índice 0 si hay GPU)
    yolo_device_arg = 0 if device.type == "cuda" else "cpu"

    # Entrenamiento con early stopping (patience)
    print("\n[INFO] Iniciando entrenamiento YOLOv8s...")
    train_results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        device=yolo_device_arg,
        project=str(yolo_project),
        name=yolo_name,
        exist_ok=True,
        rect=args.rect,
        verbose=True,
        patience=args.patience,  # <-- EARLY STOPPING
    )

    print("[INFO] Entrenamiento finalizado.")

    # Localizar pesos "best.pt"
    best_weights = yolo_project / yolo_name / "weights" / "best.pt"
    if not best_weights.exists():
        # Fallback: intentar buscar best*.pt en el subárbol
        candidates = list(yolo_project.rglob("best*.pt"))
        if candidates:
            best_weights = candidates[0]

    if best_weights.exists():
        print(f"[INFO] Pesos del mejor modelo: {best_weights}")
    else:
        print("[WARN] No se encontró best.pt, se usará el modelo en memoria para evaluación.")

    # Evaluación (val) para obtener mAP
    print("\n[INFO] Evaluando mAP en split=val...")
    eval_model = YOLO(str(best_weights)) if best_weights.exists() else model

    # Ultralytics Metrics: results.box.map (mAP50-95), results.box.map50, etc.
    val_results = eval_model.val(
        data=args.data,
        imgsz=args.imgsz,
        device=yolo_device_arg,
        split="val",
        verbose=True,
    )

    map_50 = None
    map_50_95 = None
    try:
        map_50 = float(val_results.box.map50)
        map_50_95 = float(val_results.box.map)
    except Exception:
        print("[WARN] No se pudieron extraer map_50 / map_50_95 desde Ultralytics Metrics.")

    test_metrics = {
        "map_50": map_50,
        "map_50_95": map_50_95,
    }

    if map_50 is not None and map_50_95 is not None:
        print(f"[VAL] mAP@0.5: {map_50:.4f} | mAP@0.5:0.95: {map_50_95:.4f}")
    else:
        print("[VAL] mAP no disponible (val_results.box.map/map50 faltan)")

    # Benchmark sintético
    print("\n[INFO] Ejecutando benchmark sintético de inferencia YOLOv8s...")
    benchmark = benchmark_yolo_model(
        eval_model,
        device=device,
        device_name=device_name,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        warmup=10,
        runs=50,
    )

    if benchmark["fps"] is not None:
        print(
            f"[BENCHMARK] Device: {benchmark['device_name']} | "
            f"Latencia media (ms): {benchmark['mean_latency_ms']:.3f} | "
            f"FPS: {benchmark['fps']:.2f} | "
            f"VRAM usada (MB): {benchmark['vram_used_mb']:.2f}"
        )
    else:
        print("[BENCHMARK] sin datos")

    # Gráfico resumen de benchmark
    plot_benchmark_metrics(
        benchmark,
        benchmark_fig_path,
        title=f"Benchmark computacional — {model_name} ({device_name})",
    )
    print(f"[INFO] Gráfico de benchmark guardado en: {benchmark_fig_path}")

    # Guardar JSON de métricas unificado
    metrics: Dict[str, Any] = {
        "model_name": model_name,
        "task": "detection",
        "dataset": args.data,
        "num_classes": None,   # Ultralytics conoce internamente el nº de clases
        "epochs": args.epochs,
        "best_epoch": None,    # Ultralytics no expone directamente el número de época
        "device_name": device_name,
        "device_tag": device_tag,
        "test_loss": None,     # opcional, Ultralytics puede extraerlo de logs si se desea
        "test_metrics": test_metrics,
        "benchmark": benchmark,
        "artifacts": {
            "best_model_path": str(best_weights) if best_weights.exists() else None,
            "ultralytics_project_dir": str(yolo_project / yolo_name),
            "benchmark_summary_png": str(benchmark_fig_path),
        },
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"[INFO] Métricas YOLOv8s guardadas en: {metrics_path}")
    print(f"[INFO] Artefactos de Ultralytics disponibles en: {yolo_project / yolo_name}")


if __name__ == "__main__":
    main()
