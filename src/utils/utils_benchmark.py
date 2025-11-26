"""
utils_benchmark.py

Métricas de desempeño computacional para modelos de Deep Learning:

- Latencia
- FPS (frames per second)
- Throughput
- Uso de memoria GPU
- Tamaño del modelo en MB
- FLOPs, parámetros
- Potencia y eficiencia energética
- Tiempo de entrenamiento
- mAP@0.5 IoU (detección)
"""

from __future__ import annotations

import time
from typing import Optional, Tuple, List, Dict, Any, Callable, Generator
from pathlib import Path

import numpy as np
import torch
from contextlib import contextmanager

# ============================================
# Opcional: ptflops (FLOPs y parámetros)
# ============================================

_HAS_PTFLOPS = False
try:
    from ptflops import get_model_complexity_info  # type: ignore
    _HAS_PTFLOPS = True
except Exception:
    _HAS_PTFLOPS = False

# ============================================
# Opcional: NVML (energía)
# ============================================

_HAS_NVML = False
try:
    import pynvml  # type: ignore

    pynvml.nvmlInit()
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False


# ======================================================
#  Utilities de tiempo
# ======================================================

@contextmanager
def measure_time() -> Generator[Callable[[], float], None, None]:
    """
    Context manager para medir tiempo.

    Ejemplo:
        with measure_time() as t:
            ...
        print(t())  # segundos transcurridos
    """
    start = time.time()

    def elapsed() -> float:
        return time.time() - start

    yield elapsed


def measure_training_time(start_time: float, end_time: float) -> float:
    """Tiempo total de entrenamiento en segundos."""
    return float(end_time - start_time)


# ======================================================
#  Sincronización CUDA
# ======================================================

def _sync_if_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ======================================================
#  Latencia, FPS, throughput
# ======================================================

def measure_inference_latency(
    model: torch.nn.Module,
    batch: torch.Tensor,
    runs: int = 50,
    warmup: int = 10,
    sync: bool = True,
) -> Tuple[float, float]:
    """
    Mide latencia en ms.
    """
    model.eval()
    latencies: List[float] = []

    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            _ = model(batch)
            if sync:
                _sync_if_cuda()

        # Medición
        for _ in range(runs):
            if sync:
                _sync_if_cuda()
            t0 = time.perf_counter()
            _ = model(batch)
            if sync:
                _sync_if_cuda()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

    latencies_arr = np.array(latencies, dtype=np.float64)
    return float(latencies_arr.mean()), float(latencies_arr.std())


def measure_fps(
    model: torch.nn.Module,
    batch: torch.Tensor,
    duration_seconds: float = 5.0,
    sync: bool = True,
) -> float:
    """
    Mide FPS durante `duration_seconds`.
    """
    model.eval()
    batch_size = batch.shape[0]
    n_images = 0

    start = time.perf_counter()
    with torch.no_grad():
        while True:
            if sync:
                _sync_if_cuda()

            _ = model(batch)

            if sync:
                _sync_if_cuda()

            n_images += batch_size
            elapsed = time.perf_counter() - start
            if elapsed >= duration_seconds:
                break

    fps = n_images / elapsed
    return float(fps)


def throughput_from_latency(batch_size: int, latency_ms: float) -> float:
    """
    Throughput = samples/segundo = batch_size / (lat_ms / 1000)
    """
    if latency_ms <= 0:
        return 0.0
    return float(batch_size * 1000.0 / latency_ms)


# ======================================================
#  GPU memory + Model size + FLOPs
# ======================================================

def gpu_memory_usage(device_idx: int = 0) -> Tuple[float, float]:
    """
    Retorna (memoria_usada_MB, memoria_total_MB)
    """
    if not torch.cuda.is_available():
        return 0.0, 0.0

    free, total = torch.cuda.mem_get_info(device_idx)
    used = total - free
    return float(used / 1024 ** 2), float(total / 1024 ** 2)


def model_size_in_mb(model: torch.nn.Module, save_path: Optional[Path] = None) -> float:
    """
    Tamaño del modelo en MB. Si save_path se especifica, guarda el modelo y mide el archivo.
    """
    # Tamaño estimado por parámetros
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    total_bytes = param_size + buffer_size

    estimate_mb = total_bytes / (1024 ** 2)

    if save_path is not None:
        save_path = Path(save_path)
        torch.save(model.state_dict(), save_path)
        file_mb = save_path.stat().st_size / (1024 ** 2)
        return float(file_mb)

    return float(estimate_mb)


def estimate_flops_and_params(
    model: torch.nn.Module,
    input_shape: Tuple[int, int, int, int],  # (B, C, H, W)
) -> Tuple[Optional[float], Optional[float]]:
    """
    Estima FLOPs (GFLOPs) y parámetros (M) usando ptflops si está disponible.
    """
    if not _HAS_PTFLOPS:
        return None, None

    _, c, h, w = input_shape
    macs, params = get_model_complexity_info(
        model,
        (c, h, w),
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False,
    )

    flops = 2.0 * macs  # FLOPs ≈ 2 * MACs
    flops_g = flops / 1e9
    params_m = params / 1e6

    return float(flops_g), float(params_m)


# ======================================================
#  Energía (potencia instantánea, eficiencia FPS/Watt)
# ======================================================

def gpu_power_usage(device_idx: int = 0) -> Optional[float]:
    """
    Potencia actual de la GPU en Watts si NVML está disponible.
    """
    if not _HAS_NVML:
        return None

    handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
    power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)  # mW
    return float(power_mw / 1000.0)


def energy_efficiency(fps: float, power_watts: Optional[float]) -> Optional[float]:
    """
    FPS por watt. None si no se puede calcular.
    """
    if power_watts is None or power_watts <= 0:
        return None
    return float(fps / power_watts)


# ======================================================
#  mAP@0.5 IoU para detección (simple VOC-style)
# ======================================================

def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    IoU entre dos conjuntos de cajas [x1, y1, x2, y2].
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))

    x11, y11, x12, y12 = boxes1.T
    x21, y21, x22, y22 = boxes2.T

    xa = np.maximum(x11[:, None], x21[None, :])
    ya = np.maximum(y11[:, None], y21[None, :])
    xb = np.minimum(x12[:, None], x22[None, :])
    yb = np.minimum(y12[:, None], y22[None, :])

    inter = np.clip(xb - xa, 0, None) * np.clip(yb - ya, 0, None)
    area1 = (x12 - x11) * (y12 - y11)
    area2 = (x22 - x21) * (y22 - y21)

    union = area1[:, None] + area2[None, :] - inter
    return inter / np.clip(union, 1e-6, None)


def compute_map_at_05(
    preds: List[Dict[str, np.ndarray]],
    gts: List[Dict[str, np.ndarray]],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> float:
    """
    mAP@0.5 estilo VOC (simple).

    preds[i] y gts[i]:
        {
            "boxes": (N,4),
            "scores": (N,),
            "labels": (N,)
        }
    """
    aps: List[float] = []

    for cls in range(num_classes):
        cls_scores: List[np.ndarray] = []
        cls_matches: List[np.ndarray] = []
        total_gts = 0

        for pred, gt in zip(preds, gts):
            pb, ps, pl = pred["boxes"], pred["scores"], pred["labels"]
            gb, gl = gt["boxes"], gt["labels"]

            p_mask = pl == cls
            g_mask = gl == cls

            pb, ps = pb[p_mask], ps[p_mask]
            gb = gb[g_mask]

            total_gts += len(gb)

            if len(pb) == 0:
                continue

            order = np.argsort(-ps)
            pb, ps = pb[order], ps[order]
            matches = np.zeros(len(pb), dtype=np.float32)

            if len(gb) > 0:
                ious = box_iou(pb, gb)
                gt_used = np.zeros(len(gb), dtype=bool)

                for i in range(len(pb)):
                    j = np.argmax(ious[i])
                    if ious[i, j] >= iou_threshold and not gt_used[j]:
                        matches[i] = 1.0
                        gt_used[j] = True

            cls_scores.append(ps)
            cls_matches.append(matches)

        if total_gts == 0:
            continue

        if len(cls_scores) == 0:
            aps.append(0.0)
            continue

        cls_scores_arr = np.concatenate(cls_scores)
        cls_matches_arr = np.concatenate(cls_matches)

        order = np.argsort(-cls_scores_arr)
        cls_scores_arr = cls_scores_arr[order]
        cls_matches_arr = cls_matches_arr[order]

        tp = np.cumsum(cls_matches_arr)
        fp = np.cumsum(1 - cls_matches_arr)

        recall = tp / max(total_gts, 1)
        precision = tp / np.maximum(tp + fp, 1e-6)

        ap = np.trapz(precision, recall)
        aps.append(float(ap))

    if len(aps) == 0:
        return 0.0

    return float(np.mean(aps))
