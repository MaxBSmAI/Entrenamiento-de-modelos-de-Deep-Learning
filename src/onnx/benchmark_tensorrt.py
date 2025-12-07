# src/onnx/benchmark_tensorrt.py

"""
benchmark_tensorrt.py

Benchmark de modelos ONNX usando TensorRT (vía trtexec).

Soporta ONNX exportados desde:
    - train_resnet50_imagenet.py      (clasificación)
    - train_vit_b16_imagenet.py       (clasificación)
    - train_Faster_R_CNN.py           (detección)
    - train_retinanet_coco.py         (detección)

La idea es medir:
    - latencia media GPU (ms)
    - throughput / FPS
    - batch_size
    - modo de precisión (FP32 / FP16)

Uso típico (desde /workspace):

    export PYTHONPATH=./src

    # Ejemplo ResNet-50
    python src/onnx/benchmark_tensorrt.py \
        --onnx-path models/onnx/classification/resnet50_miniimagenet__RTX_4080.onnx \
        --task classification \
        --batch-size 16 \
        --fp16 \
        --save-json

    # Ejemplo Faster R-CNN
    python src/onnx/benchmark_tensorrt.py \
        --onnx-path models/onnx/detection/fasterrcnn_coco__RTX_4080.onnx \
        --task detection \
        --batch-size 4 \
        --fp16 \
        --save-json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import onnx
import numpy as np


# ============================================================
# Helpers para leer el ONNX
# ============================================================


def get_onnx_input_info(onnx_path: Path) -> Tuple[str, Tuple[int, int, int]]:
    """
    Devuelve:
        - input_name
        - (C, H, W)

    Se asume que el primer input es la imagen:
        [batch, C, H, W] o ["batch", C, H, W]
    """
    model = onnx.load(str(onnx_path))
    graph = model.graph
    if not graph.input:
        raise ValueError(f"El ONNX no tiene inputs definidos: {onnx_path}")

    inp = graph.input[0]
    input_name = inp.name

    tensor_type = inp.type.tensor_type
    shape_proto = tensor_type.shape

    if len(shape_proto.dim) < 4:
        raise ValueError(
            f"Se esperaba input 4D [N,C,H,W] en {onnx_path}, pero tiene {len(shape_proto.dim)} dimensiones."
        )

    # dim[0] = batch (posiblemente dinámico)
    c_dim = shape_proto.dim[1]
    h_dim = shape_proto.dim[2]
    w_dim = shape_proto.dim[3]

    def dim_to_int(d) -> int:
        if d.HasField("dim_value"):
            return int(d.dim_value)
        else:
            # si es simbólico, devolvemos un valor típico
            return 3 if d is c_dim else 224

    c = dim_to_int(c_dim)
    h = dim_to_int(h_dim)
    w = dim_to_int(w_dim)

    return input_name, (c, h, w)


# ============================================================
# Parseo de la salida de trtexec
# ============================================================


def parse_trtexec_stdout(stdout: str) -> Dict[str, Any]:
    """
    Extrae latencia y throughput de la salida de trtexec.
    Es best-effort: intenta encontrar líneas con 'GPU latency' y 'Throughput'.
    """
    latency_ms = None
    throughput = None

    for line in stdout.splitlines():
        line = line.strip()

        # Ejemplo típico:
        # "Average on 10 runs - GPU latency: 1.234 ms - Host latency: 1.345 ms (end to end 1.456 ms)"
        if "GPU latency" in line and "ms" in line:
            m = re.search(r"GPU latency:\s*([0-9.]+)\s*ms", line)
            if m:
                latency_ms = float(m.group(1))

        # Ejemplo típico:
        # "Throughput: 1234 qps"
        if "Throughput:" in line:
            m = re.search(r"Throughput:\s*([0-9.]+)", line)
            if m:
                throughput = float(m.group(1))

    # En muchos casos, throughput ≈ FPS para batch_size=1.
    # Para batch_size>1, FPS ≈ throughput * batch_size / batch_size? depende de la versión de trtexec.
    # Aquí lo dejamos tal cual y el usuario puede interpretarlo.
    return {
        "mean_latency_ms": latency_ms,
        "throughput": throughput,
        "fps": throughput,  # alias
    }


# ============================================================
# Ejecutar trtexec
# ============================================================


def run_trtexec(
    onnx_path: Path,
    input_name: str,
    chw: Tuple[int, int, int],
    batch_size: int,
    fp16: bool,
    trtexec_path: str,
    workspace_mb: int = 1024,
    warmup: int = 200,
    duration: int = 3,
) -> Dict[str, Any]:
    """
    Ejecuta trtexec con parámetros razonables y devuelve métricas parseadas.
    """
    c, h, w = chw
    shapes_arg = f"{input_name}:{batch_size}x{c}x{h}x{w}"

    cmd = [
        trtexec_path,
        f"--onnx={onnx_path}",
        f"--shapes={shapes_arg}",
        f"--workspace={workspace_mb}",
        f"--warmUp={warmup}",
        f"--duration={duration}",
        "--useSpinWait",
        "--skipInference=false",
        "--reportLayers=false",
    ]

    precision_mode = "fp32"
    if fp16:
        cmd.append("--fp16")
        precision_mode = "fp16"

    print("[INFO] Ejecutando trtexec:")
    print("       " + " ".join(cmd))

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    t1 = time.time()

    stdout = proc.stdout
    retcode = proc.returncode

    print("[INFO] trtexec finalizado, código de retorno:", retcode)
    if retcode != 0:
        print("------ SALIDA COMPLETA ------")
        print(stdout)
        raise RuntimeError("trtexec devolvió un código de error distinto de 0.")

    metrics = parse_trtexec_stdout(stdout)
    metrics["precision"] = precision_mode
    metrics["batch_size"] = batch_size
    metrics["elapsed_wall_time_s"] = t1 - t0

    return metrics


# ============================================================
# Parser de argumentos
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark TensorRT (trtexec) sobre ONNX")

    parser.add_argument(
        "--onnx-path",
        type=str,
        required=True,
        help="Ruta al modelo ONNX exportado.",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["classification", "detection"],
        default="classification",
        help="Tipo de tarea (solo para guardar metadata).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Tamaño de batch utilizado en el benchmark.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Usar modo FP16 en TensorRT (si el hardware lo soporta).",
    )
    parser.add_argument(
        "--trtexec-path",
        type=str,
        default="trtexec",
        help="Ruta al binario trtexec (por defecto, se asume en el PATH).",
    )
    parser.add_argument(
        "--workspace-mb",
        type=int,
        default=1024,
        help="Tamaño de workspace en MB para TensorRT.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=200,
        help="Cantidad de iteraciones de warm-up en trtexec.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=3,
        help="Duración en segundos del benchmark en trtexec.",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Guardar resultados en un JSON al lado del ONNX.",
    )
    parser.add_argument(
        "--json-path",
        type=str,
        default=None,
        help="Ruta explícita para guardar el JSON (si no se especifica, se construye automáticamente).",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    args = parse_args()

    onnx_path = Path(args.onnx_path)
    if not onnx_path.is_file():
        print(f"[ERROR] No se encontró el archivo ONNX: {onnx_path}")
        sys.exit(1)

    print(f"[INFO] ONNX: {onnx_path}")
    print(f"[INFO] Tarea: {args.task}")
    print(f"[INFO] batch_size: {args.batch_size}")
    print(f"[INFO] FP16: {args.fp16}")

    # 1) Info del ONNX
    input_name, chw = get_onnx_input_info(onnx_path)
    print(f"[INFO] Input name: {input_name}")
    print(f"[INFO] Input shape (C,H,W): {chw}")

    # 2) Ejecutar trtexec
    metrics = run_trtexec(
        onnx_path=onnx_path,
        input_name=input_name,
        chw=chw,
        batch_size=args.batch_size,
        fp16=args.fp16,
        trtexec_path=args.trtexec_path,
        workspace_mb=args.workspace_mb,
        warmup=args.warmup,
        duration=args.duration,
    )

    print("\n[RESULTADOS TensorRT]")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # 3) Guardar en JSON (opcional)
    if args.save_json:
        if args.json_path is not None:
            json_path = Path(args.json_path)
        else:
            # Por defecto: mismo nombre que ONNX pero con sufijo _tensorrt_bench.json
            json_path = onnx_path.with_suffix("")  # quitar .onnx
            json_path = json_path.with_name(json_path.name + f"_tensorrt_bench.json")

        out_dict: Dict[str, Any] = {
            "onnx_path": str(onnx_path),
            "task": args.task,
            "batch_size": args.batch_size,
            "precision": metrics.get("precision"),
            "benchmark_tensorrt": metrics,
        }

        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(out_dict, f, indent=4)

        print(f"\n[INFO] Resultados guardados en: {json_path}")


if __name__ == "__main__":
    main()
