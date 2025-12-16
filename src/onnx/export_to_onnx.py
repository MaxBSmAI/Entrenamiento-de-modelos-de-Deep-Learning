from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

# ============================================================
# Utils
# ============================================================

def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _strip_state_dict_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state.items():
        for pref in ("module.", "model."):
            if k.startswith(pref):
                k = k[len(pref):]
        out[k] = v
    return out

def _load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))

def _infer_num_classes_from_metrics(metrics_path: Path) -> Optional[int]:
    if not metrics_path.exists():
        return None
    try:
        d = _load_json(metrics_path)
    except Exception:
        return None
    if isinstance(d.get("num_classes"), int):
        return int(d["num_classes"])
    for k in ("classes", "class_names", "labels"):
        if isinstance(d.get(k), list):
            return len(d[k])
    return None

def _resolve_device(device_id: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{device_id}")
    return torch.device("cpu")

def _device_tag_from_torch(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device).replace(" ", "_")
    return "CPU"

# ============================================================
# Classification loaders
# ============================================================

def _load_classification_model(model_key: str, num_classes: int) -> nn.Module:
    model_key = model_key.lower()
    if model_key == "resnet50_miniimagenet":
        import torchvision
        m = torchvision.models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if model_key == "vit_b16_miniimagenet":
        import timm
        return timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=num_classes,
        )
    raise ValueError(f"Modelo de clasificación no soportado: {model_key}")

def _resolve_classification_checkpoint(model_key: str, device_tag: str) -> Path:
    return (
        Path("/workspace/result/classification")
        / model_key
        / device_tag
        / f"{model_key}_best.pth"
    )

def _resolve_classification_metrics(model_key: str, device_tag: str) -> Path:
    base = Path("/workspace/result/classification") / model_key / device_tag
    for f in ("metrics.json", f"{model_key}_metrics.json"):
        if (base / f).exists():
            return base / f
    return base / "metrics.json"

# ============================================================
# Detection loaders (torchvision)
# ============================================================

def _load_torchvision_detector(model_key: str) -> nn.Module:
    import torchvision
    model_key = model_key.lower()

    if model_key == "fasterrcnn_resnet50_fpn_coco":
        w = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        return torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=w)

    if model_key == "retinanet_resnet50_fpn_coco":
        w = torchvision.models.detection.RetinaNet_ResNet50_FPN_Weights.DEFAULT
        return torchvision.models.detection.retinanet_resnet50_fpn(weights=w)

    raise ValueError(f"Detector no soportado: {model_key}")

# ============================================================
# Detection ONNX Wrapper (TensorRT-safe)
# ============================================================

class DetectionONNXWrapper(nn.Module):
    """
    Wrapper diseñado explícitamente para TensorRT:
      - input: (B,3,H,W)
      - output:
          boxes  (B, max_det, 4)
          scores (B, max_det)
          labels (B, max_det)
    SIN topk, SIN if dinámicos, SIN tensores vacíos.
    """

    def __init__(self, det_model: nn.Module, max_det: int):
        super().__init__()
        self.det_model = det_model
        self.max_det = int(max_det)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        imgs = torch.unbind(x, dim=0)  # traceable
        outputs = self.det_model(list(imgs))

        B = x.shape[0]
        device = x.device

        boxes_out = torch.zeros((B, self.max_det, 4), device=device)
        scores_out = torch.zeros((B, self.max_det), device=device)
        labels_out = torch.zeros((B, self.max_det), dtype=torch.int64, device=device)

        for i in range(B):
            boxes = outputs[i]["boxes"]
            scores = outputs[i]["scores"]
            labels = outputs[i]["labels"]

            z_boxes = torch.zeros((self.max_det, 4), device=device)
            z_scores = torch.zeros((self.max_det,), device=device)
            z_labels = torch.zeros((self.max_det,), dtype=torch.int64, device=device)

            boxes = torch.cat([boxes, z_boxes], dim=0)[: self.max_det]
            scores = torch.cat([scores, z_scores], dim=0)[: self.max_det]
            labels = torch.cat([labels, z_labels], dim=0)[: self.max_det]

            boxes_out[i] = boxes
            scores_out[i] = scores
            labels_out[i] = labels

        return boxes_out, scores_out, labels_out

# ============================================================
# Metadata
# ============================================================

@dataclass
class ExportMetadata:
    task: str
    model: str
    device_id: int
    device_tag: str
    checkpoint: Optional[str]
    img_size: int
    batch_size: int
    opset: int
    max_det: Optional[int]
    onnx_path: str

# ============================================================
# Exporters
# ============================================================

def export_classification(
    model_key: str,
    device: torch.device,
    device_tag: str,
    img_size: int,
    batch_size: int,
    opset: int,
    output_dir: Path,
    checkpoint: Optional[Path],
    save_export_json: bool,
) -> Path:

    output_dir = output_dir / "classification"
    _safe_mkdir(output_dir)

    metrics = _resolve_classification_metrics(model_key, device_tag)
    num_classes = _infer_num_classes_from_metrics(metrics) or 100

    if checkpoint is None:
        checkpoint = _resolve_classification_checkpoint(model_key, device_tag)

    model = _load_classification_model(model_key, num_classes)
    model.to(device).eval()

    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    state = _strip_state_dict_prefix(state)
    model.load_state_dict(state, strict=False)

    onnx_path = output_dir / f"{model_key}__{device_tag}.onnx"
    dummy = torch.randn(batch_size, 3, img_size, img_size, device=device)

    torch.onnx.export(
        model,
        dummy,
        onnx_path.as_posix(),
        opset_version=opset,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )

    if save_export_json:
        meta = ExportMetadata(
            task="classification",
            model=model_key,
            device_id=device.index if device.type == "cuda" else -1,
            device_tag=device_tag,
            checkpoint=str(checkpoint),
            img_size=img_size,
            batch_size=batch_size,
            opset=opset,
            max_det=None,
            onnx_path=str(onnx_path),
        )
        onnx_path.with_suffix(".export.json").write_text(
            json.dumps(asdict(meta), indent=2)
        )

    return onnx_path

def export_detection(
    model_key: str,
    device: torch.device,
    device_tag: str,
    img_size: int,
    batch_size: int,
    opset: int,
    output_dir: Path,
    max_det: int,
    save_export_json: bool,
) -> Path:

    output_dir = output_dir / "detection"
    _safe_mkdir(output_dir)

    det = _load_torchvision_detector(model_key)
    det.to(device).eval()

    wrapper = DetectionONNXWrapper(det, max_det).to(device).eval()

    onnx_path = output_dir / f"{model_key}__{device_tag}.onnx"
    dummy = torch.randn(batch_size, 3, img_size, img_size, device=device)

    torch.onnx.export(
        wrapper,
        dummy,
        onnx_path.as_posix(),
        opset_version=opset,
        input_names=["input"],
        output_names=["boxes", "scores", "labels"],
        dynamic_axes=None,  # CRÍTICO PARA TENSORRT
    )

    if save_export_json:
        meta = ExportMetadata(
            task="detection",
            model=model_key,
            device_id=device.index if device.type == "cuda" else -1,
            device_tag=device_tag,
            checkpoint=None,
            img_size=img_size,
            batch_size=batch_size,
            opset=opset,
            max_det=max_det,
            onnx_path=str(onnx_path),
        )
        onnx_path.with_suffix(".export.json").write_text(
            json.dumps(asdict(meta), indent=2)
        )

    return onnx_path

# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["classification", "detection"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--device-tag", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-det", type=int, default=100)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--output-dir", default="/workspace/models/onnx")
    p.add_argument("--save-export-json", action="store_true")
    return p

def main() -> None:
    args = build_parser().parse_args()

    device = _resolve_device(args.device_id)
    device_tag = args.device_tag or _device_tag_from_torch(device)

    out = Path(args.output_dir)
    _safe_mkdir(out)

    if args.task == "classification":
        export_classification(
            args.model,
            device,
            device_tag,
            args.img_size,
            args.batch_size,
            args.opset,
            out,
            Path(args.checkpoint) if args.checkpoint else None,
            args.save_export_json,
        )
    else:
        export_detection(
            args.model,
            device,
            device_tag,
            args.img_size,
            args.batch_size,
            args.opset,
            out,
            args.max_det,
            args.save_export_json,
        )

    print("[DONE] export_to_onnx.py finalizado correctamente.")

if __name__ == "__main__":
    main()
