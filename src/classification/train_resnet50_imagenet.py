import os
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import timm

from data.dataloaders import get_miniimagenet_dataloaders
from utils.utils_metrics import classification_metrics
from utils.utils_plot import plot_loss, plot_accuracy
from utils.utils_benchmark import (
    measure_inference_latency,
    measure_fps,
    gpu_memory_usage,
)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(num_classes: int):
    """
    ResNet-50 preentrenada en ImageNet, ajustada a mini-ImageNet (100 clases).
    """
    model = timm.create_model("resnet50", pretrained=False)
    in_features = model.get_classifier().in_features
    model.reset_classifier(num_classes=num_classes)
    # Por compatibilidad con variantes tipo torchvision
    if hasattr(model, "fc"):
        model.fc = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    n_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

    epoch_loss = running_loss / max(n_samples, 1)
    return epoch_loss


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    n_samples = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            n_samples += batch_size

            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    epoch_loss = running_loss / max(n_samples, 1)
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    return epoch_loss, all_preds, all_labels


def main():
    # -------------------------------------------------
    # Configuración general
    # -------------------------------------------------
    device = get_device()
    print(f"Usando dispositivo: {device}")

    project_root = Path(__file__).resolve().parents[2]

    result_dir = project_root / "result" / "classification" / "resnet50_miniimagenet"
    models_dir = project_root / "models" / "classification"
    result_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Hiperparámetros
    batch_size = 8          # estándar para comparar RTX 4080 vs A100
    img_size = 224
    num_epochs = 10        # máximo; early stopping cortará antes
    lr = 1e-4
    weight_decay = 1e-4

    # Early stopping
    patience = 5            # epochs sin mejora en val_loss
    patience_counter = 0

    # -------------------------------------------------
    # DataLoaders desde HuggingFace (map-style)
    # -------------------------------------------------
    # streaming=False: dataset cacheado dentro del contenedor,
    # NO en el notebook local.
    train_loader, val_loader, test_loader, num_classes = \
        get_miniimagenet_dataloaders(
            batch_size=batch_size,
            img_size=img_size,
            streaming=False,
        )

    print(f"Número de clases (mini-ImageNet): {num_classes}")

    # -------------------------------------------------
    # Modelo, loss, optimizador
    # -------------------------------------------------
    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_model_path = models_dir / "resnet50_miniimagenet_best.pth"

    train_losses, val_losses, val_accuracies = [], [], []

    # -------------------------------------------------
    # Loop de entrenamiento con EARLY STOPPING
    # -------------------------------------------------
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
        metrics_val = classification_metrics(val_labels, val_preds)
        val_acc = metrics_val["accuracy"]

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        elapsed = time.time() - epoch_start
        print(
            f"[Epoch {epoch}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | Tiempo: {elapsed:.2f} s"
        )

        # -------- Early Stopping basado en val_loss --------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            torch.save(model.state_dict(), best_model_path)
            print(f"  👉 Mejor modelo actualizado: {best_model_path}")
        else:
            patience_counter += 1
            print(f"  ⚠️ No mejora en validación ({patience_counter}/{patience})")

            if patience_counter >= patience:
                print("\n🛑 EARLY STOPPING ACTIVADO")
                print(f"Se detiene el entrenamiento en epoch {epoch}")
                break

    # -------------------------------------------------
    # Graficar métricas de entrenamiento
    # -------------------------------------------------
    plot_loss(train_losses, result_dir / "train_loss.png")
    plot_loss(val_losses, result_dir / "val_loss.png")
    plot_accuracy(val_accuracies, result_dir / "val_accuracy.png")

    # -------------------------------------------------
    # Evaluación final en TEST usando el MEJOR modelo
    # -------------------------------------------------
    print("\nEvaluando en TEST con el mejor modelo guardado...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_loss, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    test_metrics = classification_metrics(test_labels, test_preds)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")

    # -------------------------------------------------
    # Benchmark computacional (latencia, FPS, VRAM)
    # -------------------------------------------------
    dummy_input = torch.randn(batch_size, 3, img_size, img_size, device=device)

    mean_lat, std_lat = measure_inference_latency(model, dummy_input, runs=50)
    fps = measure_fps(model, dummy_input, duration_seconds=5)
    vram_used, vram_total = gpu_memory_usage()

    benchmark_info = {
        "batch_size": batch_size,
        "img_size": img_size,
        "device": str(device),
        "mean_latency_ms": float(mean_lat),
        "std_latency_ms": float(std_lat),
        "fps": float(fps),
        "vram_used_mb": float(vram_used),
        "vram_total_mb": float(vram_total),
    }

    # -------------------------------------------------
    # Guardar métricas y benchmark en JSON
    # -------------------------------------------------
    summary = {
        "test_loss": float(test_loss),
        "test_metrics": test_metrics,
        "benchmark": benchmark_info,
        "num_epochs_trained": len(train_losses),
        "batch_size": batch_size,
        "img_size": img_size,
        "early_stopping_patience": patience,
    }

    metrics_path = result_dir / "resnet50_miniimagenet_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\nMétricas y benchmark guardados en: {metrics_path}")
    print("Entrenamiento ResNet50 en mini-ImageNet completado.")


if __name__ == "__main__":
    main()
