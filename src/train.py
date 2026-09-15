import argparse
import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import torch
from ultralytics import YOLO


MODEL_NAME = "yolov8n.pt"
DATASET_PATH = "dataset.yaml"
EPOCHS = 50
BATCH_SIZE = 16
IMAGE_SIZE = 640
PROJECT_DIR = "runs/detect"
EXPERIMENT_NAME = "flir_yolov8n"
WORKERS = 4
PATIENCE = 10


def get_default_device():
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_model(
    model_name=MODEL_NAME,
    dataset_path=DATASET_PATH,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    img_size=IMAGE_SIZE,
    device=None,
    project=PROJECT_DIR,
    name=EXPERIMENT_NAME,
    workers=WORKERS,
    patience=PATIENCE,
):
    if device is None or device == "":
        device = get_default_device()

    print("=" * 60)
    print("FLIR Thermal YOLO Training")
    print("=" * 60)
    print(f"Model:           {model_name}")
    print(f"Dataset config:  {dataset_path}")
    print(f"Epochs:          {epochs}")
    print(f"Batch Size:      {batch_size}")
    print(f"Image Size:      {img_size}")
    print(f"Device:          {device}")
    print(f"Save Path:       {os.path.join(project, name)}")
    print("=" * 60)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset config '{dataset_path}' not found! "
            f"Please run 'python src/prepare_dataset.py' first."
        )

    print(f"\nLoading model: {model_name}...")
    model = YOLO(model_name)

    results = model.train(
        data=dataset_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=project,
        name=name,
        workers=workers,
        patience=patience,
        exist_ok=True,
        verbose=True,
    )

    best_weights = os.path.join(project, name, "weights", "best.pt")
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best model weights saved to: {best_weights}")
    print(f"Training logs saved to:      {os.path.join(project, name)}")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLO on FLIR thermal dataset.")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="YOLO model checkpoint or YAML")
    parser.add_argument("--data", type=str, default=DATASET_PATH, help="Path to dataset.yaml")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE, help="Image size")
    parser.add_argument("--device", type=str, default="", help="Hardware device: '0', 'cpu', 'mps'")
    parser.add_argument("--project", type=str, default=PROJECT_DIR, help="Directory to save runs")
    parser.add_argument("--name", type=str, default=EXPERIMENT_NAME, help="Experiment name")
    parser.add_argument("--workers", type=int, default=WORKERS, help="DataLoader workers")
    parser.add_argument("--patience", type=int, default=PATIENCE, help="Early stopping patience")

    args = parser.parse_args()

    train_model(
        model_name=args.model,
        dataset_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        device=args.device if args.device else None,
        project=args.project,
        name=args.name,
        workers=args.workers,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
