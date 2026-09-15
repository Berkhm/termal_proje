import argparse
import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import torch
from ultralytics import YOLO


DEFAULT_WEIGHTS = "runs/detect/flir_yolov8n/weights/best.pt"
FALLBACK_WEIGHTS = "yolov8n.pt"
DATASET_PATH = "dataset.yaml"
IMAGE_SIZE = 640
BATCH_SIZE = 16


def get_default_device():
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def evaluate_model(weights_path=DEFAULT_WEIGHTS, dataset_path=DATASET_PATH, img_size=IMAGE_SIZE, batch_size=BATCH_SIZE, device=None):
    if device is None or device == "":
        device = get_default_device()

    if not os.path.exists(weights_path):
        print(f"Warning: Specified weights '{weights_path}' not found on disk.")
        print(f"Falling back to pretrained checkpoint '{FALLBACK_WEIGHTS}' (will download if not cached).")
        weights_path = FALLBACK_WEIGHTS

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset config '{dataset_path}' not found! Run 'python src/prepare_dataset.py' first.")

    print("=" * 65)
    print("FLIR Thermal Model Evaluation")
    print("=" * 65)
    print(f"Model Weights:  {weights_path}")
    print(f"Dataset YAML:   {dataset_path}")
    print(f"Image Size:     {img_size}")
    print(f"Device:         {device}")
    print("=" * 65)

    model = YOLO(weights_path)

    metrics = model.val(
        data=dataset_path,
        split="val",
        imgsz=img_size,
        batch=batch_size,
        device=device,
        verbose=False,
    )

    box = metrics.box
    mean_precision = box.mp
    mean_recall = box.mr
    map50 = box.map50
    map50_95 = box.map

    print("\n" + "=" * 65)
    print("OVERALL DETECTION METRICS")
    print("=" * 65)
    print(f"  Precision:   {mean_precision:.4f}")
    print(f"  Recall:      {mean_recall:.4f}")
    print(f"  mAP@50:      {map50:.4f}")
    print(f"  mAP@50-95:   {map50_95:.4f}")
    print("=" * 65)

    print("\nPER-CLASS METRICS BREAKDOWN:")
    print(f"{'Class':<18} {'Precision':<12} {'Recall':<12} {'mAP@50':<12} {'mAP@50-95':<12}")
    print("-" * 65)

    class_names = model.names
    for i, c in enumerate(box.ap_class_index):
        c_name = class_names.get(c, f"Class {c}")
        p = box.p[i] if i < len(box.p) else 0.0
        r = box.r[i] if i < len(box.r) else 0.0
        ap50 = box.ap50[i] if i < len(box.ap50) else 0.0
        ap = box.ap[i] if i < len(box.ap) else 0.0
        print(f"{c_name:<18} {p:<12.4f} {r:<12.4f} {ap50:<12.4f} {ap:<12.4f}")
    print("-" * 65)

    return {
        "precision": mean_precision,
        "recall": mean_recall,
        "map50": map50,
        "map50_95": map50_95,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO model on FLIR thermal validation split.")
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS, help="Path to trained weights (.pt)")
    parser.add_argument("--data", type=str, default=DATASET_PATH, help="Path to dataset.yaml")
    parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE, help="Validation image size")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE, help="Validation batch size")
    parser.add_argument("--device", type=str, default="", help="Hardware device: '0', 'cpu', 'mps'")

    args = parser.parse_args()

    evaluate_model(
        weights_path=args.weights,
        dataset_path=args.data,
        img_size=args.imgsz,
        batch_size=args.batch,
        device=args.device if args.device else None,
    )


if __name__ == "__main__":
    main()
