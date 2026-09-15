import argparse
import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

from pathlib import Path
import torch
from ultralytics import YOLO


DEFAULT_WEIGHTS = "runs/detect/flir_yolov8n/weights/best.pt"
FALLBACK_WEIGHTS = "yolov8n.pt"
DEFAULT_SOURCE = "examples/sample_thermal_1.jpg"
DEFAULT_OUTPUT_DIR = "results/predictions"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45


def get_default_device():
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_prediction(
    weights_path=DEFAULT_WEIGHTS,
    source=DEFAULT_SOURCE,
    output_dir=DEFAULT_OUTPUT_DIR,
    conf=CONF_THRESHOLD,
    iou=IOU_THRESHOLD,
    device=None,
):
    if device is None or device == "":
        device = get_default_device()

    if not os.path.exists(weights_path):
        print(f"Notice: Trained weights '{weights_path}' not found on disk.")
        print(f"Using pretrained '{FALLBACK_WEIGHTS}' (will download if not cached).")
        weights_path = FALLBACK_WEIGHTS

    if not os.path.exists(source):
        raise FileNotFoundError(f"Input source '{source}' does not exist.")

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("FLIR Thermal Object Detection - Inference")
    print("=" * 60)
    print(f"Weights:      {weights_path}")
    print(f"Source:       {source}")
    print(f"Output Dir:   {output_dir}")
    print(f"Confidence:   {conf}")
    print(f"Device:       {device}")
    print("=" * 60)

    model = YOLO(weights_path)

    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    is_video = Path(source).suffix.lower() in video_extensions

    if is_video:
        print(f"\nProcessing video input: {source}...")
        results = model.predict(
            source=source,
            conf=conf,
            iou=iou,
            device=device,
            save=True,
            project=output_dir,
            name="video_output",
            exist_ok=True,
        )
        print(f"Video inference completed! Saved to {os.path.join(output_dir, 'video_output')}")
        return results

    print(f"\nProcessing image input: {source}...")
    results = model.predict(
        source=source,
        conf=conf,
        iou=iou,#birbirine çok benzeyen kutuların nasıl bastırılacağını belirleyen threshold
        device=device,
        save=False,
        verbose=False,
    )

    print(f"\nDetected objects across {len(results)} image(s):")
    print("-" * 60)

    for i, res in enumerate(results):
        orig_filename = os.path.basename(res.path) if res.path else f"image_{i+1}.jpg"
        save_name = f"pred_{orig_filename}"
        save_path = os.path.join(output_dir, save_name)

        res.save(filename=save_path)

        boxes = res.boxes
        num_detections = len(boxes) if boxes is not None else 0

        class_counts = {}
        if boxes is not None:
            for b in boxes:
                cls_id = int(b.cls[0].item())
                cls_name = model.names.get(cls_id, f"class_{cls_id}")
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

        summary_str = ", ".join([f"{count} {cls_name}" for cls_name, count in class_counts.items()])
        if not summary_str:
            summary_str = "No objects detected above confidence threshold"

        print(f"[{i+1}/{len(results)}] {orig_filename}")
        print(f"       Detections: {summary_str} (Total: {num_detections})")
        print(f"       Saved:      {save_path}")

    print("-" * 60)
    print(f"All inference results saved to: {output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run thermal object detection inference.")
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS, help="Path to weights file (.pt)")
    parser.add_argument("--source", type=str, default=DEFAULT_SOURCE, help="Path to image, directory, or video")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save visual results")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=IOU_THRESHOLD, help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default="", help="Hardware device: '0', 'cpu', 'mps'")

    args = parser.parse_args()

    run_prediction(
        weights_path=args.weights,
        source=args.source,
        output_dir=args.output_dir,
        conf=args.conf,
        iou=args.iou,
        device=args.device if args.device else None,
    )


if __name__ == "__main__":
    main()
