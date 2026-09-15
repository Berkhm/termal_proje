import os
import json
import shutil
import argparse
from pathlib import Path
import yaml


def find_default_flir_dir() -> str:
    candidates = [
        "../../FLIR_ADAS_v2",
        "../FLIR_ADAS_v2",
        "./FLIR_ADAS_v2",
        "/Users/berkhokmen/Desktop/python kod/yolo_termal_proje/FLIR_ADAS_v2",
    ]
    for c in candidates:
        if os.path.isdir(c) and os.path.isdir(os.path.join(c, "images_thermal_train")):
            return os.path.abspath(c)
    return "../../FLIR_ADAS_v2"


def convert_coco_bbox_to_yolo(bbox, img_width, img_height):#bbox ve imagenin boyutu
    x_min, y_min, w, h = bbox #box un boyutu

    if w <= 0 or h <= 0 or img_width <= 0 or img_height <= 0:
        return None

    x_center = (x_min + w / 2.0) / img_width
    y_center = (y_min + h / 2.0) / img_height
    norm_w = w / img_width #burda normalize yapıyoruz zaten
    norm_h = h / img_height

    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    norm_w = max(0.0, min(1.0, norm_w)) #burda eğer kutu taşıyorsa falan just in case bir daha normalize
    norm_h = max(0.0, min(1.0, norm_h))

    return x_center, y_center, norm_w, norm_h


def discover_active_categories(flir_dir):
    train_json_path = os.path.join(flir_dir, "images_thermal_train", "coco.json")
    with open(train_json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    category_names = {c["id"]: c["name"] for c in coco_data.get("categories", [])}

    active_ids = set()
    for ann in coco_data.get("annotations", []):
        active_ids.add(ann["category_id"])

    sorted_active_ids = sorted(list(active_ids))

    coco_to_yolo = {coco_id: yolo_id for yolo_id, coco_id in enumerate(sorted_active_ids)}
    yolo_names = {yolo_id: category_names.get(coco_id, f"class_{coco_id}")
                  for yolo_id, coco_id in enumerate(sorted_active_ids)}

    return coco_to_yolo, yolo_names


def process_split(split_name, flir_dir, output_dir, coco_to_yolo, symlink=False, max_samples=None):
    split_folder = f"images_thermal_{split_name}"
    split_dir = os.path.join(flir_dir, split_folder)
    coco_path = os.path.join(split_dir, "coco.json")

    if not os.path.exists(coco_path):
        raise FileNotFoundError(f"Annotation file not found: {coco_path}")

    print(f"\nProcessing {split_name} split from: {split_dir}")
    with open(coco_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    images = coco_data.get("images", [])
    annotations = coco_data.get("annotations", [])

    if max_samples:
        images = images[:max_samples]
        print(f"Limiting to first {max_samples} images for {split_name}.")

    ann_by_image = {}
    for ann in annotations:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    img_dest_dir = os.path.join(output_dir, "images", split_name)
    lbl_dest_dir = os.path.join(output_dir, "labels", split_name)
    os.makedirs(img_dest_dir, exist_ok=True)
    os.makedirs(lbl_dest_dir, exist_ok=True)

    copied_images = 0
    total_labels = 0

    for i, img_info in enumerate(images):
        img_id = img_info["id"]
        rel_path = img_info["file_name"]
        src_img_path = os.path.join(split_dir, rel_path)

        if not os.path.exists(src_img_path):
            continue

        base_name = os.path.basename(rel_path)
        dst_img_path = os.path.join(img_dest_dir, base_name)
        dst_lbl_path = os.path.join(lbl_dest_dir, os.path.splitext(base_name)[0] + ".txt")

        if not os.path.exists(dst_img_path):
            if symlink: #kopyalama işlemi burada
                os.symlink(os.path.abspath(src_img_path), dst_img_path)
            else:
                shutil.copy2(src_img_path, dst_img_path)
        copied_images += 1

        img_w = img_info.get("width", 640)
        img_h = img_info.get("height", 512)
        img_anns = ann_by_image.get(img_id, [])

        label_lines = []
        for ann in img_anns:
            coco_cat_id = ann.get("category_id")
            if coco_cat_id not in coco_to_yolo:
                continue

            yolo_class_id = coco_to_yolo[coco_cat_id]
            yolo_bbox = convert_coco_bbox_to_yolo(ann["bbox"], img_w, img_h)
            if yolo_bbox is not None:
                xc, yc, nw, nh = yolo_bbox
                label_lines.append(f"{yolo_class_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
                total_labels += 1

        with open(dst_lbl_path, "w", encoding="utf-8") as lf:
            if label_lines:
                lf.write("\n".join(label_lines) + "\n")

        if (i + 1) % 1000 == 0 or (i + 1) == len(images):
            print(f"[{split_name}] Processed {i + 1}/{len(images)} images ({total_labels} annotations)")

    print(f"[{split_name}] Completed: {copied_images} images, {total_labels} bounding boxes written.")


def generate_dataset_yaml(output_dir, yolo_names, yaml_path):
    abs_output = os.path.abspath(output_dir)
    yaml_dir = os.path.dirname(os.path.abspath(yaml_path))

    try:
        rel_dataset_path = os.path.relpath(abs_output, yaml_dir)
    except ValueError:
        rel_dataset_path = abs_output

    yaml_content = {
        "path": rel_dataset_path,
        "train": "images/train",
        "val": "images/val",
        "names": yolo_names,
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    print(f"\nGenerated YOLO configuration at: {yaml_path}")
    print(f"Classes ({len(yolo_names)}):")
    for cid, cname in yolo_names.items():
        print(f"  {cid}: {cname}")


def main():
    parser = argparse.ArgumentParser(description="Convert FLIR ADAS v2 thermal dataset to YOLO format.")
    parser.add_argument(
        "--flir-dir",
        type=str,
        default=find_default_flir_dir(),
        help="Path to the downloaded FLIR_ADAS_v2 root directory."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset",
        help="Target folder for processed YOLO dataset."
    )
    parser.add_argument(
        "--yaml-path",
        type=str,
        default="dataset.yaml",
        help="Path to output dataset.yaml file."
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Create symbolic links to images instead of copying to save disk space."
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of images per split (for quick testing/debugging)."
    )

    args = parser.parse_args()

    print("=" * 60)
    print("FLIR ADAS v2 Thermal Dataset Preparation for YOLO")
    print("=" * 60)
    print(f"FLIR Source Directory: {args.flir_dir}")
    print(f"Output Dataset Folder: {args.output_dir}")
    print(f"dataset.yaml Target:   {args.yaml_path}")
    print(f"Use Symlinks:          {args.symlink}")
    if args.max_samples:
        print(f"Max Samples per Split: {args.max_samples}")

    if not os.path.isdir(args.flir_dir):
        raise FileNotFoundError(f"FLIR dataset directory not found at: {args.flir_dir}")

    coco_to_yolo, yolo_names = discover_active_categories(args.flir_dir)

    process_split("train", args.flir_dir, args.output_dir, coco_to_yolo,
                  symlink=args.symlink, max_samples=args.max_samples)
    process_split("val", args.flir_dir, args.output_dir, coco_to_yolo,
                  symlink=args.symlink, max_samples=args.max_samples)

    generate_dataset_yaml(args.output_dir, yolo_names, args.yaml_path)

    print("\nDataset preparation completed successfully!")


if __name__ == "__main__":
    main()
