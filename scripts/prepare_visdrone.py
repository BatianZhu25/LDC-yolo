"""Convert an official VisDrone DET split to YOLO labels without changing the source."""
from __future__ import annotations
import argparse
import csv
import shutil
from pathlib import Path
from PIL import Image

def convert_annotation(annotation, width, height):
    labels = []
    with annotation.open(encoding="utf-8-sig") as stream:
        for row in csv.reader(stream):
            if not row:
                continue
            x, y, w, h, score, category = map(float, row[:6])
            category = int(category)
            if score == 0 or not 1 <= category <= 10:
                continue
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(width, x + w), min(height, y + h)
            if x2 <= x1 or y2 <= y1:
                continue
            labels.append(f"{category - 1} {(x1+x2)/2/width:.6f} {(y1+y2)/2/height:.6f} {(x2-x1)/width:.6f} {(y2-y1)/height:.6f}")
    return "\n".join(labels) + ("\n" if labels else "")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Official DET split containing images/ and annotations/")
    parser.add_argument("--output", type=Path, required=True, help="New YOLO-format dataset root")
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    args = parser.parse_args()
    images = sorted(p for p in (args.source / "images").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        parser.error("No images found in source/images")
    # Validate all inputs and avoid overwriting an existing conversion.
    for image in images:
        annotation = args.source / "annotations" / (image.stem + ".txt")
        if not annotation.is_file():
            parser.error(f"Missing annotation: {annotation}")
        for output in (args.output / "images" / args.split / image.name,
                       args.output / "labels" / args.split / (image.stem + ".txt")):
            if output.exists():
                parser.error(f"Output already exists: {output}")
    dest_images = args.output / "images" / args.split
    dest_labels = args.output / "labels" / args.split
    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)
    for image in images:
        with Image.open(image) as im:
            width, height = im.size
        text = convert_annotation(args.source / "annotations" / (image.stem + ".txt"), width, height)
        (dest_labels / (image.stem + ".txt")).write_text(text, encoding="utf-8")
        shutil.copy2(image, dest_images / image.name)
    print(f"Prepared {len(images)} images and labels in {args.output} ({args.split}).")

if __name__ == "__main__":
    main()
