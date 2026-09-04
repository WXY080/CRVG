"""Draw current and selected boxes on real source images."""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw
from analysis.common import aligned
from crvg.utils.data import read_json, row_key, row_image, current_bbox
from crvg.utils.bbox import xywh_to_xyxy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--image-root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--show-ground-truth", action="store_true")
    args = parser.parse_args()
    pair = next(((a, b) for a, b in aligned(read_json(args.before), read_json(args.after))
                 if row_key(a) == args.sample_id), None)
    if pair is None:
        parser.error("sample-id not found")
    a, b = pair
    with Image.open(row_image(a, args.image_root)) as raw:
        image = raw.convert("RGB")
    draw = ImageDraw.Draw(image)
    boxes = [(current_bbox(a), "Current", "#0072B2"), (current_bbox(b), "Selected", "#D55E00")]
    if args.show_ground_truth and b.get("gt_bbox"):
        boxes.append((b["gt_bbox"], "Ground truth", "#009E73"))
    for box, label, color in boxes:
        coords = xywh_to_xyxy(box)
        draw.rectangle(coords, outline=color, width=3)
        draw.text((max(0, coords[0]), max(0, coords[1]-12)), label, fill=color,
                  stroke_width=1, stroke_fill="white")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
