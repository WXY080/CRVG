"""Normalize local benchmark annotations without downloading or dropping samples."""
import argparse
from PIL import Image
from crvg.utils.data import read_json, write_json, normalize_sample
from crvg.utils.bbox import valid_box


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    raw = read_json(args.input)
    raw = raw["results"] if isinstance(raw, dict) else raw
    rows = []
    for item in raw:
        sentences = item.get("sents")
        variants = sentences if isinstance(sentences, list) else [sentences]
        for sentence in variants:
            row = dict(item)
            row.pop("dataset_index", None)
            if sentence is not None:
                row["sents"] = [sentence]
            sample = normalize_sample(row, len(rows), args.image_root)
            gt = sample["gt_bbox_xywh"]
            if not valid_box(gt):
                raise ValueError(f"Invalid GT bbox for row {len(rows)}")
            with Image.open(sample["image_path"]) as image:
                width, height = image.size
            rows.append({"dataset_index": len(rows), "dataset": args.dataset,
                         "image": sample["image_path"], "text": sample["expr"],
                         "bbox": gt, "width": width, "height": height})
    write_json(args.output, rows)
    print(f"Prepared {len(rows)} referring expressions")


if __name__ == "__main__":
    main()
