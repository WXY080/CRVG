"""Equivariant Candidate Expansion (ECE).

Creates geometric views (horizontal flip + 3 padded canvases) and re-runs
the backbone on each, then maps predictions back to original coordinates.
Only applied to low-consistency samples (min pairwise IoU < agree_skip_iou).

Usage:
    python -m crvg.candidate_generation.ece \
        --model-path /path/to/model --backbone internvl \
        --input logs/rec_results_refcoco_val.json \
        --save logs/equivariant_candidates_refcoco_val.json \
        --image-root /path/to/coco/train2014/ \
        --agree-skip-iou 0.50
"""
import argparse
import math
import re

from PIL import Image, ImageOps
from tqdm import tqdm

from crvg.utils.bbox import iou_xywh, min_pairwise_iou

SPATIAL_SWAP = {
    "left": "right", "right": "left",
    "leftmost": "rightmost", "rightmost": "leftmost",
}


def swap_horizontal_words(text):
    pattern = re.compile(r"\b(leftmost|rightmost|left|right)\b", re.IGNORECASE)
    def replace(match):
        source = match.group(0)
        target = SPATIAL_SWAP.get(source.lower(), source)
        return target.capitalize() if source[:1].isupper() else target
    return pattern.sub(replace, text)


def padded_view(image, factor, anchor):
    width, height = image.size
    canvas_w = max(width, int(round(width * factor)))
    canvas_h = max(height, int(round(height * factor)))
    if anchor == "top_left":
        offset = (0, 0)
    elif anchor == "bottom_right":
        offset = (canvas_w - width, canvas_h - height)
    else:
        offset = ((canvas_w - width) // 2, (canvas_h - height) // 2)
    canvas = Image.new("RGB", (canvas_w, canvas_h), (127, 127, 127))
    canvas.paste(image, offset)
    return canvas, {"kind": "offset", "offset": list(offset)}


def build_views(image, expr, pad_factor):
    views = [{
        "name": "hflip",
        "image": ImageOps.mirror(image),
        "expr": swap_horizontal_words(expr),
        "inverse": {"kind": "hflip"},
    }]
    for anchor in ("top_left", "center", "bottom_right"):
        transformed, inverse = padded_view(image, pad_factor, anchor)
        views.append({"name": f"pad_{anchor}", "image": transformed,
                      "expr": expr, "inverse": inverse})
    return views


def clip_bbox(bbox, image_size, min_retained=0.50):
    w, h = image_size
    x, y, bw, bh = [float(v) for v in bbox]
    if bw <= 0 or bh <= 0 or not all(math.isfinite(v) for v in bbox):
        return None
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(w), x + bw), min(float(h), y + bh)
    retained = max(0, x2 - x1) * max(0, y2 - y1) / max(bw * bh, 1e-6)
    if x2 <= x1 or y2 <= y1 or retained < min_retained:
        return None
    return [x1, y1, x2 - x1, y2 - y1]


def inverse_bbox(pred_xywh, inverse, original_size):
    if not pred_xywh:
        return None
    x, y, w, h = [float(v) for v in pred_xywh]
    if inverse["kind"] == "hflip":
        x = float(original_size[0]) - (x + w)
    elif inverse["kind"] == "offset":
        x -= float(inverse["offset"][0])
        y -= float(inverse["offset"][1])
    return clip_bbox([x, y, w, h], original_size)


def expand(data, engine, image_root, gamma0=.5, pad_factor=1.25):
    from crvg.utils.data import results, row_key, row_image, extract_expression, fingerprint
    records = {}
    for row in tqdm(results(data), desc="ECE"):
        boxes = [c["bbox"] for c in row.get("candidates", [])]
        routed = len(boxes) >= 2 and min_pairwise_iou(boxes) < gamma0
        record = {"dataset_index": row["dataset_index"], "routed": routed,
                  "gamma0": gamma0, "transformed_candidates": []}
        if routed:
            with Image.open(row_image(row, image_root)) as raw_image:
                image = raw_image.convert("RGB")
            views = build_views(image, extract_expression(row), pad_factor)
            predictions = engine.predict([v["image"] for v in views], [v["expr"] for v in views])
            for view, outputs in zip(views, predictions):
                box = inverse_bbox(outputs[0]["bbox"], view["inverse"], image.size)
                if box:
                    record["transformed_candidates"].append({
                        "bbox": box, "source": "equivariant_" + view["name"], "view": view["name"],
                        "score": 0., "iou": iou_xywh(box, row["gt_bbox"]) if row.get("gt_bbox") else None})
        records[row_key(row)] = record
    return {"meta": {"source_sha256": fingerprint(data), "complete": True,
                      "gamma0": gamma0, "pad_factor": pad_factor}, "records": records}


def main():
    from crvg.candidate_generation.bon_dump import add_engine_args, engine_from_args
    from crvg.utils.data import read_json, write_json
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--backbone", choices=("internvl", "vlmr1"), required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--save", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--agree-skip-iou", type=float, default=.5)
    parser.add_argument("--pad-factor", type=float, default=1.25)
    add_engine_args(parser)
    args = parser.parse_args()
    if not 0 < args.agree_skip_iou <= 1.01 or args.pad_factor < 1:
        parser.error("Invalid consensus threshold or pad factor")
    data = read_json(args.input)
    engine = engine_from_args(args, args.backbone)
    output = expand(data, engine, args.image_root, args.agree_skip_iou, args.pad_factor)
    write_json(args.save, output)


if __name__ == "__main__":
    main()
