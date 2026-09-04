"""Explicit data contracts, stable sample keys and atomic artifact writes."""
import hashlib
import json
import os
import re
from pathlib import Path

from crvg.utils.bbox import iou_xywh

DATASETS = ("refcoco_val", "refcoco_testA", "refcoco_testB", "refcoco+_val",
            "refcoco+_testA", "refcoco+_testB", "refcocog_val", "refcocog_test")


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def row_key(row):
    if row.get("dataset_index") is None:
        raise ValueError("Missing dataset_index; prepare data before inference")
    return str(row["dataset_index"])


def index_rows(rows):
    index = {}
    for row in rows:
        key = row_key(row)
        if key in index:
            raise ValueError(f"Duplicate dataset_index: {key}")
        index[key] = row
    return index


def results(payload):
    rows = payload["results"] if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("Expected a JSON results array")
    index_rows(rows)
    return rows


def fingerprint(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def check_source(cache, source):
    expected = cache.get("meta", {}).get("source_sha256")
    if not expected or expected != fingerprint(source):
        raise ValueError("Cache provenance mismatch; regenerate evidence for this exact source")


def current_bbox(row):
    box = row.get("pred_bbox") or (row.get("selection") or {}).get("bbox")
    if box is not None:
        return box
    candidates = row.get("candidates", [])
    return next((c["bbox"] for c in candidates if c.get("source") == "current_system"),
                candidates[0]["bbox"] if candidates else None)


def current_iou(row):
    if row.get("gt_bbox") is not None:
        return iou_xywh(current_bbox(row), row["gt_bbox"])
    return row.get("iou", row.get("iou_selected", row.get("iou_greedy")))


def current_candidate(row):
    box = current_bbox(row)
    return {"bbox": box, "iou": current_iou(row), "source": "current_system"} if box else None


def set_prediction(row, box, source):
    output = dict(row)
    output["pred_bbox"] = list(box)
    output["selection"] = {"bbox": list(box), "source": source}
    output["iou"] = iou_xywh(box, row["gt_bbox"]) if row.get("gt_bbox") else None
    output["iou_selected"] = output["iou"]
    output["correct"] = int(output["iou"] >= .5) if output["iou"] is not None else None
    return output


def resolve_image_path(img_root, img_name):
    name = str(img_name)
    base = name.replace("\\", "/").split("/")[-1]
    candidates = [name, os.path.join(img_root, name), os.path.join(img_root, base),
                  os.path.join(img_root, "train2014", base)]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"Image not found: {name} (root={img_root})")


def row_image(row, root=None):
    name = row.get("image") or row.get("image_path")
    if not name:
        raise ValueError("Missing image path")
    return resolve_image_path(root or "", name)


def extract_expression(value):
    if isinstance(value, dict):
        value = value.get("expression") or value.get("expr") or value.get("text") or value.get("category") or ""
    text = str(value or "").strip()
    match = re.search(r'describes:\s*"(.+)"\s*\.?$', text, re.I)
    return match.group(1) if match else text


def normalize_sample(data, idx, img_root):
    name = data.get("file_name") or data.get("image") or data.get("image_path")
    if not name and ("img_id" in data or "image_id" in data):
        raw_id = str(data.get("img_id", data.get("image_id")))
        match = re.search(r"(\d{12})", raw_id)
        image_id = match.group(1) if match else f"{int(raw_id):012d}"
        name = f"COCO_train2014_{image_id}.jpg"
    if not name:
        raise ValueError(f"Missing image at row {idx}")
    image_path = resolve_image_path(img_root, name)
    text = data.get("sents") or data.get("sent") or data.get("problem") or extract_expression(data)
    if isinstance(text, list):
        if len(text) != 1:
            raise ValueError("Explode multi-sentence rows with tools.prepare_data first")
        text = text[0]
    if isinstance(text, dict):
        text = text.get("sent") or text.get("text")
    if not text:
        raise ValueError(f"Missing expression at row {idx}")
    if "solution" in data:
        box = data["solution"]
        if isinstance(box, str):
            box = json.loads(box)
        x1, y1, x2, y2 = map(float, box)
        gt = [x1, y1, x2 - x1, y2 - y1]
    else:
        gt = data.get("gt_bbox", data.get("bbox"))
        gt = list(map(float, gt)) if gt is not None else None
    return {"dataset_index": data.get("dataset_index", idx), "image_path": image_path,
            "expr": extract_expression(text), "gt_bbox_xywh": gt,
            "gt_bbox_xyxy": [gt[0], gt[1], gt[0] + gt[2], gt[1] + gt[3]] if gt else None,
            "image_size": [data.get("width", 0), data.get("height", 0)]}
