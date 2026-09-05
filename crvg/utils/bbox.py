"""Bounding box utilities shared across CRVG."""
import math


def iou_xywh(a, b):
    """IoU between two xywh format boxes."""
    if not valid_box(a) or not valid_box(b):
        return 0.0
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return float(inter) / union if union > 0 else 0.0


def xywh_to_xyxy(b):
    return [b[0], b[1], b[0] + b[2], b[1] + b[3]]


def xyxy_to_xywh(b):
    return [b[0], b[1], b[2] - b[0], b[3] - b[1]]


def rescale_xyxy(bbox, source_size, target_size):
    """Map an xyxy box between pixel coordinate systems."""
    source_w, source_h = map(float, source_size)
    target_w, target_h = map(float, target_size)
    if not all(math.isfinite(value) and value > 0
               for value in (source_w, source_h, target_w, target_h)):
        raise ValueError("Source and target image sizes must be finite and positive")
    x1, y1, x2, y2 = map(float, bbox)
    return [x1 * target_w / source_w, y1 * target_h / source_h,
            x2 * target_w / source_w, y2 * target_h / source_h]


def min_pairwise_iou(boxes):
    """Minimum pairwise IoU across a list of xywh boxes (consistency signal)."""
    if len(boxes) < 2:
        return 1.0
    return min(iou_xywh(boxes[i], boxes[j])
               for i in range(len(boxes)) for j in range(i + 1, len(boxes)))


def norm1000_to_pixel_xywh(bbox, W, H):
    """Convert [0,1000] normalized xyxy to pixel xywh."""
    x1, y1, x2, y2 = bbox
    return [x1 / 1000.0 * W, y1 / 1000.0 * H,
            (x2 - x1) / 1000.0 * W, (y2 - y1) / 1000.0 * H]


def valid_box(box):
    try:
        return len(box) == 4 and all(math.isfinite(float(v)) for v in box) and float(box[2]) > 0 and float(box[3]) > 0
    except (TypeError, ValueError):
        return False


def append_distinct(pool, candidate, duplicate_iou=0.92):
    box = candidate.get("bbox")
    if not valid_box(box) or any(iou_xywh(box, c["bbox"]) >= duplicate_iou for c in pool):
        return False
    pool.append(candidate)
    return True
