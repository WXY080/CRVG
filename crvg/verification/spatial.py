"""Deterministic target/anchor parsing and spatial evidence for CRVG."""

import math
import re

from crvg.utils.bbox import iou_xywh


RELATION_PATTERNS = (
    ("left_of", re.compile(r"\b(?:to\s+the\s+)?left\s+of\b", re.I)),
    ("right_of", re.compile(r"\b(?:to\s+the\s+)?right\s+of\b", re.I)),
    ("above", re.compile(r"\b(?:above|over)\b", re.I)),
    ("below", re.compile(r"\b(?:below|under|underneath)\b", re.I)),
    ("next_to", re.compile(r"\b(?:next\s+to|beside|adjacent\s+to)\b", re.I)),
    ("near", re.compile(r"\b(?:near|nearest|closest\s+to)\b", re.I)),
    ("far", re.compile(r"\b(?:far\s+from|farthest\s+from)\b", re.I)),
    ("between", re.compile(r"\bbetween\b", re.I)),
    ("inside", re.compile(r"\b(?:inside|within)\b", re.I)),
    ("outside", re.compile(r"\boutside\b", re.I)),
    ("near", re.compile(r"\baround\b", re.I)),
)


def clean_phrase(value):
    value = re.sub(r"^[\s,;:.\-]+|[\s,;:.\-]+$", "", value)
    return re.sub(r"\s+", " ", value).strip()


def parse_spatial_plan(expression):
    """Return a small executable spatial plan, or None when unsupported."""
    text = clean_phrase(expression.lower())
    for relation, pattern in RELATION_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        target = clean_phrase(text[: match.start()])
        anchor = clean_phrase(text[match.end() :])
        if relation == "between":
            anchor = anchor.replace(" and ", ". ", 1)
        if anchor:
            return {
                "relation": relation,
                "target_phrase": target,
                "anchor_phrase": anchor,
                "requires_anchor": True,
            }

    absolute_patterns = (
        ("absolute_upper_left", r"\b(?:upper|top)[\s-]+left\b|\bleft[\s-]+(?:upper|top)\b"),
        ("absolute_upper_right", r"\b(?:upper|top)[\s-]+right\b|\bright[\s-]+(?:upper|top)\b"),
        ("absolute_lower_left", r"\b(?:lower|bottom)[\s-]+left\b|\bleft[\s-]+(?:lower|bottom)\b"),
        ("absolute_lower_right", r"\b(?:lower|bottom)[\s-]+right\b|\bright[\s-]+(?:lower|bottom)\b"),
        ("absolute_left", r"\b(?:leftmost|on\s+the\s+left|left\s+side|left)\b"),
        ("absolute_right", r"\b(?:rightmost|on\s+the\s+right|right\s+side|right)\b"),
        ("absolute_top", r"\b(?:topmost|at\s+the\s+top|upper|top)\b"),
        ("absolute_bottom", r"\b(?:bottommost|at\s+the\s+bottom|lower|bottom)\b"),
        ("absolute_center", r"\b(?:center|centre|middle)\b"),
    )
    for relation, pattern in absolute_patterns:
        if re.search(pattern, text, re.I):
            return {
                "relation": relation,
                "target_phrase": text,
                "anchor_phrase": "",
                "requires_anchor": False,
            }
    return None


def center(box):
    x, y, width, height = [float(value) for value in box]
    return x + width / 2.0, y + height / 2.0


def containment(inner, outer):
    x, y, width, height = [float(value) for value in inner]
    ox, oy, outer_width, outer_height = [float(value) for value in outer]
    ix1, iy1 = max(x, ox), max(y, oy)
    ix2, iy2 = min(x + width, ox + outer_width), min(y + height, oy + outer_height)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return intersection / max(width * height, 1e-9)


def _usable_anchors(candidate, anchors):
    ordered = sorted(anchors, key=lambda item: float(item.get("score", 0.0)), reverse=True)
    distinct = [item for item in ordered if iou_xywh(candidate, item["bbox"]) < 0.6]
    return distinct or ordered


def spatial_score(candidate, anchors, plan, image_size):
    """Execute a spatial plan and return a score in roughly [0, 1]."""
    if not plan:
        return None
    image_width, image_height = [max(float(value), 1.0) for value in image_size]
    diagonal = math.hypot(image_width, image_height)
    candidate_x, candidate_y = center(candidate)
    relation = plan["relation"]

    corner_scores = {
        "absolute_upper_left": (1.0 - candidate_x / image_width, 1.0 - candidate_y / image_height),
        "absolute_upper_right": (candidate_x / image_width, 1.0 - candidate_y / image_height),
        "absolute_lower_left": (1.0 - candidate_x / image_width, candidate_y / image_height),
        "absolute_lower_right": (candidate_x / image_width, candidate_y / image_height),
    }
    if relation in corner_scores:
        horizontal, vertical = corner_scores[relation]
        return (horizontal + vertical) / 2.0

    if relation == "absolute_left":
        return 1.0 - candidate_x / image_width
    if relation == "absolute_right":
        return candidate_x / image_width
    if relation == "absolute_top":
        return 1.0 - candidate_y / image_height
    if relation == "absolute_bottom":
        return candidate_y / image_height
    if relation == "absolute_center":
        distance = math.hypot(candidate_x - image_width / 2.0, candidate_y - image_height / 2.0)
        return 1.0 - min(distance / (diagonal / 2.0), 1.0)

    anchors = _usable_anchors(candidate, anchors)
    if not anchors:
        return None
    if relation == "between":
        if len(anchors) < 2:
            return None
        first_x, first_y = center(anchors[0]["bbox"])
        second_x, second_y = center(anchors[1]["bbox"])
        midpoint_x = (first_x + second_x) / 2.0
        midpoint_y = (first_y + second_y) / 2.0
        distance = math.hypot(candidate_x - midpoint_x, candidate_y - midpoint_y)
        return 1.0 - min(distance / diagonal, 1.0)

    anchor = anchors[0]["bbox"]
    anchor_x, anchor_y = center(anchor)
    if relation == "left_of":
        return 0.5 + 0.5 * max(-1.0, min(1.0, (anchor_x - candidate_x) / image_width))
    if relation == "right_of":
        return 0.5 + 0.5 * max(-1.0, min(1.0, (candidate_x - anchor_x) / image_width))
    if relation == "above":
        return 0.5 + 0.5 * max(-1.0, min(1.0, (anchor_y - candidate_y) / image_height))
    if relation == "below":
        return 0.5 + 0.5 * max(-1.0, min(1.0, (candidate_y - anchor_y) / image_height))
    distance = math.hypot(candidate_x - anchor_x, candidate_y - anchor_y) / diagonal
    if relation in ("near", "next_to"):
        overlap_penalty = 0.25 * iou_xywh(candidate, anchor) if relation == "next_to" else 0.0
        return 1.0 - min(distance + overlap_penalty, 1.0)
    if relation == "far":
        return min(distance, 1.0)
    if relation == "inside":
        return containment(candidate, anchor)
    if relation == "outside":
        return 1.0 - containment(candidate, anchor)
    return None


def phrase_match_score(candidate, detections):
    return max(
        (
            float(item.get("score", 0.0)) * iou_xywh(candidate, item["bbox"])
            for item in detections
        ),
        default=0.0,
    )
