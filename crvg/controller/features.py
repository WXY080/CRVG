#!/usr/bin/env python3
"""Shared features and policy utilities for DINO KEEP/SWITCH/ABSTAIN routing."""

import hashlib
import json
import math
import re
from collections import defaultdict

from crvg.utils.bbox import iou_xywh
from crvg.utils.data import current_candidate, extract_expression
from crvg.utils.data import index_rows, row_key
from crvg.settings import DINO_EVIDENCE_SCHEMA


LABELS = ("keep", "switch", "abstain")
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}

FEATURE_NAMES = (
    "qwen_current_probability",
    "qwen_alternative_probability",
    "qwen_alternative_advantage",
    "qwen_permutation_agree",
    "qwen_permutation_logit_gap",
    "qwen_above_frozen_g06",
    "dino_phrase_score",
    "spatial_skill_score",
    "current_alternative_iou",
    "current_area_fraction",
    "alternative_area_fraction",
    "log_area_ratio",
    "current_center_x",
    "current_center_y",
    "alternative_center_x",
    "alternative_center_y",
    "center_delta_x",
    "center_delta_y",
    "center_distance",
    "challenger_count_log",
    "existing_candidate_count_log",
    "qwen_rank",
    "dino_rank",
    "qwen_competitor_margin",
    "dino_competitor_margin",
    "existing_pool_min_iou",
    "expression_token_count_log",
    "cue_spatial",
    "cue_ordinal",
    "cue_attribute",
    "cue_color",
    "requires_anchor",
    "anchor_confidence",
    "domain_refcoco",
    "domain_refcoco_plus",
    "domain_refcocog",
)


SPATIAL_RE = re.compile(
    r"\b(left|right|above|below|under|over|behind|front|between|beside|next|near|"
    r"corner|center|middle|top|bottom|back|foreground|background)\b"
)
ORDINAL_RE = re.compile(
    r"\b(first|second|third|fourth|fifth|last|leftmost|rightmost|topmost|bottommost|"
    r"\d+(?:st|nd|rd|th))\b"
)
ATTRIBUTE_RE = re.compile(
    r"\b(wearing|holding|with|without|striped|spotted|small|smaller|smallest|large|"
    r"larger|largest|tall|short|young|old|open|closed|standing|sitting|lying)\b"
)
COLOR_RE = re.compile(
    r"\b(red|orange|yellow|green|blue|purple|pink|brown|black|white|gray|grey|gold|silver)\b"
)


def finite_float(value, default=0.0, lower=None, upper=None):
    try:
        output = float(value)
    except (TypeError, ValueError):
        output = float(default)
    if not math.isfinite(output):
        output = float(default)
    if lower is not None:
        output = max(float(lower), output)
    if upper is not None:
        output = min(float(upper), output)
    return output


def canonical_domain(value):
    text = str(value or "").lower().replace("\\", "/")
    if "refcoco+" in text or "refcocop" in text or "refcoco_plus" in text:
        return "refcoco+"
    if "refcocog" in text:
        return "refcocog"
    if "refcoco" in text:
        return "refcoco"
    return None


def infer_domain(row, evidence_meta=None, explicit_domain=None):
    if explicit_domain and explicit_domain != "auto":
        domain = canonical_domain(explicit_domain)
        if not domain:
            raise ValueError(f"Unknown explicit domain: {explicit_domain}")
        return domain
    candidates = (
        row.get("training_source"),
        row.get("dataset"),
        row.get("dataset_name"),
        (evidence_meta or {}).get("training_source"),
        (evidence_meta or {}).get("dataset"),
        (evidence_meta or {}).get("source"),
    )
    for candidate in candidates:
        domain = canonical_domain(candidate)
        if domain:
            return domain
    return None


def is_train_row(row, evidence_meta=None):
    markers = (
        row.get("training_source"),
        row.get("split"),
        row.get("dataset"),
        (evidence_meta or {}).get("training_source"),
        (evidence_meta or {}).get("split"),
        (evidence_meta or {}).get("dataset"),
        (evidence_meta or {}).get("source"),
    )
    text = " ".join(str(value or "").lower() for value in markers)
    if re.search(r"(?:^|[\W_])(val|validation|test[ab]?)(?:$|[\W_])", text):
        return False
    return bool(re.search(r"(?:^|[\W_])train(?:$|[\W_])", text))


def image_group(row):
    path = str(row.get("image_path") or row.get("image") or "")
    if path:
        return path.replace("\\", "/").rsplit("/", 1)[-1]
    return f"dataset-index:{row.get('dataset_index')}"


def stable_fraction(value, seed):
    digest = hashlib.sha1(f"{seed}:{value}".encode("utf-8")).hexdigest()[:12]
    return int(digest, 16) / float(16**12)


def sample_identifier(domain, row):
    payload = [
        domain,
        image_group(row),
        row.get("dataset_index"),
        extract_expression(row),
    ]
    return hashlib.sha1(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def decision_label(current_iou, alternative_iou, threshold=0.5, refine_gap=0.15):
    current_correct = current_iou >= threshold
    alternative_correct = alternative_iou >= threshold
    if not current_correct and alternative_correct:
        return "switch"
    if current_correct and not alternative_correct:
        return "keep"
    if current_correct and alternative_correct:
        delta = alternative_iou - current_iou
        if delta >= refine_gap:
            return "switch"
        if delta <= -refine_gap:
            return "keep"
    return "abstain"


def _image_size(row, boxes):
    value = row.get("image_size")
    width = height = None
    if isinstance(value, dict):
        width = value.get("width") or value.get("w")
        height = value.get("height") or value.get("h")
    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        width, height = value[:2]
    width = finite_float(width, 0.0)
    height = finite_float(height, 0.0)
    if width <= 0 or height <= 0:
        width = max((finite_float(box[0]) + finite_float(box[2]) for box in boxes), default=1.0)
        height = max((finite_float(box[1]) + finite_float(box[3]) for box in boxes), default=1.0)
    return max(width, 1.0), max(height, 1.0)


def _candidate_extra(row, bbox):
    best = None
    best_iou = -1.0
    for candidate in row.get("candidates", []):
        candidate_bbox = candidate.get("bbox")
        if not isinstance(candidate_bbox, (list, tuple)) or len(candidate_bbox) != 4:
            continue
        overlap = iou_xywh(bbox, candidate_bbox)
        if overlap > best_iou:
            best_iou = overlap
            best = candidate
    return best or {}


def _min_existing_iou(row):
    state = row.get("crvg", {})
    if "b1_min_iou" in state and "b1_count" in state:
        return float(state["b1_min_iou"]), int(state["b1_count"])
    boxes = [
        candidate.get("bbox")
        for candidate in row.get("candidates", [])
        if candidate.get("source") != "grounding_dino_phrase"
        and isinstance(candidate.get("bbox"), (list, tuple))
        and len(candidate["bbox"]) == 4
    ]
    if len(boxes) < 2:
        return 1.0, len(boxes)
    return min(
        iou_xywh(boxes[left], boxes[right])
        for left in range(len(boxes))
        for right in range(left + 1, len(boxes))
    ), len(boxes)


def _rank(values, index):
    if len(values) <= 1:
        return 1.0
    order = sorted(range(len(values)), key=lambda item: values[item], reverse=True)
    position = order.index(index)
    return 1.0 - position / float(len(values) - 1)


def _competitor_margin(values, index):
    others = [value for item, value in enumerate(values) if item != index]
    return values[index] - max(others) if others else values[index]


def _expression_features(expression):
    text = str(expression or "").lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return (
        math.log1p(len(tokens)),
        float(bool(SPATIAL_RE.search(text))),
        float(bool(ORDINAL_RE.search(text))),
        float(bool(ATTRIBUTE_RE.search(text))),
        float(bool(COLOR_RE.search(text))),
    )


def extract_feature_vector(row, record, challenger_index, domain):
    current = current_candidate(row)
    challenger = record["challengers"][challenger_index]
    current_box = current["bbox"]
    alternative_box = challenger["bbox"]
    all_boxes = [
        candidate.get("bbox")
        for candidate in row.get("candidates", [])
        if isinstance(candidate.get("bbox"), (list, tuple)) and len(candidate["bbox"]) == 4
    ]
    width, height = _image_size(row, all_boxes + [current_box, alternative_box])
    image_area = width * height

    cx = (finite_float(current_box[0]) + 0.5 * finite_float(current_box[2])) / width
    cy = (finite_float(current_box[1]) + 0.5 * finite_float(current_box[3])) / height
    ax = (finite_float(alternative_box[0]) + 0.5 * finite_float(alternative_box[2])) / width
    ay = (finite_float(alternative_box[1]) + 0.5 * finite_float(alternative_box[3])) / height
    current_area = finite_float(current_box[2]) * finite_float(current_box[3]) / image_area
    alternative_area = finite_float(alternative_box[2]) * finite_float(alternative_box[3]) / image_area

    challengers = record.get("challengers", [])
    qwen_values = [finite_float(item.get("alternative_advantage")) for item in challengers]
    dino_values = [
        finite_float(item.get("dino_phrase_score", item.get("score", 0.0)))
        for item in challengers
    ]
    extra = _candidate_extra(row, alternative_box)
    existing_min_iou, existing_count = _min_existing_iou(row)
    expression_features = _expression_features(extract_expression(row))
    plan = row.get("relation_plan") or {}
    requires_anchor = float(bool(plan.get("requires_anchor")))
    anchor_confidence = finite_float(row.get("anchor_confidence"), 1.0)
    if not requires_anchor:
        anchor_confidence = 1.0

    domain_values = (
        float(domain == "refcoco"),
        float(domain == "refcoco+"),
        float(domain == "refcocog"),
    )
    qwen_advantage = finite_float(challenger.get("alternative_advantage"), -1.0)
    features = (
        finite_float(challenger.get("current_probability")),
        finite_float(challenger.get("alternative_probability")),
        qwen_advantage,
        float(bool(challenger.get("permutation_agree", False))),
        math.log1p(finite_float(challenger.get("permutation_logit_gap"), 0.0, lower=0.0)),
        float(qwen_advantage > 0.6 and bool(challenger.get("permutation_agree", False))),
        dino_values[challenger_index],
        finite_float(extra.get("spatial_skill_score")),
        finite_float(iou_xywh(current_box, alternative_box)),
        finite_float(current_area, lower=0.0, upper=4.0),
        finite_float(alternative_area, lower=0.0, upper=4.0),
        math.log(max(alternative_area, 1e-6) / max(current_area, 1e-6)),
        cx,
        cy,
        ax,
        ay,
        ax - cx,
        ay - cy,
        math.hypot(ax - cx, ay - cy),
        math.log1p(len(challengers)),
        math.log1p(existing_count),
        _rank(qwen_values, challenger_index),
        _rank(dino_values, challenger_index),
        _competitor_margin(qwen_values, challenger_index),
        _competitor_margin(dino_values, challenger_index),
        finite_float(existing_min_iou, lower=0.0, upper=1.0),
        *expression_features,
        requires_anchor,
        anchor_confidence,
        *domain_values,
    )
    if len(features) != len(FEATURE_NAMES):
        raise AssertionError(f"feature length mismatch: {len(features)} != {len(FEATURE_NAMES)}")
    return [finite_float(value) for value in features]


def build_risk_examples(
    evidence,
    picks,
    source_name,
    explicit_domain="auto",
    require_train=False,
    threshold=0.5,
    refine_gap=0.15,
):
    if evidence.get("meta", {}).get("evidence_schema") != DINO_EVIDENCE_SCHEMA:
        raise ValueError("DINO evidence schema mismatch; regenerate evidence with this pipeline")
    rows = evidence.get("results", [])
    records = index_rows(picks.get("picks", []))
    if set(index_rows(rows)) != set(records):
        raise ValueError(f"picks/evidence identity mismatch for {source_name}")
    examples = []
    stats = defaultdict(int)
    meta = evidence.get("meta", {})
    for row_index, row in enumerate(rows):
        domain = infer_domain(row, meta, explicit_domain)
        if not domain:
            raise ValueError("Unknown domain; set dataset metadata or --domain explicitly")
        if require_train and not is_train_row(row, meta):
            raise ValueError("Only TRAIN-domain evidence may create controller training data")
        current = current_candidate(row)
        record = records[row_key(row)]
        if current is None or not record.get("challengers"):
            stats["no_scored_challenger"] += 1
            continue
        sample_id = sample_identifier(domain, row)
        current_iou = finite_float(current.get("iou"))
        if require_train and current.get("iou") is None:
            raise ValueError("Training requires ground-truth IoUs")
        for challenger_index, challenger in enumerate(record["challengers"]):
            if not isinstance(challenger.get("bbox"), (list, tuple)):
                stats["invalid_challenger"] += 1
                continue
            alternative_iou = finite_float(challenger.get("iou"))
            if require_train and challenger.get("iou") is None:
                raise ValueError("Training requires challenger ground-truth IoUs")
            label = decision_label(current_iou, alternative_iou, threshold, refine_gap)
            examples.append(
                {
                    "sample_id": sample_id,
                    "image_group": image_group(row),
                    "domain": domain,
                    "training_source": row.get("training_source"),
                    "source_name": source_name,
                    "source_evidence": meta.get("source"),
                    "record_index": row_index,
                    "dataset_index": row.get("dataset_index"),
                    "image_path": row.get("image_path") or row.get("image"),
                    "expression": extract_expression(row),
                    "current_bbox": current["bbox"],
                    "alternative_bbox": challenger["bbox"],
                    "current_iou": current_iou,
                    "alternative_iou": alternative_iou,
                    "label": label,
                    "label_index": LABEL_TO_INDEX[label],
                    "permutation_agree": bool(challenger.get("permutation_agree", False)),
                    "qwen_alternative_advantage": finite_float(
                        challenger.get("alternative_advantage"), -1.0
                    ),
                    "dino_phrase_score": finite_float(challenger.get("dino_phrase_score")),
                    "features": extract_feature_vector(row, record, challenger_index, domain),
                    "feature_names": FEATURE_NAMES,
                    "scoring_backend": picks.get("meta", {}).get("scoring_backend"),
                }
            )
            stats[f"label_{label}"] += 1
            stats[f"domain_{domain}"] += 1
        stats["rows_scored"] += 1
    stats["examples"] = len(examples)
    return examples, dict(stats)
