"""Strict, full-denominator comparisons for all reporting commands."""
from pathlib import Path

from crvg.utils.data import results, index_rows, row_key, current_bbox, current_iou
from crvg.utils.metrics import acc_miou


def aligned(before, after):
    left, right = index_rows(results(before)), index_rows(results(after))
    if left.keys() != right.keys():
        raise ValueError("Comparison has missing/extra sample IDs")
    pairs = [(row, right[key]) for key, row in left.items()]
    for a, b in pairs:
        if a.get("image") != b.get("image") or a.get("text") != b.get("text"):
            raise ValueError(f"Sample identity mismatch: {row_key(a)}")
    return pairs


def compare(before, after):
    pairs = aligned(before, after)
    if not pairs:
        raise ValueError("Cannot evaluate an empty split")
    ai, bi = [current_iou(a) for a, _ in pairs], [current_iou(b) for _, b in pairs]
    a_metrics, b_metrics = acc_miou(ai), acc_miou(bi)
    rescue = sum(a < .5 <= b for a, b in zip(ai, bi))
    damage = sum(b < .5 <= a for a, b in zip(ai, bi))
    return {"n": len(pairs), "before": a_metrics, "after": b_metrics,
            "delta_pp": {k: 100*(b_metrics[k]-a_metrics[k]) for k in a_metrics},
            "switches": sum(current_bbox(a) != current_bbox(b) for a, b in pairs),
            "rescue": rescue, "damage": damage, "net": rescue-damage,
            "flip_precision": rescue/(rescue+damage) if rescue+damage else None}


def average(reports):
    if not reports:
        raise ValueError("No reports to average")
    return {key: sum(r["delta_pp"][key] for r in reports)/len(reports)
            for key in reports[0]["delta_pp"]}


def paths(log_dir, dataset):
    root = Path(log_dir)
    return {key: root / template.format(dataset=dataset) for key, template in {
        "base": "rec_results_{dataset}.json",
        "expanded": "rec_results_{dataset}_expanded.json",
        "qwen": "rec_results_{dataset}_qwen.json",
        "final": "rec_results_{dataset}_crvg.json",
        "crops": "crop_picks_{dataset}.json",
        "dino": "dino_evidence_{dataset}.json",
        "pairs": "pairwise_picks_{dataset}.json",
        "risk": "risk_decisions_{dataset}.json",
    }.items()}
