"""Select the DINO routing threshold using TRAIN calibration only.

This is the portable CRVG counterpart of PaDT's
``audit_dino_route_calibration.py``. It evaluates the already-frozen risk
policy on nested consensus routes and serializes the selected threshold for
the threshold-selection figure. No evaluation split is read.
"""
import argparse
from pathlib import Path

import torch

from crvg.controller.apply import load_controller
from crvg.controller.features import FEATURE_NAMES
from crvg.controller.model import evaluate_policy, policy_is_safe, strip_decisions
from crvg.utils.data import read_jsonl, write_json


def parse_floats(value):
    return sorted({float(item.strip()) for item in value.split(",") if item.strip()})


def routed_sample_ids(rows, threshold):
    feature_index = FEATURE_NAMES.index("existing_pool_min_iou")
    sample_values = {}
    for row in rows:
        sample_values.setdefault(row["sample_id"], float(row["features"][feature_index]))
    return {
        sample_id
        for sample_id, min_pairwise_iou in sample_values.items()
        if min_pairwise_iou < threshold
    }


def run_threshold(rows, probabilities, threshold, policy, expected_domains,
                  min_domain_net=0, min_total_net=1):
    sample_ids = routed_sample_ids(rows, threshold)
    indices = [index for index, row in enumerate(rows) if row["sample_id"] in sample_ids]
    if not indices:
        return {
            "threshold": float(threshold), "n": 0, "safe": False,
            "net": 0, "rescue": 0, "damage": 0,
            "delta_acc50": 0.0, "delta_miou": 0.0, "domain": {},
        }
    subset = [rows[index] for index in indices]
    run = evaluate_policy(
        subset,
        probabilities[indices],
        gate=float(policy["gate"]),
        damage_cost=float(policy["damage_cost"]),
        abstain_cost=float(policy["abstain_cost"]),
        require_permutation_agree=bool(policy["require_permutation_agree"]),
    )
    run["threshold"] = float(threshold)
    run["safe"] = policy_is_safe(
        run,
        expected_domains,
        min_domain_net=min_domain_net,
        min_total_net=min_total_net,
    )
    return strip_decisions(run)


def selection_key(run):
    domain_nets = [item["net"] for item in run.get("domain", {}).values()]
    minimum_domain_net = min(domain_nets) if domain_nets else -(10 ** 9)
    return (
        int(run.get("safe", False)),
        int(run.get("net", 0)),
        minimum_domain_net,
        float(run.get("delta_miou", 0.0)),
        -int(run.get("damage", 0)),
        -float(run["threshold"]),
    )


@torch.inference_mode()
def calibrate(calibration, controller, thresholds, min_domain_net=0,
              min_total_net=1):
    rows = read_jsonl(calibration)
    if not rows:
        raise ValueError("TRAIN calibration data is empty")
    model, center, scale, config = load_controller(controller)
    features = torch.tensor([row["features"] for row in rows], dtype=torch.float32)
    probabilities = model((features - center) / scale).softmax(-1)
    policy = config["selected_policy"]
    constraints = config.get("selection_constraints", {})
    expected_domains = tuple(
        constraints.get("expected_domains", ("refcoco", "refcoco+", "refcocog"))
    )
    base_samples = len({row["sample_id"] for row in rows})
    deployment_route_fraction = float(constraints.get("deployment_route_fraction", 0.0))
    runs = []
    for threshold in thresholds:
        run = run_threshold(
            rows, probabilities, threshold, policy, expected_domains,
            min_domain_net=min_domain_net, min_total_net=min_total_net,
        )
        relative_route_fraction = run["n"] / max(base_samples, 1)
        run["relative_route_fraction"] = relative_route_fraction
        run["estimated_full_route_fraction"] = (
            deployment_route_fraction * relative_route_fraction
        )
        run["estimated_full_delta_acc50"] = (
            run["delta_acc50"] * run["estimated_full_route_fraction"]
        )
        runs.append(run)
    selected = max(runs, key=selection_key)
    return {
        "method": "TRAIN-only calibration of DINO disagreement routing",
        "calibration": str(Path(calibration).resolve()),
        "controller": str(Path(controller).resolve()),
        "threshold_semantics": "route iff existing_pool_min_iou < threshold",
        "base_calibration_samples": base_samples,
        "frozen_controller_policy": {
            key: policy[key]
            for key in ("gate", "damage_cost", "abstain_cost", "require_permutation_agree")
        },
        "runs": runs,
        "selected_threshold": selected["threshold"],
        "selected": selected,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--controller", required=True)
    parser.add_argument("--thresholds", type=parse_floats, default=parse_floats(".5,.75"))
    parser.add_argument("--min-domain-net", type=int, default=0)
    parser.add_argument("--min-total-net", type=int, default=1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = calibrate(
        args.calibration, args.controller, args.thresholds,
        min_domain_net=args.min_domain_net, min_total_net=args.min_total_net,
    )
    write_json(args.out, payload)
    print("=== TRAIN-only DINO route calibration ===")
    print("route rule: existing_pool_min_iou < threshold")
    for run in payload["runs"]:
        domain_net = {
            domain: item["net"] for domain, item in run.get("domain", {}).items()
        }
        print(
            f"threshold={run['threshold']:.2f} n={run['n']:4d} "
            f"switch={run.get('switches', 0):3d} rescue={run['rescue']:3d} "
            f"damage={run['damage']:3d} net={run['net']:+3d} "
            f"d50={run['delta_acc50']:+.4f} dIoU={run['delta_miou']:+.4f} "
            f"domain_net={domain_net} safe={run['safe']}"
        )
    selected = payload["selected"]
    print(
        f"TRAIN-CALIBRATED ROUTE: threshold={selected['threshold']:.2f} "
        f"net={selected['net']:+d} safe={selected['safe']}"
    )
    print(f"audit saved to: {args.out}")


if __name__ == "__main__":
    main()
