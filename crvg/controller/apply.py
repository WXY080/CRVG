"""Apply a frozen controller; missing routed evidence is never silently skipped."""
import argparse
from pathlib import Path

import torch

from crvg.controller.features import FEATURE_NAMES, build_risk_examples
from crvg.controller.model import DinoRiskMLP, evaluate_policy
from crvg.utils.data import write_json, results, index_rows, row_key, current_bbox, set_prediction, check_source
from crvg.utils.data import fingerprint, read_json
from crvg.verification.dino_detector import dino_route


def load_controller(directory):
    directory = Path(directory)
    config = read_json(directory / "risk_controller_config.json")
    if config.get("scoring_backend") != "base_qwen_next_token":
        raise ValueError("Controller must be trained on the frozen-base scoring backend used here")
    if config.get("selected_policy", {}).get("require_permutation_agree") is not True:
        raise ValueError("Paper Section 3.4 requires an order-consistent frozen controller policy")
    if tuple(config.get("feature_names", [])) != FEATURE_NAMES:
        raise ValueError("Controller requires the manuscript's 36-feature schema")
    cfg = config["model"]
    if cfg["input_dim"] != 36 or cfg["hidden_dim"] != 64:
        raise ValueError("Controller architecture must be 36-64-32-3")
    model = DinoRiskMLP(cfg["input_dim"], cfg["hidden_dim"], cfg["dropout"])
    model.load_state_dict(torch.load(directory / "risk_controller.pt", map_location="cpu", weights_only=True))
    model.eval()
    model.requires_grad_(False)
    mean = torch.tensor(config["normalization"]["mean"])
    scale = torch.tensor(config["normalization"]["scale"])
    if len(mean) != len(FEATURE_NAMES) or len(scale) != len(FEATURE_NAMES) or not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Invalid controller normalization")
    return model, mean, scale, config


@torch.inference_mode()
def score_evidence(evidence, picks, directory, domain="auto"):
    check_source(picks, evidence)
    if picks.get("meta", {}).get("scoring_backend") != "base_qwen_next_token":
        raise ValueError("Unsupported pairwise scoring backend")
    model, mean, scale, config = load_controller(directory)
    examples, stats = build_risk_examples(evidence, picks, "inference", explicit_domain=domain)
    if not examples:
        return [], config, stats
    features = torch.tensor([e["features"] for e in examples], dtype=torch.float32)
    probabilities = model((features-mean)/scale).softmax(-1)
    policy = config["selected_policy"]
    kwargs = {k: policy[k] for k in ("gate", "damage_cost", "abstain_cost", "require_permutation_agree")}
    # Unsafe learned policies are artifacts for inspection, not deployable selectors.
    if not policy.get("safe", config.get("passed", False)):
        kwargs["gate"] = 10.
    decisions = evaluate_policy(examples, probabilities, **kwargs)["decisions"]
    return decisions, config, stats


def merge_decisions(source, evidence, decisions, gamma1=.35):
    source_rows = results(source)
    available = index_rows(results(evidence))
    mapped = index_rows(decisions)
    output = []
    for row in source_rows:
        active = dino_route(row, gamma1)
        updated = dict(row)
        action = "NOT_INVOKED"
        if active:
            cached = available.get(row_key(row))
            if cached is None:
                raise ValueError(f"DINO evidence missing for sample {row_key(row)}")
            if current_bbox(cached) != current_bbox(row):
                raise ValueError(f"Current box changed for {row_key(row)}; rerun pairwise scoring")
            if cached.get("text") != row.get("text") or cached.get("image") != row.get("image"):
                raise ValueError(f"Sample identity mismatch for {row_key(row)}")
            decision = mapped.get(row_key(row))
            action = decision.get("action", "SWITCH" if decision["switched"] else "ABSTAIN") if decision else "ABSTAIN"
            if decision and decision["switched"]:
                updated = set_prediction(row, decision["selected"]["bbox"], "dino_risk_controller")
        updated["crvg"] = {**row.get("crvg", {}), "dino_routed": active,
                           "risk_action": action, "gamma1": gamma1}
        output.append(updated)
    return {"meta": {**source.get("meta", {}), "gamma1": gamma1, "complete": True}, "results": output}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence")
    parser.add_argument("--picks", required=True)
    parser.add_argument("--controller", required=True)
    parser.add_argument("--domain", default="auto")
    parser.add_argument("--source-json", required=True)
    parser.add_argument("--selected-out", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gamma1", type=float, default=.35)
    args = parser.parse_args()
    source, evidence, picks = map(read_json, (args.source_json, args.evidence, args.picks))
    check_source(evidence, source)
    decisions, config, stats = score_evidence(evidence, picks, args.controller, args.domain)
    output = merge_decisions(source, evidence, decisions, args.gamma1)
    write_json(args.selected_out, output)
    write_json(args.out, {"decisions": decisions, "stats": stats,
                         "source_sha256": fingerprint(source), "evidence_sha256": fingerprint(evidence),
                         "selected_policy": config["selected_policy"],
                         "controller": args.controller, "gamma1": args.gamma1})


if __name__ == "__main__":
    main()
