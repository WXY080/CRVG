"""Section 3.4 risk controller and expected-utility selection."""
from collections import defaultdict

import torch
from torch import nn
from crvg.controller.features import LABELS, LABEL_TO_INDEX, finite_float
from crvg.utils.metrics import acc_miou

class DinoRiskMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.10):
        super().__init__()
        middle = max(hidden_dim // 2, 16)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, middle),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(middle, len(LABELS)),
        )

    def forward(self, features):
        return self.network(features)


def policy_score(probabilities, damage_cost=2.0, abstain_cost=0.25):
    keep = probabilities[..., LABEL_TO_INDEX["keep"]]
    switch = probabilities[..., LABEL_TO_INDEX["switch"]]
    abstain = probabilities[..., LABEL_TO_INDEX["abstain"]]
    return switch - damage_cost * keep - abstain_cost * abstain


def evaluate_policy(
    examples,
    probabilities,
    gate,
    damage_cost=2.0,
    abstain_cost=0.25,
    require_permutation_agree=True,
):
    if len(examples) != len(probabilities):
        raise ValueError("example/probability length mismatch")
    grouped = defaultdict(list)
    for index, example in enumerate(examples):
        grouped[example["sample_id"]].append((index, example))

    current_ious = []
    selected_ious = []
    domain_ious = defaultdict(lambda: [[], []])
    decisions = []
    rescue = damage = switches = 0
    for sample_id, candidates in grouped.items():
        scored = []
        for index, example in candidates:
            probability = probabilities[index]
            score = finite_float(
                policy_score(probability, damage_cost, abstain_cost).item()
                if torch.is_tensor(probability)
                else (
                    probability[LABEL_TO_INDEX["switch"]]
                    - damage_cost * probability[LABEL_TO_INDEX["keep"]]
                    - abstain_cost * probability[LABEL_TO_INDEX["abstain"]]
                )
            )
            scored.append((score, finite_float(probability[LABEL_TO_INDEX["switch"]]), index, example))
        score, switch_probability, selected_index, selected_example = max(
            scored, key=lambda item: (item[0], item[1], item[3]["dino_phrase_score"])
        )
        do_switch = score >= gate
        if require_permutation_agree and not selected_example["permutation_agree"]:
            do_switch = False
        class_probabilities = {label: float(probabilities[selected_index][i]) for i, label in enumerate(LABELS)}
        predicted_class = max(class_probabilities, key=class_probabilities.get)
        action = "SWITCH" if do_switch else ("KEEP" if predicted_class == "keep" else "ABSTAIN")
        current_iou = selected_example["current_iou"]
        selected_iou = selected_example["alternative_iou"] if do_switch else current_iou
        current_ious.append(current_iou)
        selected_ious.append(selected_iou)
        domain_pair = domain_ious[selected_example["domain"]]
        domain_pair[0].append(current_iou)
        domain_pair[1].append(selected_iou)
        switches += int(do_switch)
        rescue += int(current_iou < 0.5 <= selected_iou)
        damage += int(selected_iou < 0.5 <= current_iou)
        decisions.append(
            {
                "sample_id": sample_id,
                "domain": selected_example["domain"],
                "dataset_index": selected_example.get("dataset_index"),
                "image_path": selected_example.get("image_path"),
                "record_index": selected_example.get("record_index"),
                "switched": do_switch,
                "action": action,
                "class_probabilities": class_probabilities,
                "risk_score": score,
                "switch_probability": switch_probability,
                "current_iou": current_iou,
                "selected_iou": selected_iou,
                "selected": {
                    "bbox": selected_example["alternative_bbox"],
                    "iou": selected_example["alternative_iou"],
                    "source": "grounding_dino_phrase",
                }
                if do_switch
                else None,
            }
        )

    current_metrics = acc_miou(current_ious)
    selected_metrics = acc_miou(selected_ious)
    domain_metrics = {}
    for domain, (domain_current, domain_selected) in sorted(domain_ious.items()):
        before = acc_miou(domain_current)
        after = acc_miou(domain_selected)
        domain_rescue = sum(a < 0.5 <= b for a, b in zip(domain_current, domain_selected))
        domain_damage = sum(b < 0.5 <= a for a, b in zip(domain_current, domain_selected))
        domain_metrics[domain] = {
            "n": len(domain_current),
            "current_metrics": before,
            "selected_metrics": after,
            "delta_acc50": after["acc0.5"] - before["acc0.5"],
            "delta_miou": after["miou"] - before["miou"],
            "rescue": domain_rescue,
            "damage": domain_damage,
            "net": domain_rescue - domain_damage,
        }
    return {
        "gate": gate,
        "damage_cost": damage_cost,
        "abstain_cost": abstain_cost,
        "require_permutation_agree": require_permutation_agree,
        "n": len(current_ious),
        "current_metrics": current_metrics,
        "selected_metrics": selected_metrics,
        "delta_acc50": selected_metrics["acc0.5"] - current_metrics["acc0.5"],
        "delta_miou": selected_metrics["miou"] - current_metrics["miou"],
        "switches": switches,
        "rescue": rescue,
        "damage": damage,
        "net": rescue - damage,
        "domain": domain_metrics,
        "decisions": decisions,
    }


def policy_is_safe(run, expected_domains, min_domain_net=0, min_total_net=1):
    if run["net"] < min_total_net or run["delta_miou"] < 0.0:
        return False
    for domain in expected_domains:
        if domain not in run["domain"] or run["domain"][domain]["net"] < min_domain_net:
            return False
    return True


def policy_sort_key(run, expected_domains, min_domain_net=0, min_total_net=1):
    safe = policy_is_safe(run, expected_domains, min_domain_net, min_total_net)
    domain_deltas = [run["domain"].get(domain, {}).get("delta_acc50", -1.0) for domain in expected_domains]
    return (
        int(safe),
        run["delta_acc50"],
        min(domain_deltas) if domain_deltas else -1.0,
        run["net"],
        run["delta_miou"],
        -run["damage"],
        -run["switches"],
    )


def select_policy(
    examples,
    probabilities,
    gates,
    damage_costs,
    abstain_costs,
    expected_domains,
    min_domain_net=0,
    min_total_net=1,
):
    runs = []
    for damage_cost in damage_costs:
        for abstain_cost in abstain_costs:
            for gate in gates:
                run = evaluate_policy(examples, probabilities, gate, damage_cost, abstain_cost,
                                      require_permutation_agree=True)
                run["safe"] = policy_is_safe(run, expected_domains, min_domain_net, min_total_net)
                runs.append(run)
    selected = max(
        runs,
        key=lambda run: policy_sort_key(
            run, expected_domains, min_domain_net, min_total_net
        ),
    )
    return selected, runs


def strip_decisions(run):
    return {key: value for key, value in run.items() if key != "decisions"}
