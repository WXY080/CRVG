#!/usr/bin/env python3
"""Train a cost-sensitive three-class controller for accepting DINO proposals."""

import argparse
import copy
import json
import math
import os
import random
from collections import Counter

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from crvg.controller.model import (
    LABELS,
    LABEL_TO_INDEX,
    DinoRiskMLP,
    policy_sort_key,
    select_policy,
    strip_decisions,
)


from crvg.controller.features import FEATURE_NAMES
from crvg.utils.data import read_jsonl, write_json


def csv_floats(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-weight", type=float, default=2.5)
    parser.add_argument("--switch-weight", type=float, default=1.0)
    parser.add_argument("--abstain-weight", type=float, default=1.5)
    parser.add_argument("--false-switch-penalty", type=float, default=2.0)
    parser.add_argument("--missed-switch-penalty", type=float, default=0.5)
    parser.add_argument("--abstain-switch-penalty", type=float, default=1.0)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument(
        "--gates",
        default="-0.5,-0.3,-0.2,-0.1,0,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.75,1.01,10",
    )
    parser.add_argument("--damage-costs", default="1.5,2,3")
    parser.add_argument("--abstain-costs", default="0,0.25,0.5")
    parser.add_argument("--min-domain-net", type=int, default=0)
    parser.add_argument("--min-total-net", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda requested but CUDA is unavailable")
    return torch.device(value)


def tensor_data(rows, mean=None, scale=None):
    features = torch.tensor([row["features"] for row in rows], dtype=torch.float32)
    labels = torch.tensor([int(row["label_index"]) for row in rows], dtype=torch.long)
    if mean is None:
        mean = features.mean(dim=0)
        scale = features.std(dim=0, unbiased=False).clamp_min(1e-4)
    normalized = (features - mean) / scale
    return normalized, labels, mean, scale


def sampler_weights(rows):
    counts = Counter((row["domain"], row["label"]) for row in rows)
    weights = [1.0 / counts[(row["domain"], row["label"])] for row in rows]
    mean = sum(weights) / max(len(weights), 1)
    return torch.tensor([value / mean for value in weights], dtype=torch.double)


def classification_metrics(labels, probabilities):
    predictions = probabilities.argmax(dim=-1)
    output = {"accuracy": float((predictions == labels).float().mean())}
    for label, index in LABEL_TO_INDEX.items():
        mask = labels == index
        output[label] = {
            "n": int(mask.sum()),
            "accuracy": float((predictions[mask] == labels[mask]).float().mean())
            if mask.any()
            else 0.0,
            "mean_predicted_probability": float(probabilities[mask, index].mean())
            if mask.any()
            else 0.0,
        }
    return output


def training_loss(logits, labels, class_weights, args):
    probabilities = torch.softmax(logits, dim=-1)
    loss = F.cross_entropy(logits, labels, weight=class_weights)
    switch_probability = probabilities[:, LABEL_TO_INDEX["switch"]]
    keep_mask = labels == LABEL_TO_INDEX["keep"]
    switch_mask = labels == LABEL_TO_INDEX["switch"]
    abstain_mask = labels == LABEL_TO_INDEX["abstain"]
    penalties = []
    if keep_mask.any():
        penalties.append(args.false_switch_penalty * switch_probability[keep_mask].mean())
    if switch_mask.any():
        penalties.append(args.missed_switch_penalty * (1.0 - switch_probability[switch_mask]).mean())
    if abstain_mask.any():
        penalties.append(args.abstain_switch_penalty * switch_probability[abstain_mask].mean())
    if penalties:
        loss = loss + sum(penalties)
    return loss


@torch.no_grad()
def predict(model, features, device, batch_size=1024):
    outputs = []
    model.eval()
    for start in range(0, len(features), batch_size):
        logits = model(features[start : start + batch_size].to(device))
        outputs.append(torch.softmax(logits.float(), dim=-1).cpu())
    return torch.cat(outputs, dim=0) if outputs else torch.empty((0, len(LABELS)))


def validate_rows(rows, name, expected_feature_names=None):
    if not rows:
        raise ValueError(f"{name} data is empty")
    feature_names = tuple(rows[0].get("feature_names", ()))
    if feature_names != FEATURE_NAMES:
        raise ValueError(f"{name} feature schema is empty or contains duplicates")
    if expected_feature_names is not None and feature_names != tuple(expected_feature_names):
        raise ValueError(f"{name} feature schema differs from training data")
    for index, row in enumerate(rows):
        if row.get("scoring_backend") != "base_qwen_next_token":
            raise ValueError(f"{name} needs frozen-base Qwen scores at row {index}")
        if tuple(row.get("feature_names", ())) != feature_names:
            raise ValueError(f"{name} feature schema mismatch at row {index}")
        if row.get("label") not in LABEL_TO_INDEX:
            raise ValueError(f"{name} unknown label at row {index}: {row.get('label')}")
        if len(row.get("features", ())) != len(feature_names):
            raise ValueError(f"{name} feature length mismatch at row {index}")
        if not all(math.isfinite(v) for v in row["features"]):
            raise ValueError(f"{name} nonfinite feature at row {index}")
        if row.get("label_index") != LABEL_TO_INDEX[row["label"]]:
            raise ValueError(f"{name} inconsistent label index at row {index}")
    return feature_names


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    train_rows = read_jsonl(args.train)
    val_rows = read_jsonl(args.val)
    overlap = {row["image_group"] for row in train_rows} & {row["image_group"] for row in val_rows}
    if overlap:
        raise ValueError(f"TRAIN/calibration image overlap: {len(overlap)}")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    feature_names = validate_rows(train_rows, "train")
    validate_rows(val_rows, "validation", feature_names)
    expected_domains = ("refcoco", "refcoco+", "refcocog")
    found_domains = {row["domain"] for row in train_rows} & {row["domain"] for row in val_rows}
    if found_domains != set(expected_domains):
        raise ValueError(f"three-domain coverage required, found {sorted(found_domains)}")

    train_features, train_labels, mean, scale = tensor_data(train_rows)
    val_features, val_labels, _, _ = tensor_data(val_rows, mean, scale)
    sample_weights = sampler_weights(train_rows)
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(train_rows),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=args.batch_size,
        sampler=sampler,
        drop_last=False,
    )

    model = DinoRiskMLP(len(feature_names), 64, args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    class_weights = torch.tensor(
        [args.keep_weight, args.switch_weight, args.abstain_weight],
        dtype=torch.float32,
        device=device,
    )
    gates = csv_floats(args.gates)
    damage_costs = csv_floats(args.damage_costs)
    abstain_costs = csv_floats(args.abstain_costs)

    print("=== Training three-domain DINO risk controller ===")
    print(f"train / calibration: {len(train_rows)} / {len(val_rows)}")
    print(f"device: {device}; parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"feature dimension: {len(feature_names)}")
    print(f"train labels: {dict(Counter(row['label'] for row in train_rows))}")
    print(f"calibration labels: {dict(Counter(row['label'] for row in val_rows))}")

    best_state = None
    best_run = None
    best_epoch = 0
    best_key = None
    best_classification = None
    stale_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = training_loss(logits, labels, class_weights, args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        val_probabilities = predict(model, val_features, device)
        selected, runs = select_policy(
            val_rows,
            val_probabilities,
            gates,
            damage_costs,
            abstain_costs,
            expected_domains,
            args.min_domain_net,
            args.min_total_net,
        )
        key = policy_sort_key(
            selected,
            expected_domains,
            args.min_domain_net,
            args.min_total_net,
        )
        classification = classification_metrics(val_labels, val_probabilities)
        history.append(
            {
                "epoch": epoch,
                "loss": sum(losses) / max(len(losses), 1),
                "classification": classification,
                "selected_policy": strip_decisions(selected),
            }
        )
        if best_key is None or key > best_key:
            best_key = key
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            best_run = copy.deepcopy(selected)
            best_epoch = epoch
            best_classification = classification
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            domain_net = {domain: selected["domain"][domain]["net"] for domain in expected_domains}
            print(
                f"epoch={epoch:03d} loss={sum(losses) / max(len(losses), 1):.4f} "
                f"class_acc={classification['accuracy']:.3f} gate={selected['gate']:.2f} "
                f"cost={selected['damage_cost']:.2f}/{selected['abstain_cost']:.2f} "
                f"switch={selected['switches']} rescue={selected['rescue']} "
                f"damage={selected['damage']} net={selected['net']:+d} "
                f"d50={selected['delta_acc50']:+.4f} domain_net={domain_net} safe={selected['safe']}"
            )
        if stale_epochs >= args.patience:
            print(f"early stopping at epoch {epoch}; best epoch={best_epoch}")
            break

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    best_probabilities = predict(model, val_features, device)
    best_run, all_runs = select_policy(
        val_rows,
        best_probabilities,
        gates,
        damage_costs,
        abstain_costs,
        expected_domains,
        args.min_domain_net,
        args.min_total_net,
    )
    passed = bool(best_run["safe"])

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, "risk_controller.pt")
    torch.save(best_state, checkpoint_path)
    top_runs = sorted(
        all_runs,
        key=lambda run: policy_sort_key(
            run, expected_domains, args.min_domain_net, args.min_total_net
        ),
        reverse=True,
    )[:20]
    config = {
        "scoring_backend": "base_qwen_next_token",
        "method": "Three-domain cost-sensitive DINO KEEP/SWITCH/ABSTAIN risk controller",
        "train": os.path.realpath(args.train),
        "validation": os.path.realpath(args.val),
        "feature_names": feature_names,
        "labels": LABELS,
        "normalization": {"mean": mean.tolist(), "scale": scale.tolist()},
        "model": {
            "input_dim": len(feature_names),
            "hidden_dim": 64,
            "dropout": args.dropout,
        },
        "training": {
            "epochs_requested": args.epochs,
            "best_epoch": best_epoch,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "class_weights": {
                "keep": args.keep_weight,
                "switch": args.switch_weight,
                "abstain": args.abstain_weight,
            },
            "false_switch_penalty": args.false_switch_penalty,
            "missed_switch_penalty": args.missed_switch_penalty,
            "abstain_switch_penalty": args.abstain_switch_penalty,
        },
        "selection_constraints": {
            "expected_domains": expected_domains,
            "min_domain_net": args.min_domain_net,
            "min_total_net": args.min_total_net,
        },
        "selected_policy": strip_decisions(best_run),
        "classification": best_classification,
        "top_policies": [strip_decisions(run) for run in top_runs],
        "history": history,
        "passed": passed,
    }
    config_path = os.path.join(args.output_dir, "risk_controller_config.json")
    write_json(config_path, config)

    print("=== Frozen TRAIN-calibration policy ===")
    print(json.dumps(strip_decisions(best_run), indent=2, ensure_ascii=False))
    print(f"RISK CONTROLLER TRAIN CHECK: {'PASS' if passed else 'FAIL'}")
    print(f"checkpoint saved to: {checkpoint_path}")
    print(f"config saved to: {config_path}")
    if args.strict and not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
