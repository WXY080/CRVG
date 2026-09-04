"""Build all three TRAIN domains with a stable global image-group split."""
import argparse
from collections import Counter
from pathlib import Path

from crvg.controller.features import build_risk_examples, stable_fraction
from crvg.utils.data import read_json, write_json, write_jsonl, check_source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", nargs="+", required=True)
    parser.add_argument("--picks", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--calibration-fraction", type=float, default=.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if len(args.evidence) != len(args.picks) or not 0 < args.calibration_fraction < 1:
        parser.error("Provide one picks file per evidence file and a fraction in (0,1)")
    groups = {}
    for epath, ppath in zip(args.evidence, args.picks):
        evidence, picks = read_json(epath), read_json(ppath)
        check_source(picks, evidence)
        if picks.get("meta", {}).get("scoring_backend") != "base_qwen_next_token":
            raise ValueError("Evidence must declare the frozen-base Qwen scoring protocol")
        examples, _ = build_risk_examples(evidence, picks, Path(epath).name, require_train=True)
        for example in examples:
            key = (example["domain"], example["image_group"], example["expression"],
                   tuple(example["current_bbox"]), tuple(example["alternative_bbox"]))
            previous = groups.get(key)
            if previous and (previous["features"] != example["features"] or previous["label"] != example["label"]):
                raise ValueError("Conflicting duplicate example; do not mix different evidence policies")
            groups[key] = example
    train, calibration = [], []
    for row in groups.values():
        (calibration if stable_fraction(row["image_group"], args.seed) < args.calibration_fraction else train).append(row)
    expected = {"refcoco", "refcoco+", "refcocog"}
    if any({r["domain"] for r in split} != expected for split in (train, calibration)):
        raise ValueError("Both partitions need all three TRAIN domains; collect more natural evidence")
    if not train or not calibration:
        raise ValueError("Empty partition")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("calibration", calibration)):
        write_jsonl(output / f"{name}.jsonl", split)
    write_json(output / "audit.json", {"train": len(train), "calibration": len(calibration),
                                     "seed": args.seed, "roles": dict(Counter(r["label"] for r in train)),
                                     "train_images": len({r["image_group"] for r in train}),
                                     "calibration_images": len({r["image_group"] for r in calibration})})
    print(f"TRAIN/calibration examples: {len(train)} / {len(calibration)}")


if __name__ == "__main__":
    main()
