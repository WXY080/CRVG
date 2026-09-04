"""Validate/import canonical caches or explicitly convert pixel-xywh PaDT caches."""
import argparse
import copy

from crvg.utils.data import read_json, write_json, results, index_rows, row_key, current_bbox, check_source, fingerprint
from crvg.utils.bbox import min_pairwise_iou


def convert_padt(base, ece, dataset, gamma0=.5):
    base = copy.deepcopy(base)
    records = index_rows(ece["picks"])
    converted = {}
    for row in results(base):
        row["dataset"] = dataset
        row["split"] = "train" if dataset.endswith("_train") else dataset.rsplit("_", 1)[-1]
        row["pred_bbox"] = current_bbox(row)
        candidates = row.get("candidates")
        if not candidates:
            raise ValueError("Every imported sample must include its initial candidate pool")
        active = len(candidates) >= 2 and min_pairwise_iou([c["bbox"] for c in candidates]) < gamma0
        record = records.get(row_key(row))
        ready = bool(record and record.get("status") == "generated_equivariant_candidates")
        if active and not ready:
            raise ValueError(f"Missing completed PaDT ECE evidence for {row_key(row)}")
        if ready and record.get("current_bbox") != row["pred_bbox"]:
            raise ValueError("PaDT ECE was generated for a different current prediction")
        converted[row_key(row)] = {"dataset_index": row["dataset_index"], "routed": ready,
                                  "transformed_candidates": record.get("transformed_candidates", []) if ready else []}
    base["meta"] = {**base.get("meta", {}), "dataset": dataset, "backbone": "padt", "complete": True}
    output = {"meta": {"source_sha256": fingerprint(base), "complete": True, "gamma0": gamma0,
                       "pad_factor": ece.get("meta", {}).get("pad_factor")},
              "records": converted}
    return base, output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--ece", required=True)
    parser.add_argument("--out-base", required=True)
    parser.add_argument("--out-ece", required=True)
    parser.add_argument("--format", choices=("canonical", "padt"), default="canonical")
    parser.add_argument("--dataset")
    parser.add_argument("--gamma0", type=float, default=.5)
    args = parser.parse_args()
    base, ece = read_json(args.base), read_json(args.ece)
    if args.format == "padt":
        if not args.dataset:
            parser.error("--dataset is required for PaDT conversion")
        base, ece = convert_padt(base, ece, args.dataset, args.gamma0)
    check_source(ece, base)
    if set(ece["records"]) != set(index_rows(results(base))):
        raise ValueError("ECE/base sample identity mismatch")
    write_json(args.out_base, base)
    write_json(args.out_ece, ece)


if __name__ == "__main__":
    main()
