"""Replay crop acceptance. A weak crop decision never terminates the DINO route."""
import argparse

from crvg.utils.bbox import min_pairwise_iou
from crvg.utils.data import read_json, write_json, results, index_rows, row_key, check_source, set_prediction


def apply_crop(source, picks, gate=.3, gamma0=.5):
    check_source(picks, source)
    records = index_rows(picks["picks"])
    output = []
    for row in results(source):
        boxes = [c["bbox"] for c in row["candidates"]]
        state = dict(row.get("crvg", {}))
        count = state.get("b0_count", len(boxes))
        gamma = state.get("b0_min_iou", min_pairwise_iou(boxes))
        active = count >= 2 and gamma < gamma0
        record = records.get(row_key(row))
        result = dict(row)
        changed = False
        if not active and state.get("pre_ece_bbox") is not None:
            result = set_prediction(result, state["pre_ece_bbox"], "greedy")
            state["ece_changed"] = False
        if active:
            if not state.get("ece_available", False):
                raise ValueError(f"ECE evidence missing for routed sample {row_key(row)}")
            if not record or record.get("status") != "scored":
                raise ValueError(f"Missing crop scores for routed sample {row_key(row)}")
            candidate = max(record["candidates"], key=lambda c: c["p_yes"])
            if candidate["p_yes"] - record["current_probability"] > gate:
                changed = candidate["bbox"] != row["pred_bbox"]
                result = set_prediction(row, candidate["bbox"], "qwen_crop")
        state.update(cascade_entered=active, qwen_changed=changed, gamma0=gamma0,
                     qwen_gate=gate, b1_min_iou=min_pairwise_iou(boxes), b1_count=len(boxes))
        result["crvg"] = state
        output.append(result)
    return {"meta": {**source.get("meta", {}), "complete": True,
                     "gamma0": gamma0, "gate": gate}, "results": output}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--picks", required=True)
    parser.add_argument("--gate", type=float, default=.3)
    parser.add_argument("--gamma0", type=float, default=.5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    write_json(args.out, apply_crop(read_json(args.input), read_json(args.picks), args.gate, args.gamma0))


if __name__ == "__main__":
    main()
