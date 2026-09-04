"""Build B1 and apply the paper's provenance-preserving ECE consensus update."""
import argparse
import copy
from crvg.utils.bbox import min_pairwise_iou
from crvg.utils.data import read_json, write_json, results, row_key, check_source, fingerprint
from crvg.utils.data import set_prediction
from crvg.candidate_generation.consensus import consensus_update


def merge(base, ece, dedup_iou=.95, cluster_iou=.45, min_support=2, score_gate=.25):
    check_source(ece, base)
    records = ece["records"]
    output = []
    for row in results(base):
        key = row_key(row)
        if key not in records:
            raise ValueError(f"ECE record missing: {key}")
        record = records[key]
        result = copy.deepcopy(row)
        result["crvg"] = {"b0_count": len(row["candidates"]),
                          "b0_min_iou": min_pairwise_iou([c["bbox"] for c in row["candidates"]]),
                          "ece_available": record["routed"]}
        result["crvg"]["pre_ece_bbox"] = list(row["pred_bbox"])
        result["crvg"]["ece_changed"] = False
        if record["routed"]:
            pool, chosen, diagnostic = consensus_update(row, record["transformed_candidates"],
                                                        dedup_iou, cluster_iou, min_support=min_support,
                                                        score_gate=score_gate)
            result["candidates"] = pool
            result["crvg"]["ece_consensus"] = diagnostic
            if chosen:
                result = set_prediction(result, chosen["bbox"], "ece_consensus")
                result["crvg"]["ece_changed"] = chosen["bbox"] != row["pred_bbox"]
        result["num_candidates"] = len(result["candidates"])
        result["crvg"]["b1_count"] = len(result["candidates"])
        result["crvg"]["b1_min_iou"] = min_pairwise_iou([c["bbox"] for c in result["candidates"]])
        output.append(result)
    return {"meta": {**base.get("meta", {}), "source_sha256": fingerprint(base),
                     "complete": True, "ece_gamma0": ece["meta"]["gamma0"]}, "results": output}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--ece", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dedup-iou", type=float, default=.95)
    parser.add_argument("--cluster-iou", type=float, default=.45)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--score-gate", type=float, default=.25)
    args = parser.parse_args()
    write_json(args.output, merge(read_json(args.base), read_json(args.ece), args.dedup_iou,
                                  args.cluster_iou, args.min_support, args.score_gate))


if __name__ == "__main__":
    main()
