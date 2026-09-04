"""Count actual routed samples and scored images, not inferred latency."""
import argparse

from analysis.common import paths
from crvg.utils.data import DATASETS, read_json, write_json, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    args = parser.parse_args()
    report = {}
    for dataset in args.datasets:
        p = paths(args.log_dir, dataset)
        rows = results(read_json(p["final"]))
        crops, pairs = read_json(p["crops"])["picks"], read_json(p["pairs"])["picks"]
        n = len(rows)
        counts = {key: sum(bool(r.get("crvg", {}).get(key)) for r in rows)
                  for key in ("cascade_entered", "qwen_changed", "dino_routed")}
        report[dataset] = {"n": n, **counts,
            "crop_model_inputs": sum(1+len(r["candidates"]) for r in crops if r["status"] == "scored"),
            "pairwise_model_inputs": 2*sum(len(r.get("challengers", [])) for r in pairs),
            "dino_route_pct": 100*counts["dino_routed"]/n if n else None}
        print(dataset, report[dataset])
    write_json(p["base"].parent/"routing_efficiency.json", report)


if __name__ == "__main__":
    main()
