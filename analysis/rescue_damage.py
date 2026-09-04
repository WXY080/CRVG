"""Full-split decision anatomy for direct DINO, pairwise, and controller."""
import argparse

from analysis.common import compare, paths
from crvg.utils.data import DATASETS, read_json, write_json, results, index_rows, row_key, set_prediction, check_source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    args = parser.parse_args()
    report = {}
    for dataset in args.datasets:
        p = paths(args.log_dir, dataset)
        source, evidence, picks, final = (read_json(p[key]) for key in ("qwen", "dino", "pairs", "final"))
        check_source(evidence, source)
        check_source(picks, evidence)
        pool, pairs = index_rows(results(evidence)), index_rows(picks["picks"])
        direct, pairwise = [], []
        for row in results(source):
            entry = pool.get(row_key(row), {})
            proposals = [c for c in entry.get("candidates", []) if "grounding_dino" in c.get("source", "")]
            best = max(proposals, key=lambda c: c.get("dino_phrase_score", 0)) if proposals else None
            direct.append(set_prediction(row, best["bbox"], "direct_dino") if best else row)
            candidates = [c for c in pairs.get(row_key(row), {}).get("challengers", [])
                          if c.get("permutation_agree") and c.get("alternative_advantage", 0) > 0]
            choice = max(candidates, key=lambda c: c["alternative_advantage"]) if candidates else None
            pairwise.append(set_prediction(row, choice["bbox"], "pairwise_zero_gate") if choice else row)
        report[dataset] = {name: compare(source, value) for name, value in {
            "direct_dino": direct, "pairwise_zero_gate": pairwise, "risk_controller": final}.items()}
        for name, result in report[dataset].items():
            print(f"{dataset} {name}: rescue={result['rescue']} damage={result['damage']} "
                  f"net={result['net']:+d} dAcc50={result['delta_pp']['acc0.5']:+.3f} pp")
    write_json(p["base"].parent/"rescue_damage.json", report)


if __name__ == "__main__":
    main()
