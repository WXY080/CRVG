"""Offline replay only where the cached model evidence remains valid."""
import argparse
from pathlib import Path

from analysis.common import paths, compare, average
from crvg.utils.data import read_json, write_json, fingerprint
from crvg.verification.apply_gate import apply_crop
from crvg.controller.apply import merge_decisions


def replay_split(log_dir, dataset, gamma0=.5, gate=.3, gamma1=.35):
    if not 0 < gamma1 <= gamma0 <= 1.01 or not 0 <= gate <= 1:
        raise ValueError("Invalid threshold combination")
    p = paths(log_dir, dataset)
    expanded, crops = read_json(p["expanded"]), read_json(p["crops"])
    evidence, risk = read_json(p["dino"]), read_json(p["risk"])
    original_qwen = read_json(p["qwen"])
    if risk.get("evidence_sha256") != fingerprint(evidence) or risk.get("source_sha256") != fingerprint(original_qwen):
        raise ValueError("Risk decisions do not match the cached evidence/current system")
    current = apply_crop(expanded, crops, gate, gamma0)
    selected = merge_decisions(current, evidence, risk["decisions"], gamma1)
    report = compare(read_json(p["final"]), selected)
    report["vs_greedy"] = compare(read_json(p["base"]), selected)
    return selected, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--gamma0", type=float, default=.5)
    parser.add_argument("--gate", type=float, default=.3)
    parser.add_argument("--gamma1", type=float, default=.35)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if Path(args.output_dir).resolve() == Path(args.log_dir).resolve():
        parser.error("Use a separate replay output directory")
    reports, outputs = {}, {}
    for dataset in args.datasets:
        outputs[dataset], reports[dataset] = replay_split(args.log_dir, dataset, args.gamma0, args.gate, args.gamma1)
    for dataset, output in outputs.items():
        write_json(Path(args.output_dir)/f"rec_results_{dataset}_replay.json", output)
        print(dataset, reports[dataset]["delta_pp"])
    summary = {"thresholds": {"gamma0": args.gamma0, "gate": args.gate, "gamma1": args.gamma1},
               "splits": reports, "average_delta_pp_vs_original": average(list(reports.values())),
               "interpretation": "Exact covered-cache replay; not threshold training or a GPU rerun."}
    write_json(Path(args.output_dir)/"replay_report.json", summary)
    print("Average delta vs original:", summary["average_delta_pp_vs_original"])


if __name__ == "__main__":
    main()
