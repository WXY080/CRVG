"""Report equal-weight split averages; missing splits never become zeros."""
import argparse
import csv
from pathlib import Path

from analysis.common import compare, paths
from crvg.utils.data import DATASETS, read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    args = parser.parse_args()
    table, reports = [], {}
    for dataset in args.datasets:
        p = paths(args.log_dir, dataset)
        base, qwen, final = (read_json(p[key]) for key in ("base", "qwen", "final"))
        original, residual = compare(base, final), compare(qwen, final)
        reports[dataset] = {"vs_greedy": original, "vs_qwen": residual}
        table.append([dataset, original["n"], 100*original["before"]["acc0.5"],
                      100*residual["before"]["acc0.5"], 100*original["after"]["acc0.5"],
                      residual["delta_pp"]["acc0.5"],
                      100*original["after"]["acc0.75"], 100*original["after"]["acc0.9"],
                      100*original["after"]["miou"]])
    table.append(["Average", sum(r[1] for r in table)] +
                 [sum(r[i] for r in table)/len(table) for i in range(2, 9)])
    headers = ["Dataset", "N", "Greedy Acc50", "Qwen Acc50", "CRVG Acc50",
               "Delta vs Qwen (pp)", "CRVG Acc75", "CRVG Acc90", "CRVG mIoU"]
    text = "| " + " | ".join(headers) + " |\n"
    text += "| " + " | ".join(["---"]*len(headers)) + " |\n"
    for row in table:
        text += "| " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in row) + " |\n"
    root = Path(args.log_dir)
    (root/"results_table.md").write_text(text, encoding="utf-8")
    with open(root/"results_table.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(table)
    write_json(root/"results_table.json", {"reports": reports, "average_definition": "equal-weight split mean"})
    print(text)


if __name__ == "__main__":
    main()
