"""Three-panel sensitivity analysis with explicit missing-evidence gaps."""
import argparse
import math
from pathlib import Path

from analysis.common import average
from analysis.replay_thresholds import replay_split
from crvg.utils.data import write_json, results


def sweep(log_dir, datasets, grids, frozen):
    curves = {}
    for parameter, values in grids.items():
        points = []
        for value in values:
            config = {**frozen, parameter: value}
            reports, counts = [], []
            try:
                for dataset in datasets:
                    output, report = replay_split(log_dir, dataset, **config)
                    reports.append(report["vs_greedy"])
                    rows = results(output)
                    field = {"gamma0": "cascade_entered", "gate": "qwen_changed", "gamma1": "dino_routed"}[parameter]
                    counts.append((sum(bool(r["crvg"][field]) for r in rows), len(rows)))
                points.append({"threshold": value, "available": True,
                               "average_delta_pp": average(reports),
                               "split_delta_pp": [r["delta_pp"]["acc0.5"] for r in reports],
                               "intervention_pct": 100*sum(n for n, _ in counts)/sum(n for _, n in counts)})
            except (ValueError, FileNotFoundError) as error:
                points.append({"threshold": value, "available": False, "reason": str(error)})
                print(f"[unavailable] {parameter}={value}: {error}")
        curves[parameter] = points
    return curves


def plot(curves, frozen, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    matplotlib.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.75), sharey=True)
    titles = ["(a) Cascade entry", "(b) Qwen update", "(c) DINO route"]
    labels = [r"Consensus threshold $\gamma_0$", r"Margin threshold $\delta_Q$", r"Consensus threshold $\gamma_1$"]
    for ax, (parameter, points), title, label in zip(axes, curves.items(), titles, labels):
        x = [p["threshold"] for p in points]
        y = [p["average_delta_pp"]["acc0.5"] if p["available"] else math.nan for p in points]
        nsplit = next((len(p["split_delta_pp"]) for p in points if p["available"]), 0)
        for split in range(nsplit):
            ax.plot(x, [p["split_delta_pp"][split] if p["available"] else math.nan for p in points],
                    color=".8", lw=.7, zorder=1)
        ax.plot(x, y, "o-", color="black", markerfacecolor="white", lw=1.3, ms=3, zorder=3)
        ax.axhline(0, color=".7", lw=.6, ls=":")
        ax.axvline(frozen[parameter], color="#D55E00", ls="--", lw=1)
        ax.grid(axis="y", alpha=.25)
        ax.set(title=title, xlabel=label)
        ax.spines["top"].set_visible(False)
        right = ax.twinx()
        right.plot(x, [p["intervention_pct"] if p["available"] else math.nan for p in points],
                   "s--", color="#0072B2", ms=3, lw=1)
        right.set_ylim(bottom=0)
        right.tick_params(axis="y", colors="#0072B2", labelsize=7)
        right.spines["top"].set_visible(False)
        for p in points:
            if p["available"] and abs(p["threshold"]-frozen[parameter]) < 1e-9:
                ax.plot(p["threshold"], p["average_delta_pp"]["acc0.5"], "D", ms=5, color="#D55E00", zorder=5)
    axes[0].set_ylabel("Average delta Acc@0.50 (pp)")
    right.set_ylabel("Intervention rate (%)", color="#0072B2")
    handles = [Line2D([], [], color="black", marker="o", markerfacecolor="white", label="Average gain"),
               Line2D([], [], color=".8", label="Individual split"),
               Line2D([], [], color="#0072B2", marker="s", ls="--", label="Intervention rate"),
               Line2D([], [], color="#D55E00", marker="D", ls="--", label="Frozen value")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=7)
    fig.subplots_adjust(left=.09, right=.92, bottom=.22, top=.77, wspace=.44)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(Path(output_dir)/f"threshold_sensitivity.{extension}", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gamma0", type=float, default=.5)
    parser.add_argument("--gate", type=float, default=.3)
    parser.add_argument("--gamma1", type=float, default=.35)
    parser.add_argument("--gamma0-grid", default="0.35,0.4,0.45,0.5,0.6,0.75")
    parser.add_argument("--gate-grid", default="0,0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--gamma1-grid", default="0.2,0.25,0.3,0.35,0.4,0.5")
    args = parser.parse_args()
    grids = {key: sorted(set(map(float, getattr(args, key+"_grid").split(","))) | {getattr(args, key)})
             for key in ("gamma0", "gate", "gamma1")}
    frozen = {key: getattr(args, key) for key in grids}
    curves = sweep(args.log_dir, args.datasets, grids, frozen)
    write_json(Path(args.output_dir)/"threshold_sensitivity.json", {"frozen": frozen, "curves": curves})
    plot(curves, frozen, args.output_dir)
    caption = ("One threshold varies at a time; other thresholds and the controller are frozen. "
               "Gain is measured against the same full-split greedy baseline and averaged equally across splits. "
               "Intervention rate counts stage-specific routed or updated samples divided by all evaluated samples. "
               "Gaps indicate insufficient or invalid cached evidence, not zero gain. This is sensitivity analysis, "
               "not evidence that the frozen values maximize a training objective.\n")
    (Path(args.output_dir)/"threshold_sensitivity_caption.txt").write_text(caption, encoding="utf-8")


if __name__ == "__main__":
    main()
