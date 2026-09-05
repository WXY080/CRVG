"""Select the three CRVG thresholds from difficult TRAIN examples only.

The cascade-entry and Qwen-update curves use aligned BoN, broad-ECE and
frozen-Qwen caches from the three REC training domains. The DINO-route curve
uses the released held-out TRAIN calibration examples and frozen controller.
No validation or test split is accepted by this analysis.
"""
import argparse
import csv
import math
from pathlib import Path
from statistics import mean

import torch

from crvg.controller.apply import load_controller
from crvg.controller.features import FEATURE_NAMES, canonical_domain
from crvg.controller.model import evaluate_policy, policy_is_safe
from crvg.utils.bbox import iou_xywh, min_pairwise_iou
from crvg.utils.data import (
    check_source,
    current_bbox,
    current_iou,
    index_rows,
    read_json,
    read_jsonl,
    results,
    write_json,
)

TRAIN_DOMAINS = ("refcoco", "refcoco+", "refcocog")


def parse_floats(value):
    values = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not values or any(not math.isfinite(item) for item in values):
        raise argparse.ArgumentTypeError("expected a nonempty list of finite numbers")
    return values


def row_domain(row):
    for value in (row.get("dataset"), row.get("training_source"), row.get("source")):
        domain = canonical_domain(value)
        if domain:
            return domain
    raise ValueError("Every threshold-selection row must identify a REC TRAIN domain")


def assert_train_payload(payload, path):
    rows = results(payload)
    if not rows:
        raise ValueError(f"Empty TRAIN cache: {path}")
    for row in rows:
        marker = " ".join(str(row.get(key, "")).lower()
                          for key in ("dataset", "split", "training_source"))
        if any(token in marker for token in ("_val", "testa", "testb", "_test")):
            raise ValueError(f"Evaluation row found in TRAIN-only input: {path}")
    return rows


def load_bundles(base_paths, expanded_paths, pick_paths):
    if not (len(base_paths) == len(expanded_paths) == len(pick_paths)):
        raise ValueError("--base-pools, --expanded-pools and --qwen-picks must align")
    bundles = []
    observed = set()
    for base_path, expanded_path, pick_path in zip(base_paths, expanded_paths, pick_paths):
        base = read_json(base_path)
        expanded = read_json(expanded_path)
        picks = read_json(pick_path)
        base_rows = assert_train_payload(base, base_path)
        expanded_rows = assert_train_payload(expanded, expanded_path)
        check_source(expanded, base)
        check_source(picks, expanded)
        base_index = index_rows(base_rows)
        expanded_index = index_rows(expanded_rows)
        pick_index = index_rows(picks.get("picks", []))
        if base_index.keys() != expanded_index.keys():
            raise ValueError(f"Base and expanded TRAIN caches are not aligned: {base_path}")
        domains = {row_domain(row) for row in base_rows}
        if len(domains) != 1:
            raise ValueError(f"A TRAIN cache must contain exactly one domain: {base_path}")
        domain = domains.pop()
        observed.add(domain)
        bundles.append((domain, base_index, expanded_index, pick_index, expanded))
    if observed != set(TRAIN_DOMAINS):
        raise ValueError(f"Expected the three TRAIN domains, found {sorted(observed)}")
    return bundles


def qwen_iou(row, record, gate):
    if not record or record.get("status") != "scored":
        raise ValueError("Frozen Qwen scores are missing for a threshold-eligible TRAIN row")
    candidates = record.get("candidates", [])
    if not candidates:
        raise ValueError("Frozen Qwen record has no candidate scores")
    winner = max(candidates, key=lambda item: float(item["p_yes"]))
    advantage = float(winner["p_yes"]) - float(record["current_probability"])
    if advantage > gate:
        gt = row.get("gt_bbox")
        return iou_xywh(winner["bbox"], gt) if gt is not None else None, winner["bbox"] != current_bbox(row)
    return current_iou(row), False


def summarize(stage, threshold, records, max_qwen_active_pct=10.0):
    domain = {}
    all_before, all_after = [], []
    active = rescue = damage = 0
    failures = []
    for name in TRAIN_DOMAINS:
        subset = [record for record in records if record[0] == name]
        before = [record[1] for record in subset]
        after = [record[2] for record in subset]
        if not before or any(value is None for value in before + after):
            raise ValueError(f"Missing ground truth or domain support for {name}")
        domain_rescue = sum(left < .5 <= right for left, right in zip(before, after))
        domain_damage = sum(right < .5 <= left for left, right in zip(before, after))
        domain[name] = {
            "n": len(before),
            "dacc50_pp": 100.0 * (sum(value >= .5 for value in after) / len(after)
                                    - sum(value >= .5 for value in before) / len(before)),
            "dmiou_pp": 100.0 * (mean(after) - mean(before)),
            "rescue": domain_rescue,
            "damage": domain_damage,
            "net": domain_rescue - domain_damage,
        }
        all_before.extend(before)
        all_after.extend(after)
        active += sum(bool(record[3]) for record in subset)
        rescue += domain_rescue
        damage += domain_damage
        if domain[name]["net"] < 0:
            failures.append(f"{name}_net")
    average_dacc = mean(item["dacc50_pp"] for item in domain.values())
    average_dmiou = mean(item["dmiou_pp"] for item in domain.values())
    active_pct = 100.0 * active / len(records)
    if average_dmiou < 0:
        failures.append("average_dmiou")
    if rescue - damage < 0:
        failures.append("total_net")
    if stage == "gate" and active_pct > max_qwen_active_pct + 1e-12:
        failures.append("qwen_intervention_budget")
    return {
        "stage": stage,
        "threshold": float(threshold),
        "n": len(records),
        "average_dacc50_pp": average_dacc,
        "average_dmiou_pp": average_dmiou,
        "overall_dacc50_pp": 100.0 * (sum(value >= .5 for value in all_after) / len(all_after)
                                       - sum(value >= .5 for value in all_before) / len(all_before)),
        "overall_dmiou_pp": 100.0 * (mean(all_after) - mean(all_before)),
        "active": active,
        "active_pct": active_pct,
        "rescue": rescue,
        "damage": damage,
        "net": rescue - damage,
        "safe": not failures,
        "constraint_failures": failures,
        "domain": domain,
    }


def gain_selection_key(run):
    return (int(run["safe"]), run["average_dacc50_pp"],
            min(run["domain"][name]["dacc50_pp"] for name in TRAIN_DOMAINS),
            run["net"], run["average_dmiou_pp"], -run["damage"],
            -run["active_pct"], -run["threshold"])


def sweep_gate(bundles, thresholds, gamma0, max_active):
    output = []
    for gate in thresholds:
        records = []
        for domain, base, expanded, picks, _ in bundles:
            for key, base_row in base.items():
                expanded_row = expanded[key]
                boxes = [candidate["bbox"] for candidate in base_row["candidates"]]
                eligible = len(boxes) >= 2 and min_pairwise_iou(boxes) < gamma0
                before = current_iou(expanded_row) if eligible else current_iou(base_row)
                after, changed = (qwen_iou(expanded_row, picks.get(key), gate)
                                  if eligible else (before, False))
                records.append((domain, before, after, changed))
        output.append(summarize("gate", gate, records, max_active))
    return output


def sweep_gamma0(bundles, thresholds, gate, max_active):
    output = []
    for gamma0 in thresholds:
        records = []
        for domain, base, expanded, picks, expanded_payload in bundles:
            broad = float(expanded_payload.get("meta", {}).get("ece_gamma0", -1))
            if gamma0 > broad + 1e-12:
                raise ValueError(
                    f"gamma0={gamma0:g} exceeds broad ECE cache coverage {broad:g}"
                )
            for key, base_row in base.items():
                before = current_iou(base_row)
                boxes = [candidate["bbox"] for candidate in base_row["candidates"]]
                active = len(boxes) >= 2 and min_pairwise_iou(boxes) < gamma0
                if active:
                    after, _ = qwen_iou(expanded[key], picks.get(key), gate)
                else:
                    after = before
                records.append((domain, before, after, active))
        output.append(summarize("gamma0", gamma0, records, max_active))
    return output


@torch.inference_mode()
def sweep_gamma1(calibration_path, controller_path, thresholds):
    rows = read_jsonl(calibration_path)
    if not rows:
        raise ValueError("DINO TRAIN calibration data is empty")
    model, center, scale, config = load_controller(controller_path)
    features = torch.tensor([row["features"] for row in rows], dtype=torch.float32)
    probabilities = model((features - center) / scale).softmax(-1)
    policy = config["selected_policy"]
    expected = tuple(config["selection_constraints"]["expected_domains"])
    feature_index = FEATURE_NAMES.index("existing_pool_min_iou")
    sample_min_iou = {}
    for row in rows:
        sample_min_iou.setdefault(row["sample_id"], float(row["features"][feature_index]))
    total_samples = len(sample_min_iou)
    deployment_fraction = float(config["selection_constraints"]["deployment_route_fraction"])
    output = []
    for threshold in thresholds:
        keep = {sample_id for sample_id, value in sample_min_iou.items() if value < threshold}
        indices = [index for index, row in enumerate(rows) if row["sample_id"] in keep]
        if not indices:
            raise ValueError(f"No DINO calibration examples route at gamma1={threshold:g}")
        subset = [rows[index] for index in indices]
        run = evaluate_policy(
            subset,
            probabilities[indices],
            gate=float(policy["gate"]),
            damage_cost=float(policy["damage_cost"]),
            abstain_cost=float(policy["abstain_cost"]),
            require_permutation_agree=bool(policy["require_permutation_agree"]),
        )
        failures = []
        if not policy_is_safe(run, expected, min_domain_net=0, min_total_net=0):
            failures.append("risk_guardrail")
        domain = {
            name: {
                "n": int(run["domain"].get(name, {}).get("n", 0)),
                "dacc50_pp": 100.0 * float(run["domain"].get(name, {}).get("delta_acc50", 0)),
                "dmiou_pp": 100.0 * float(run["domain"].get(name, {}).get("delta_miou", 0)),
                "rescue": int(run["domain"].get(name, {}).get("rescue", 0)),
                "damage": int(run["domain"].get(name, {}).get("damage", 0)),
                "net": int(run["domain"].get(name, {}).get("net", 0)),
            }
            for name in TRAIN_DOMAINS
        }
        routed_samples = int(run["n"])
        output.append({
            "stage": "gamma1",
            "threshold": float(threshold),
            "n": routed_samples,
            "average_dacc50_pp": mean(item["dacc50_pp"] for item in domain.values()),
            "average_dmiou_pp": mean(item["dmiou_pp"] for item in domain.values()),
            "overall_dacc50_pp": 100.0 * float(run["delta_acc50"]),
            "overall_dmiou_pp": 100.0 * float(run["delta_miou"]),
            "active": routed_samples,
            "active_pct": 100.0 * deployment_fraction * routed_samples / total_samples,
            "rescue": int(run["rescue"]),
            "damage": int(run["damage"]),
            "net": int(run["net"]),
            "safe": not failures,
            "constraint_failures": failures,
            "domain": domain,
        })
    return output


def dino_selection_key(run):
    return (int(run["safe"]), run["net"],
            min(run["domain"][name]["net"] for name in TRAIN_DOMAINS),
            run["average_dmiou_pp"], -run["damage"], -run["threshold"])


def declared_run(runs, threshold, stage):
    matches = [run for run in runs if abs(run["threshold"] - threshold) < 1e-12]
    if not matches:
        raise ValueError(f"Declared {stage} threshold {threshold:g} is absent from its grid")
    return matches[0]


def save_csv(path, stages, selected):
    fields = ["stage", "threshold", "selected", "safe", "average_dacc50_pp",
              "average_dmiou_pp", "active_pct", "active", "rescue", "damage",
              "net", "constraint_failures"]
    for domain in TRAIN_DOMAINS:
        fields.extend((f"dacc50_pp_{domain}", f"dmiou_pp_{domain}", f"net_{domain}"))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for stage, runs in stages.items():
            for run in runs:
                output = {
                    **{key: run.get(key) for key in fields},
                    "selected": int(abs(run["threshold"] - selected[stage]["threshold"]) < 1e-12),
                    "constraint_failures": ";".join(run["constraint_failures"]),
                }
                for domain in TRAIN_DOMAINS:
                    output[f"dacc50_pp_{domain}"] = run["domain"][domain]["dacc50_pp"]
                    output[f"dmiou_pp_{domain}"] = run["domain"][domain]["dmiou_pp"]
                    output[f"net_{domain}"] = run["domain"][domain]["net"]
                writer.writerow(output)


def save_summary(path, selected, diagnostic, guard):
    lines = [
        "# TRAIN-Only Threshold Selection",
        "",
        "All operating points are computed from difficult TRAIN examples; no validation or test split is used.",
        "",
        "| Stage | Paper point | Average dAcc50 (pp) | Average dIoU (pp) | Intervention (%) | Net | Safe | Diagnostic objective best |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    labels = {"gamma0": "Cascade entry", "gate": "Qwen update", "gamma1": "DINO route"}
    for stage in ("gamma0", "gate", "gamma1"):
        run = selected[stage]
        lines.append(
            f"| {labels[stage]} | {run['threshold']:.2f} | "
            f"{run['average_dacc50_pp']:+.3f} | {run['average_dmiou_pp']:+.3f} | "
            f"{run['active_pct']:.2f} | {run['net']:+d} | "
            f"{'yes' if run['safe'] else 'no'} | {diagnostic[stage]['threshold']:.2f} |"
        )
    lines.extend([
        "",
        "The diagnostic-best column reports the best point under the script's scalar tie-breaking rule; it is retained as an audit aid and does not replace the manuscript's frozen operating point.",
        "",
        f"The feasibility guard `|B0| < 2` applies to {guard['count']}/{guard['total']} difficult TRAIN inputs ({guard['pct']:.3f}%). It is not swept because pairwise consensus is undefined for fewer than two boxes.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(stages, selected, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size": 7.2, "axes.labelsize": 7.2,
                         "axes.titlesize": 7.6, "xtick.labelsize": 6.8,
                         "ytick.labelsize": 6.8, "legend.fontsize": 6.3,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none"})
    panels = (("gamma0", "(a) Cascade entry", r"Consensus threshold $\gamma_0$"),
              ("gate", "(b) Qwen update", r"Margin threshold $\delta_Q$"),
              ("gamma1", "(c) DINO route", r"Consensus threshold $\gamma_1$"))
    values = [value for runs in stages.values() for run in runs
              for value in ([run["average_dacc50_pp"]]
                            + [run["domain"][name]["dacc50_pp"] for name in TRAIN_DOMAINS])]
    low, high = min(min(values), 0.0), max(max(values), 0.0)
    padding = max(.05, .08 * max(.1, high - low))
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.72), sharey=True)
    colors = {"blue": "#0072B2", "orange": "#D55E00", "red": "#A51C30"}
    for axis, (stage, title, xlabel) in zip(axes, panels):
        runs = stages[stage]
        x = [run["threshold"] for run in runs]
        for name in TRAIN_DOMAINS:
            axis.plot(x, [run["domain"][name]["dacc50_pp"] for run in runs],
                      color="#AFAFAF", lw=.7, alpha=.55, zorder=1)
        axis.plot(x, [run["average_dacc50_pp"] for run in runs], "o-",
                  color="black", markerfacecolor="white", lw=1.35, ms=3.2, zorder=4)
        unsafe = [run for run in runs if not run["safe"]]
        if unsafe:
            axis.scatter([run["threshold"] for run in unsafe],
                         [run["average_dacc50_pp"] for run in unsafe],
                         marker="x", color=colors["red"], s=20, lw=1, zorder=6)
        chosen = selected[stage]
        axis.axvline(chosen["threshold"], color=colors["orange"], ls=(0, (3, 2)), lw=1)
        axis.scatter(chosen["threshold"], chosen["average_dacc50_pp"], marker="D",
                     color=colors["orange"], edgecolor="white", lw=.5, s=24, zorder=7)
        offset = (6, 8) if stage != "gamma1" else (6, -8)
        axis.annotate(f"selected {chosen['threshold']:.2f}\n"
                      f"{chosen['average_dacc50_pp']:+.3f} pp, {chosen['active_pct']:.2f}%",
                      (chosen["threshold"], chosen["average_dacc50_pp"]),
                      xytext=offset, textcoords="offset points", ha="left",
                      va="bottom" if offset[1] > 0 else "top", color=colors["orange"],
                      fontsize=5.9, bbox={"facecolor": "white", "edgecolor": "none", "pad": .2})
        axis.axhline(0, color="#777777", lw=.65, ls=":")
        axis.set(title=title, xlabel=xlabel, ylim=(low-padding, high+padding))
        axis.grid(axis="y", color="#E5E5E5", lw=.5)
        axis.spines["top"].set_visible(False)
        rate = axis.twinx()
        rate.plot(x, [run["active_pct"] for run in runs], "s--",
                  color=colors["blue"], ms=2.8, lw=1)
        rate.set_ylim(0, max(1, 1.12 * max(run["active_pct"] for run in runs)))
        rate.tick_params(axis="y", colors=colors["blue"], pad=1.5)
        rate.spines["top"].set_visible(False)
        rate.spines["right"].set_color(colors["blue"])
        if axis is axes[-1]:
            rate.set_ylabel("Intervention rate (%)", color=colors["blue"])
    axes[0].set_ylabel(r"Average training $\Delta$Acc@0.50 (pp)")
    handles = [Line2D([], [], color="black", marker="o", markerfacecolor="white",
                      label="Average training gain"),
               Line2D([], [], color="#AFAFAF", label="Training domain"),
               Line2D([], [], color=colors["blue"], marker="s", ls="--",
                      label="Intervention rate"),
               Line2D([], [], color=colors["red"], marker="x", ls="none",
                      label="Constraint violation"),
               Line2D([], [], color=colors["orange"], marker="D", ls=(0, (3, 2)),
                      label="Selected point")]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False)
    fig.subplots_adjust(left=.075, right=.93, bottom=.20, top=.78, wspace=.23)
    paths = []
    for extension, dpi in (("pdf", None), ("svg", None), ("png", 600)):
        path = output_dir / f"training_threshold_selection.{extension}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-pools", nargs="+", required=True)
    parser.add_argument("--expanded-pools", nargs="+", required=True)
    parser.add_argument("--qwen-picks", nargs="+", required=True)
    parser.add_argument("--dino-calibration", required=True)
    parser.add_argument("--controller", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gamma0-grid", type=parse_floats,
                        default=parse_floats(".4,.5,.6,.7,.75,.8,.85,.9,.95"))
    parser.add_argument("--gate-grid", type=parse_floats,
                        default=parse_floats("0,.05,.1,.15,.2,.25,.3,.35,.4,.5"))
    parser.add_argument("--gamma1-grid", type=parse_floats,
                        default=parse_floats(".2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.75"))
    parser.add_argument("--selected-gamma0", type=float, default=.5)
    parser.add_argument("--selected-gate", type=float, default=.3)
    parser.add_argument("--selected-gamma1", type=float, default=.35)
    parser.add_argument("--gate-calibration-gamma0", type=float, default=.75)
    parser.add_argument("--max-qwen-intervention-pct", type=float, default=10.0)
    parser.add_argument("--require-selected", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundles = load_bundles(args.base_pools, args.expanded_pools, args.qwen_picks)
    guard_total = sum(len(base) for _, base, _, _, _ in bundles)
    guard_count = sum(
        len(row.get("candidates", [])) < 2
        for _, base, _, _, _ in bundles for row in base.values()
    )
    guard = {"count": guard_count, "total": guard_total,
             "pct": 100.0 * guard_count / max(1, guard_total)}
    stages = {}
    stages["gate"] = sweep_gate(bundles, args.gate_grid,
                                args.gate_calibration_gamma0,
                                args.max_qwen_intervention_pct)
    diagnostic_gate = max(stages["gate"], key=gain_selection_key)
    stages["gamma0"] = sweep_gamma0(bundles, args.gamma0_grid,
                                    args.selected_gate,
                                    args.max_qwen_intervention_pct)
    diagnostic_gamma0 = max(stages["gamma0"], key=gain_selection_key)
    stages["gamma1"] = sweep_gamma1(args.dino_calibration, args.controller,
                                    args.gamma1_grid)
    expected = {"gamma0": args.selected_gamma0, "gate": args.selected_gate,
                "gamma1": args.selected_gamma1}
    selected = {key: declared_run(stages[key], value, key)
                for key, value in expected.items()}
    diagnostic = {"gamma0": diagnostic_gamma0, "gate": diagnostic_gate,
                  "gamma1": max(stages["gamma1"], key=dino_selection_key)}
    matches = {key: selected[key]["safe"] for key in expected}
    payload = {"method": "TRAIN-only risk-constrained threshold selection",
               "selection_order": ["gate", "gamma0", "gamma1"],
               "constraints": {"max_qwen_intervention_pct": args.max_qwen_intervention_pct,
                               "gate_calibration_gamma0": args.gate_calibration_gamma0,
                               "nonnegative_average_miou": True,
                               "nonnegative_domain_net": True},
               "sources": {"base_pools": list(map(str, args.base_pools)),
                           "expanded_pools": list(map(str, args.expanded_pools)),
                           "qwen_picks": list(map(str, args.qwen_picks)),
                           "dino_calibration": str(args.dino_calibration),
                           "controller": str(args.controller)},
               "candidate_guard": guard,
               "selected_thresholds": {key: value["threshold"] for key, value in selected.items()},
               "paper_thresholds": expected, "paper_points_satisfy_constraints": matches,
               "diagnostic_objective_best": {
                   key: value["threshold"] for key, value in diagnostic.items()
               },
               "stages": stages}
    write_json(output_dir / "training_threshold_selection.json", payload)
    save_csv(output_dir / "training_threshold_selection.csv", stages, selected)
    save_summary(output_dir / "training_threshold_selection_summary.md",
                 selected, diagnostic, guard)
    caption = ("Average training gain and intervention rate under risk-constrained threshold selection. "
               "Black curves average the three REC training domains; gray curves show individual "
               "training domains. Blue curves report the stage-specific intervention rate, crosses "
               "mark constraint violations, and orange markers denote the selected operating points.")
    (output_dir / "training_threshold_selection_caption.txt").write_text(caption + "\n", encoding="utf-8")
    figure_paths = plot(stages, selected, output_dir)
    for stage in ("gamma0", "gate", "gamma1"):
        run = selected[stage]
        print(f"{stage}: selected={run['threshold']:.2f} "
              f"average_dAcc50={run['average_dacc50_pp']:+.3f}pp "
              f"active={run['active_pct']:.2f}% safe={run['safe']}")
    print("Saved:", output_dir / "training_threshold_selection.json")
    for path in figure_paths:
        print("Saved:", path)
    if args.require_selected and not all(matches.values()):
        failed = [key for key, safe in matches.items() if not safe]
        raise SystemExit(f"Declared paper operating points fail TRAIN constraints: {failed}")


if __name__ == "__main__":
    main()
