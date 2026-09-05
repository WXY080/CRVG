"""TRAIN-only threshold-selection contracts."""
import unittest
import tempfile
from pathlib import Path

from analysis.training_threshold_selection import (
    convert_dino_report,
    gain_selection_key,
    load_bundles,
    parse_floats,
    same_threshold,
    summarize,
    sweep_gamma0,
    sweep_gate,
)
from analysis.dino_route_calibration import selection_key as dino_selection_key
from crvg.utils.data import fingerprint, write_json


class TrainingThresholdSelectionTests(unittest.TestCase):
    def test_parse_floats_is_sorted_and_unique(self):
        self.assertEqual(parse_floats(".5,.3,.5"), [.3, .5])

    def test_qwen_budget_marks_an_otherwise_positive_point_unsafe(self):
        records = []
        for domain in ("refcoco", "refcoco+", "refcocog"):
            records.extend((domain, .4, .6, True) for _ in range(2))
        run = summarize("gate", .1, records, max_qwen_active_pct=10)
        self.assertGreater(run["average_dacc50_pp"], 0)
        self.assertFalse(run["safe"])
        self.assertIn("qwen_intervention_budget", run["constraint_failures"])

    def test_safe_point_is_selected_before_larger_unsafe_gain(self):
        domain = {name: {"dacc50_pp": 1.0} for name in ("refcoco", "refcoco+", "refcocog")}
        safe = {"safe": True, "average_dacc50_pp": 1.0, "domain": domain,
                "net": 1, "average_dmiou_pp": 1.0, "damage": 0,
                "active_pct": 5.0, "threshold": .3}
        unsafe = {**safe, "safe": False, "average_dacc50_pp": 10.0, "threshold": .1}
        self.assertIs(max((unsafe, safe), key=gain_selection_key), safe)

    def test_dino_selection_uses_overall_miou_after_net(self):
        lower = {"safe": True, "net": 10, "damage": 1, "delta_miou": .03,
                 "threshold": .3, "domain": {"refcoco": {"net": 4},
                 "refcoco+": {"net": 0}, "refcocog": {"net": 6}}}
        higher = {**lower, "delta_miou": .04, "threshold": .35}
        self.assertIs(max((lower, higher), key=dino_selection_key), higher)

    def test_dino_report_preserves_serialized_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dino.json"
            def run(threshold, delta_miou):
                domain = {name: {"n": 3, "delta_acc50": .01,
                                  "delta_miou": .01, "rescue": 1,
                                  "damage": 0, "net": 1}
                          for name in ("refcoco", "refcoco+", "refcocog")}
                return {"threshold": threshold, "n": 9, "switches": 3,
                        "rescue": 3, "damage": 0, "net": 3,
                        "delta_acc50": .01, "delta_miou": delta_miou,
                        "safe": True, "domain": domain}
            write_json(path, {"base_calibration_samples": 10,
                              "selected_threshold": .35,
                              "runs": [run(.3, .04), run(.35, .05)]})
            runs, selected, _ = convert_dino_report(path)
            self.assertEqual(len(runs), 2)
            self.assertTrue(same_threshold(selected["threshold"], .35))

    def test_three_domain_cached_qwen_sweep(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_paths, expanded_paths, pick_paths = [], [], []
            for index, domain in enumerate(("refcoco", "refcoco+", "refcocog")):
                base = {"meta": {"complete": True}, "results": [{
                    "dataset_index": index,
                    "dataset": domain + "_train",
                    "split": "train",
                    "image": f"{index}.jpg",
                    "text": "right object",
                    "gt_bbox": [20, 0, 10, 10],
                    "pred_bbox": [0, 0, 10, 10],
                    "candidates": [{"bbox": [0, 0, 10, 10]},
                                   {"bbox": [20, 0, 10, 10]}],
                }]}
                expanded = {"meta": {"complete": True,
                                      "source_sha256": fingerprint(base),
                                      "ece_gamma0": .95},
                            "results": base["results"]}
                picks = {"meta": {"complete": True,
                                   "source_sha256": fingerprint(expanded)},
                         "picks": [{"dataset_index": index, "status": "scored",
                                    "current_probability": .1,
                                    "candidates": [{"bbox": [20, 0, 10, 10],
                                                    "p_yes": .9}]}]}
                paths = [root / f"{domain}-{name}.json"
                         for name in ("base", "expanded", "picks")]
                for path, payload in zip(paths, (base, expanded, picks)):
                    write_json(path, payload)
                base_paths.append(paths[0])
                expanded_paths.append(paths[1])
                pick_paths.append(paths[2])
            bundles = load_bundles(base_paths, expanded_paths, pick_paths)
            gate = sweep_gate(bundles, [.3], .75, 100)[0]
            gamma0 = sweep_gamma0(bundles, [.5], .3, 100)[0]
            self.assertEqual((gate["net"], gamma0["net"]), (3, 3))
            self.assertTrue(gate["safe"] and gamma0["safe"])


if __name__ == "__main__":
    unittest.main()
