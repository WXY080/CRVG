import copy
import math
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch
import yaml

from analysis.common import compare, average
from crvg.candidate_generation.bon_dump import extract_bbox
from crvg.candidate_generation.consensus import connected_clusters, consensus_update
from crvg.candidate_generation.ece import inverse_bbox, swap_horizontal_words, build_views
from crvg.controller.apply import load_controller, score_evidence, merge_decisions
from crvg.controller.model import evaluate_policy, policy_score
from crvg.controller.features import FEATURE_NAMES, build_risk_examples, image_group, is_train_row, stable_fraction
from crvg.utils.bbox import iou_xywh, norm1000_to_pixel_xywh
from crvg.utils.data import index_rows, fingerprint, set_prediction
from crvg.verification.apply_gate import apply_crop
from crvg.verification.crop_verifier import crop_with_context
from crvg.verification.dino_detector import (dino_route, proposal_pool, normalized_phrase,
                                             merge_detections, attach_phrase_scores)
from crvg.verification.pairwise import aligned_scores, select_challengers
from crvg.verification.render import render_pairwise_relation_montage
from tools.merge_ece import merge
from tests.fixtures import A, B, C, case, evidence, controller


class CoreTests(unittest.TestCase):
    def test_frozen_paper_settings(self):
        config = yaml.safe_load((Path(__file__).parents[1]/"configs/default.yaml").read_text())
        from crvg.settings import DEFAULTS
        self.assertEqual(config, DEFAULTS)
        self.assertEqual((config["gamma0"], config["gate"], config["gamma1"]), (.5, .3, .35))
        self.assertEqual((config["ece_min_support"], config["dino_max_detections"], config["dino_max_challengers"]), (2, 12, 8))

    def test_iou_invalid(self):
        self.assertEqual(iou_xywh(A, A), 1)
        self.assertEqual(iou_xywh(A, B), 0)
        self.assertEqual(iou_xywh([0, 0, -1, 3], A), 0)
        self.assertEqual(iou_xywh([math.nan, 0, 1, 3], A), 0)

    def test_small_normalized_coordinates_are_not_unit_fractions(self):
        self.assertEqual(norm1000_to_pixel_xywh([0, 0, 1, 1], 100, 100), [0, 0, .1, .1])

    def test_bbox_answer_parser(self):
        self.assertEqual(extract_bbox('<think>[1,2,3,4]</think><answer>{"box":[10,20,30,40]}</answer>'), [10,20,30,40])
        self.assertEqual(extract_bbox("no box"), [0]*4)

    def test_inverse_views(self):
        self.assertEqual(inverse_bbox([60, 20, 30, 40], {"kind": "hflip"}, (100,100)), [10,20,30,40])
        self.assertEqual(inverse_bbox([35,45,30,40], {"kind": "offset", "offset": [25,25]}, (100,100)), [10,20,30,40])
        self.assertIsNone(inverse_bbox([200, 200, 10,10], {"kind":"hflip"}, (100,100)))
        self.assertEqual(swap_horizontal_words("Left hand, rightmost box"), "Right hand, leftmost box")
        self.assertEqual(len(build_views(Image.new("RGB", (100,100)), "left", 1.25)), 4)

    def test_connected_not_greedy_clustering(self):
        pool = [{"bbox":[x,0,10,10]} for x in (0,3,6)]
        self.assertEqual(len(connected_clusters(pool,.45)), 1)

    def test_ece_retains_votes_and_updates_current(self):
        row = {"pred_bbox": A, "candidates":[{"bbox":b} for b in (A,B,B,B)]}
        transformed = [{"bbox":B,"view":v} for v in ("hflip","pad_center")]
        pool, selected, info = consensus_update(row, transformed)
        self.assertEqual(len(pool), 2)
        self.assertEqual(selected["bbox"], B)
        self.assertAlmostEqual(info["winner_score"], 1.75)
        self.assertAlmostEqual(info["current_score"], 1.25)
        self.assertEqual(info["winner_view_support"], 2)

    def test_ece_one_view_cannot_override(self):
        row = {"pred_bbox": A, "candidates":[{"bbox":b} for b in (A,B,B,B)]}
        _, selected, info = consensus_update(row, [{"bbox":B,"view":"hflip"}]*3)
        self.assertIsNone(selected)
        self.assertEqual(info["winner_view_support"], 1)

    def test_duplicate_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            index_rows([{"dataset_index":1},{"dataset_index":"1"}])

    def test_crop_strict_gate_and_weak_decision_routes_to_dino(self):
        row, _ = case()
        row["crvg"].update(b0_count=3, b0_min_iou=0., ece_available=True)
        source = {"results":[row]}
        picks = {"meta":{"source_sha256":fingerprint(source)},
                 "picks":[{"dataset_index":0,"status":"scored","current_probability":.25,
                            "candidates":[{"bbox":B,"p_yes":.5}]}]}
        selected = apply_crop(source,picks,gate=.25)["results"][0]
        self.assertEqual(selected["pred_bbox"], A)
        self.assertTrue(dino_route(selected,.35))
        selected = apply_crop(source,picks,gate=.249)["results"][0]
        self.assertEqual(selected["pred_bbox"], B)

    def test_crop_matches_padt_minimum_context_window(self):
        image = Image.new("RGB", (100, 100), "white")
        crop = crop_with_context(image, [49, 49, 2, 2], context=0)
        self.assertEqual(crop.size, (56, 56))
        tiny = crop_with_context(image, [49, 49, 1, 2], context=0)
        self.assertEqual(tiny.size, image.size)

    def test_early_exit_restores_pre_ece_current(self):
        row, _ = case()
        row["pred_bbox"] = B
        row["crvg"].update(b0_count=3,b0_min_iou=.5,pre_ece_bbox=A,ece_available=True,ece_changed=True)
        source={"results":[row]}
        picks={"meta":{"source_sha256":fingerprint(source)},"picks":[]}
        selected=apply_crop(source,picks,gamma0=.5)["results"][0]
        self.assertEqual(selected["pred_bbox"], A)
        self.assertFalse(dino_route(selected,.35))
        self.assertFalse(selected["crvg"]["ece_changed"])

    def test_pairwise_order_alignment(self):
        a=aligned_scores(torch.tensor([1.,3.]),torch.tensor([3.,1.]))
        self.assertGreater(a["alternative_advantage"],0)
        self.assertTrue(a["permutation_agree"])
        b=aligned_scores(torch.tensor([1.,3.]),torch.tensor([1.,3.]))
        self.assertFalse(b["permutation_agree"])
        self.assertAlmostEqual(b["alternative_advantage"],0)
        self.assertAlmostEqual(a["alternative_probability"]+a["current_probability"],1,places=6)
        tied=aligned_scores(torch.tensor([1.,1.]),torch.tensor([1.,1.]))
        self.assertFalse(tied["permutation_agree"])

    def test_pairwise_montage_dimensions(self):
        image=render_pairwise_relation_montage(Image.new("RGB",(100,100)),[{"bbox":A},{"bbox":B}])
        self.assertEqual(image.size,(960,750))
        self.assertIsNotNone(image.getbbox())

    def test_dino_rebuilds_pool_at_append_threshold(self):
        row,_=case()
        near=[.3,0.,10.,10.]
        row["candidates"]=[{"bbox":A},{"bbox":near},{"bbox":C}]
        before=copy.deepcopy(row["candidates"])
        self.assertGreater(iou_xywh(A,near),.92)
        self.assertLess(iou_xywh(A,near),.95)
        pool=proposal_pool(row,[{"bbox":A,"score":.99},{"bbox":B,"score":.9}],max_challengers=1)
        self.assertEqual([c["bbox"] for c in pool],[A,C,B])
        self.assertEqual(pool[0]["source"],"current_system")
        self.assertEqual(row["candidates"],before)

    def test_dino_dual_phrase_merge_and_max_score(self):
        self.assertEqual(normalized_phrase("  The dog. "),"the dog")
        merged=merge_detections([{"bbox":[0.]*4,"score":.5}],[{"bbox":[1.]*4,"score":.9}])
        self.assertEqual([d["score"] for d in merged],[.9,.5])
        pool=[{"bbox":[0.,0.,5.,5.]},{"bbox":[40.,0.,5.,5.]}]
        target=[{"bbox":[0.,0.,6.,6.],"score":.8}]
        full=[{"bbox":[40.,0.,6.,6.],"score":.6}]
        attach_phrase_scores(pool,target,full)
        self.assertAlmostEqual(pool[0]["dino_phrase_score"],.8*25/36)
        self.assertAlmostEqual(pool[1]["dino_phrase_score"],.6*25/36)
        self.assertGreater(pool[1]["dino_full_score"],0.)
        self.assertEqual(pool[0]["dino_target_score"],pool[0]["dino_phrase_score"])

    def test_pairwise_skips_near_current_and_two_key_sort(self):
        row,_=case()
        near=[.5,0.,10.,10.]
        self.assertGreater(iou_xywh(A,near),.85)
        row["candidates"]=[{"bbox":A,"source":"current_system"},
                           {"bbox":C,"source":"sampled"},
                           {"bbox":B,"source":"grounding_dino_phrase","dino_phrase_score":.7},
                           {"bbox":[65.,65.,10.,10.],"source":"grounding_dino_phrase",
                            "dino_phrase_score":.7,"score":.9},
                           {"bbox":near,"source":"grounding_dino_phrase","dino_phrase_score":.99}]
        picked=select_challengers(row,{"bbox":list(A)})
        self.assertEqual([c["bbox"] for c in picked],[[65.,65.,10.,10.],B])
        self.assertEqual(select_challengers(row,{"bbox":list(A)},max_challengers=1)[0]["bbox"],
                         [65.,65.,10.,10.])

    def test_dino_route_requires_upstream_state(self):
        row,_=case()
        row.pop("crvg")
        with self.assertRaisesRegex(ValueError,"completed ECE/Qwen"):
            dino_route(row,.35)

    def test_consensus_feature_uses_existing_pool(self):
        row,record=case()
        row["crvg"].update(b1_count=5,b1_min_iou=.21)
        data,picks=evidence([row],[record])
        example=build_risk_examples(data,picks,"test")[0][0]
        values=dict(zip(FEATURE_NAMES,example["features"]))
        self.assertAlmostEqual(values["existing_pool_min_iou"],iou_xywh(A,C))
        self.assertAlmostEqual(values["existing_candidate_count_log"],math.log1p(2))

    def test_unknown_evidence_schema_rejected(self):
        row,record=case()
        data,picks=evidence([row],[record])
        data["meta"].pop("evidence_schema")
        with self.assertRaisesRegex(ValueError,"schema mismatch"):
            build_risk_examples(data,picks,"test")

    def test_anchor_features_use_declared_evidence(self):
        row,record=case()
        row.update(relation_plan={"requires_anchor":True},anchor_confidence=.7)
        data,picks=evidence([row],[record])
        example=build_risk_examples(data,picks,"test")[0][0]
        values=dict(zip(FEATURE_NAMES,example["features"]))
        self.assertEqual(values["requires_anchor"],1.)
        self.assertAlmostEqual(values["anchor_confidence"],.7)

    def test_three_actions_and_utility_boundary(self):
        row,record=case()
        data,picks=evidence([row],[record])
        examples=build_risk_examples(data,picks,"test")[0]
        for probabilities,expected in (([.9,.05,.05],"KEEP"),([.05,.05,.9],"ABSTAIN"),
                                       ([.05,.9,.05],"SWITCH")):
            run=evaluate_policy(examples,torch.tensor([probabilities]),gate=0.)
            self.assertEqual(run["decisions"][0]["action"],expected)
        probs=torch.tensor([[.05,.9,.05]])
        boundary=float(policy_score(probs[0]))
        self.assertTrue(evaluate_policy(examples,probs,gate=boundary)["decisions"][0]["switched"])
        self.assertFalse(evaluate_policy(examples,probs,gate=boundary+.001)["decisions"][0]["switched"])

    def test_highest_utility_must_itself_be_order_consistent(self):
        row,record=case()
        data,picks=evidence([row],[record])
        first=build_risk_examples(data,picks,"test")[0][0]
        second=copy.deepcopy(first)
        first["permutation_agree"]=False
        run=evaluate_policy([first,second],torch.tensor([[.01,.98,.01],[.1,.8,.1]]),gate=0.)
        self.assertEqual(run["decisions"][0]["action"],"ABSTAIN")
        self.assertEqual(run["switches"],0)

    def test_symmetric_controller_labels(self):
        from crvg.controller.features import decision_label
        for a,b,label in ((.2,.7,"switch"),(.7,.2,"keep"),(.51,.8,"switch"),
                          (.8,.51,"keep"),(.7,.72,"abstain"),(.1,.4,"abstain")):
            self.assertEqual(decision_label(a,b),label)

    def test_off_image_pairwise_crop_does_not_crash(self):
        for boxes in (([1000,1000,10,10], B), ([1000,1000,10,10], [1100,1100,10,10])):
            image=render_pairwise_relation_montage(Image.new("RGB",(100,100)),
                                                  [{"bbox":box} for box in boxes])
            self.assertEqual(image.size,(960,750))

    def test_train_markers_not_image_filename(self):
        self.assertTrue(is_train_row({"dataset":"refcoco+_train"}))
        self.assertFalse(is_train_row({"dataset":"refcoco_val","image":"COCO_train2014_1.jpg"}))
        self.assertFalse(is_train_row({"split":"testA","training_source":"refcoco_train"}))
        self.assertFalse(is_train_row({},{"source":"/data/train2014/1.json"}))

    def test_global_image_group(self):
        self.assertEqual(image_group({"image":"C:\\images\\1.jpg"}),image_group({"image":"/data/1.jpg"}))
        self.assertEqual(stable_fraction("1.jpg",42),stable_fraction("1.jpg",42))

    def test_features_keyed_and_gt_independent(self):
        row, record=case()
        data,picks=evidence([row],[record])
        examples,_=build_risk_examples(data,picks,"test",require_train=True)
        self.assertEqual(len(examples[0]["features"]),36)
        self.assertEqual(examples[0]["label"],"switch")
        changed=copy.deepcopy(data)
        changed["results"][0]["gt_bbox"]=A
        altered=copy.deepcopy(picks)
        altered["picks"][0]["challengers"][0]["iou"]=0.
        other,_=build_risk_examples(changed,altered,"test")
        self.assertEqual(examples[0]["features"],other[0]["features"])
        self.assertEqual(other[0]["label"],"keep")
        r2,p2=case(2)
        joined,ps=evidence([row,r2],[p2,record])
        self.assertEqual(len(build_risk_examples(joined,ps,"test")[0]),2)

    def test_training_rejects_official_split(self):
        row, record=case()
        row.update(dataset="refcoco_val",split="val")
        data,picks=evidence([row],[record])
        with self.assertRaisesRegex(ValueError,"TRAIN"):
            build_risk_examples(data,picks,"test",require_train=True)

    def test_checkpoint_scoring_and_merge(self):
        row,record=case()
        data,picks=evidence([row],[record])
        with tempfile.TemporaryDirectory() as tmp:
            controller(tmp)
            self.assertEqual(len(load_controller(tmp)[1]),len(FEATURE_NAMES))
            decisions,_,_=score_evidence(data,picks,tmp)
            selected=merge_decisions(data,data,decisions)
            self.assertEqual(selected["results"][0]["pred_bbox"],B)
            self.assertEqual(compare(data,selected)["net"],1)
            record["challengers"][0]["permutation_agree"]=False
            data,picks=evidence([row],[record])
            decisions,_,_=score_evidence(data,picks,tmp)
            self.assertFalse(decisions[0]["switched"])

    def test_unsafe_checkpoint_falls_back(self):
        row,record=case()
        data,picks=evidence([row],[record])
        with tempfile.TemporaryDirectory() as tmp:
            controller(tmp,safe=False)
            decisions,_,_=score_evidence(data,picks,tmp)
            self.assertFalse(decisions[0]["switched"])

    def test_replay_missing_evidence_and_changed_current_rejected(self):
        row,_=case()
        data={"results":[row]}
        with self.assertRaisesRegex(ValueError,"missing"):
            merge_decisions(data,{"results":[]},[])
        altered=copy.deepcopy(data)
        altered["results"][0]["pred_bbox"]=B
        with self.assertRaisesRegex(ValueError,"Current box"):
            merge_decisions(altered,data,[])

    def test_switch_counts_boxes_not_iou_change(self):
        row,_=case(role="neutral")
        selected=set_prediction(row,B,"test")
        report=compare([row],[selected])
        self.assertEqual(report["switches"],1)
        self.assertEqual(report["net"],0)

    def test_full_denominator_and_equal_weight_average(self):
        row,_=case()
        second,_=case(2)
        report=compare([row,second],[set_prediction(row,B,"test"),second])
        self.assertEqual(report["delta_pp"]["acc0.5"],50.)
        other=copy.deepcopy(report)
        other["delta_pp"]={k:0. for k in report["delta_pp"]}
        other["n"]=10000
        self.assertEqual(average([report,other])["acc0.5"],25.)
        with self.assertRaises(ValueError):
            compare([row,second],[row])

    def test_merge_rejects_cache_wrong_source(self):
        row,_=case()
        with self.assertRaises(ValueError):
            merge({"results":[row]},{"meta":{"source_sha256":"bad"},"records":{}})


if __name__ == "__main__":
    unittest.main()
