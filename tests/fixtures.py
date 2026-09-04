"""Synthetic records for contract tests, never benchmark evidence."""
import copy
from pathlib import Path
import torch

from crvg.controller.features import FEATURE_NAMES
from crvg.controller.model import DinoRiskMLP
from crvg.utils.data import fingerprint, write_json
from crvg.settings import DINO_EVIDENCE_SCHEMA

A = [0., 0., 10., 10.]
B = [30., 0., 10., 10.]
C = [70., 70., 10., 10.]


def case(i=0, domain="refcoco", role="rescue"):
    gt = B if role == "rescue" else A if role == "protect" else C
    row = {"dataset_index": i, "dataset": domain+"_train", "split": "train",
           "image": f"COCO_train2014_{i:012d}.jpg", "image_size": [100, 100],
           "text": "object on the right", "gt_bbox": list(gt), "pred_bbox": list(A),
           "crvg": {"cascade_entered": True},
           "candidates": [{"bbox": list(A), "source": "current_system"},
                          {"bbox": list(C), "source": "sampled"},
                          {"bbox": list(B), "source": "grounding_dino_phrase", "dino_phrase_score": .7}]}
    alt = copy.deepcopy(row["candidates"][-1])
    alt.update(iou=1. if gt == B else 0., current_probability=.1, alternative_probability=.9,
               alternative_advantage=.8, permutation_agree=True, permutation_logit_gap=.05)
    record = {"dataset_index": i, "current": {"bbox": list(A)}, "challengers": [alt], "status": "scored"}
    return row, record


def evidence(rows, records):
    data = {"meta": {"synthetic": True, "evidence_schema": DINO_EVIDENCE_SCHEMA}, "results": rows}
    picks = {"meta": {"source_sha256": fingerprint(data), "scoring_backend": "base_qwen_next_token"},
             "picks": records}
    return data, picks


def controller(directory, safe=True):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model = DinoRiskMLP(len(FEATURE_NAMES), 64, .1)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.network[-1].bias.copy_(torch.tensor([-10., 10., -10.]))
    torch.save(model.state_dict(), directory/"risk_controller.pt")
    config = {"scoring_backend": "base_qwen_next_token",
              "feature_names": FEATURE_NAMES,
              "normalization": {"mean": [0.]*len(FEATURE_NAMES), "scale": [1.]*len(FEATURE_NAMES)},
              "model": {"input_dim": len(FEATURE_NAMES), "hidden_dim": 64, "dropout": .1},
              "selected_policy": {"gate": 0., "damage_cost": 2., "abstain_cost": .25,
                                  "require_permutation_agree": True, "safe": safe}, "passed": safe}
    write_json(directory/"risk_controller_config.json", config)
    return config
