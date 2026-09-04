"""Packaged defaults mirrored by configs/default.yaml."""
DINO_EVIDENCE_SCHEMA = "crvg.dino.v1"

DEFAULTS = {
    "gamma0": .50, "gate": .30, "gamma1": .35,
    "bon_n": 8, "bon_temperature": 1.3, "pad_factor": 1.25, "dedup_iou": .95,
    "ece_cluster_iou": .45, "ece_min_support": 2, "ece_score_gate": .25,
    "dino_box_threshold": .20, "dino_text_threshold": .20,
    "dino_max_detections": 12, "dino_max_challengers": 8,
}
