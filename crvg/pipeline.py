"""Portable, fail-fast pipeline with content-addressed resume manifests."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys

import yaml

from crvg.utils.data import read_json, write_json, fingerprint
from crvg.settings import DEFAULTS


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_stage(module, argv, inputs, outputs, python, rebuild=False, dry_run=False):
    command = [python, "-m", module, *map(str, argv)]
    if dry_run:
        print(subprocess.list2cmdline(command))
        return
    code_root = Path(__file__).resolve().parent.parent
    code = {str(p.relative_to(code_root)): file_hash(p)
            for folder in ("crvg", "tools", "analysis")
            for p in (code_root / folder).rglob("*.py")}
    signature = fingerprint({"command": command, "inputs": {str(p): file_hash(p) for p in inputs}, "code": code})
    manifest = Path(str(outputs[0]) + ".manifest.json")
    if not rebuild and manifest.exists() and all(Path(p).is_file() for p in outputs):
        saved = read_json(manifest)
        if saved.get("signature") == signature and saved.get("outputs") == {str(p): file_hash(p) for p in outputs}:
            print(f"[reuse] {module}", flush=True)
            return
    print(f"[run] {module}", flush=True)
    subprocess.run(command, check=True)
    for path in outputs:
        payload = read_json(path)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid output: {path}")
    write_json(manifest, {"signature": signature, "outputs": {str(p): file_hash(p) for p in outputs}})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=("internvl", "vlmr1", "padt"), required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--data-root")
    parser.add_argument("--candidate-cache-dir", help="Canonical BoN + ECE artifacts; required for the external PaDT generator")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--qwen-model", required=True)
    parser.add_argument("--dino-model", required=True)
    parser.add_argument("--controller")
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--datasets", nargs="+", default=["refcoco_val"])
    parser.add_argument("--config", help="YAML overrides; omitted uses packaged paper defaults")
    parser.add_argument("--gamma0", type=float)
    parser.add_argument("--gate", type=float)
    parser.add_argument("--gamma1", type=float)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--backbone-batch-size", type=int, default=1)
    parser.add_argument("--backbone-max-tokens", type=int, default=256)
    parser.add_argument("--backbone-seed", type=int, default=42)
    parser.add_argument("--internvl-max-tiles", type=int, default=12)
    parser.add_argument("--backbone-python", default=sys.executable)
    parser.add_argument("--verifier-python", default=sys.executable)
    parser.add_argument("--stop-after", choices=("bon", "ece", "qwen", "dino", "pairwise", "final"), default="final")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.batch_size, args.backbone_batch_size, args.backbone_max_tokens, args.internvl_max_tiles) < 1:
        parser.error("Batch sizes, token limit and tile limit must be positive")
    if not args.candidate_cache_dir and (args.backbone == "padt" or not args.model_path or not args.data_root):
        parser.error("Use --candidate-cache-dir for PaDT, or supply --model-path and --data-root for backbone generation")
    if args.stop_after == "final" and not args.controller:
        parser.error("--controller is required for final selection")
    config = dict(DEFAULTS)
    if args.config:
        with open(args.config, encoding="utf-8") as handle:
            config.update(yaml.safe_load(handle) or {})
    g0 = config["gamma0"] if args.gamma0 is None else args.gamma0
    gate = config["gate"] if args.gate is None else args.gate
    g1 = config["gamma1"] if args.gamma1 is None else args.gamma1
    if not 0 < g1 <= g0 <= 1.01 or not 0 <= gate <= 1:
        parser.error("Require 0 < gamma1 <= gamma0 <= 1.01 and gate in [0,1]")
    log = Path(args.log_dir).resolve()
    log.mkdir(parents=True, exist_ok=True)
    engine_args = ["--backbone-batch-size", args.backbone_batch_size,
                   "--max-tokens", args.backbone_max_tokens, "--seed", args.backbone_seed,
                   "--internvl-max-tiles", args.internvl_max_tiles]
    def stage(module, argv, inputs, outputs, interpreter=sys.executable):
        run_stage(module, argv, inputs, outputs, interpreter, args.rebuild, args.dry_run)
    for dataset in args.datasets:
        base = log / f"rec_results_{dataset}.json"
        ece = log / f"ece_{dataset}.json"
        if args.candidate_cache_dir:
            cached_base = Path(args.candidate_cache_dir).resolve() / base.name
            cached_ece = Path(args.candidate_cache_dir).resolve() / ece.name
            stage("tools.import_candidates", ["--base", cached_base, "--ece", cached_ece,
                                             "--out-base", base, "--out-ece", ece],
                  [cached_base, cached_ece], [base, ece])
        expanded = log / f"rec_results_{dataset}_expanded.json"
        crops = log / f"crop_picks_{dataset}.json"
        qwen = log / f"rec_results_{dataset}_qwen.json"
        dino = log / f"dino_evidence_{dataset}.json"
        pairs = log / f"pairwise_picks_{dataset}.json"
        final = log / f"rec_results_{dataset}_crvg.json"
        risk = log / f"risk_decisions_{dataset}.json"
        if not args.candidate_cache_dir:
            stage(f"crvg.candidate_generation.{args.backbone}_bon",
              ["--model-path", args.model_path, "--data-root", args.data_root, "--image-root", args.image_root,
               "--datasets", dataset, "--output-dir", log, "--bon-n", config["bon_n"],
               "--bon-temperature", config["bon_temperature"],
               "--chunk-size", args.backbone_batch_size, *engine_args],
              [Path(args.data_root)/f"{dataset}.json"], [base], args.backbone_python)
        if args.stop_after == "bon": continue
        if not args.candidate_cache_dir:
            stage("crvg.candidate_generation.ece",
              ["--model-path", args.model_path, "--backbone", args.backbone, "--input", base, "--save", ece,
               "--image-root", args.image_root, "--agree-skip-iou", g0,
               "--pad-factor", config["pad_factor"], *engine_args],
              [base], [ece], args.backbone_python)
        stage("tools.merge_ece", ["--base", base, "--ece", ece, "--output", expanded,
                                 "--dedup-iou", config["dedup_iou"], "--cluster-iou", config["ece_cluster_iou"],
                                 "--min-support", config["ece_min_support"], "--score-gate", config["ece_score_gate"]],
              [base, ece], [expanded])
        if args.stop_after == "ece": continue
        model_args = ["--verifier-model", args.qwen_model, "--image-root", args.image_root, "--batch-size", args.batch_size]
        stage("crvg.verification.crop_verifier", [expanded, "--save", crops, "--agree-skip-iou", g0, *model_args],
              [expanded], [crops], args.verifier_python)
        stage("crvg.verification.apply_gate", [expanded, "--picks", crops, "--gate", gate, "--gamma0", g0, "--out", qwen],
              [expanded, crops], [qwen])
        if args.stop_after == "qwen": continue
        stage("crvg.verification.dino_detector",
              [qwen, "--grounding-model", args.dino_model, "--save", dino, "--image-root", args.image_root,
               "--agree-skip-iou", g1, "--batch-size", args.batch_size,
               "--box-threshold", config["dino_box_threshold"], "--text-threshold", config["dino_text_threshold"],
               "--max-detections", config["dino_max_detections"],
               "--max-tool-candidates", config["dino_max_challengers"]], [qwen], [dino], args.verifier_python)
        if args.stop_after == "dino": continue
        stage("crvg.verification.pairwise", [dino, "--save", pairs,
              "--max-challengers", config["dino_max_challengers"], *model_args], [dino], [pairs], args.verifier_python)
        if args.stop_after == "pairwise": continue
        stage("crvg.controller.apply",
              [dino, "--picks", pairs, "--controller", args.controller, "--source-json", qwen,
               "--gamma1", g1, "--selected-out", final, "--out", risk],
              [qwen, dino, pairs, Path(args.controller)/"risk_controller_config.json",
               Path(args.controller)/"risk_controller.pt"], [final, risk])
    print(f"Pipeline complete: {log}")


if __name__ == "__main__":
    main()
