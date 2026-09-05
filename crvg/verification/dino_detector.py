"""Target-phrase Grounding-DINO proposals; spatial evidence is not a hard selector."""
import argparse
import copy

import torch
from PIL import Image
from tqdm import tqdm

from crvg.utils.bbox import append_distinct, iou_xywh, min_pairwise_iou, xyxy_to_xywh
from crvg.utils.data import (read_json, write_json, results, row_image, current_bbox, current_iou,
                             extract_expression, fingerprint)
from crvg.verification.spatial import parse_spatial_plan, phrase_match_score, spatial_score
from crvg.settings import DINO_EVIDENCE_SCHEMA


def dino_route(row, threshold):
    state = row.get("crvg", {})
    if "cascade_entered" not in state:
        raise ValueError("DINO routing requires the completed ECE/Qwen stage")
    return (state["cascade_entered"] and len(row.get("candidates", [])) >= 2
            and min_pairwise_iou([c["bbox"] for c in row["candidates"]]) < threshold)


def normalized_phrase(value):
    return " ".join(str(value or "").strip().rstrip(".").lower().split())


def proposal_pool(row, detections, max_challengers=8, duplicate_iou=.92):
    """Rebuild the pool at the append threshold with the current box first."""
    pool = [{"bbox": current_bbox(row), "iou": current_iou(row), "source": "current_system"}]
    for candidate in row.get("candidates", []):
        item = copy.deepcopy(candidate)
        item.setdefault("source", "existing_pool")
        append_distinct(pool, item, duplicate_iou)
    added = 0
    for detection in detections:
        candidate = {**detection, "source": "grounding_dino_phrase",
                     "iou": iou_xywh(detection["bbox"], row["gt_bbox"]) if row.get("gt_bbox") else None}
        if append_distinct(pool, candidate, duplicate_iou):
            added += 1
            if added >= max_challengers:
                break
    return pool


def merge_detections(target, full):
    return sorted([*target, *full], key=lambda item: item["score"], reverse=True)


def attach_phrase_scores(pool, target_detections, full_detections):
    for candidate in pool:
        target_score = phrase_match_score(candidate["bbox"], target_detections)
        full_score = phrase_match_score(candidate["bbox"], full_detections)
        candidate["dino_target_score"] = target_score
        candidate["dino_full_score"] = full_score
        candidate["dino_phrase_score"] = max(target_score, full_score)
    return pool


@torch.inference_mode()
def detect(model, processor, images, phrases, device, threshold, text_threshold):
    if not images:
        return []
    inputs = processor(images=images, text=[p.strip().rstrip(".")+"." for p in phrases],
                       padding=True, return_tensors="pt").to(device)
    output = model(**inputs)
    processed = processor.post_process_grounded_object_detection(
        output, input_ids=inputs.input_ids, threshold=threshold, text_threshold=text_threshold,
        target_sizes=[image.size[::-1] for image in images])
    return [sorted([{"bbox": xyxy_to_xywh(b.tolist()), "score": float(s)}
                    for b, s in zip(item["boxes"].cpu(), item["scores"].cpu())],
                   key=lambda c: c["score"], reverse=True) for item in processed]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--grounding-model", required=True)
    parser.add_argument("--save", required=True)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--agree-skip-iou", type=float, default=.35)
    parser.add_argument("--box-threshold", type=float, default=.2)
    parser.add_argument("--text-threshold", type=float, default=.2)
    parser.add_argument("--max-tool-candidates", type=int, default=8)
    parser.add_argument("--max-detections", type=int, default=12)
    parser.add_argument("--duplicate-iou", type=float, default=.92)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_tool_candidates < 1 or args.max_detections < 1:
        parser.error("batch-size and max-tool-candidates must be positive")
    data = read_json(args.input)
    rows = results(data)
    queued = [r for r in rows if dino_route(r, args.agree_skip_iou)]
    output_rows, model, processor = [], None, None
    if queued:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        processor = AutoProcessor.from_pretrained(args.grounding_model)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(args.grounding_model).to(args.device).eval()
        model.requires_grad_(False)
    for start in tqdm(range(0, len(queued), args.batch_size), desc="Grounding-DINO"):
        batch = queued[start:start+args.batch_size]
        images, plans, phrases, exprs = [], [], [], []
        for row in batch:
            with Image.open(row_image(row, args.image_root)) as raw:
                images.append(raw.convert("RGB"))
            expr = extract_expression(row)
            exprs.append(expr)
            plan = parse_spatial_plan(expr)
            plans.append(plan)
            phrases.append(plan["target_phrase"] if plan and plan["target_phrase"] else expr)
        target_sets = [found[:args.max_detections] for found in
                       detect(model, processor, images, phrases, args.device,
                              args.box_threshold, args.text_threshold)]
        full_positions = [i for i in range(len(batch))
                          if normalized_phrase(exprs[i]) != normalized_phrase(phrases[i])]
        full_sets = [[] for _ in batch]
        if full_positions:
            raw_full = detect(model, processor, [images[i] for i in full_positions],
                              [exprs[i] for i in full_positions], args.device,
                              args.box_threshold, args.text_threshold)
            for i, found in zip(full_positions, raw_full):
                full_sets[i] = found[:args.max_detections]
        merged = [merge_detections(target, full)
                  for target, full in zip(target_sets, full_sets)]
        anchor_idx = [i for i, plan in enumerate(plans) if plan and plan["requires_anchor"]]
        anchors = detect(model, processor, [images[i] for i in anchor_idx],
                         [plans[i]["anchor_phrase"] for i in anchor_idx], args.device, args.box_threshold, args.text_threshold)
        by_anchor = dict(zip(anchor_idx, [found[:args.max_detections] for found in anchors]))
        for i, (row, image) in enumerate(zip(batch, images)):
            pool = attach_phrase_scores(
                proposal_pool(row, merged[i], args.max_tool_candidates, args.duplicate_iou),
                target_sets[i], full_sets[i])
            for candidate in pool:
                candidate["spatial_skill_score"] = spatial_score(candidate["bbox"], by_anchor.get(i, []), plans[i], image.size)
            output_rows.append({**row, "image_size": list(image.size), "candidates": pool,
                                "relation_plan": plans[i], "target_detections": target_sets[i],
                                "full_detections": full_sets[i],
                                "anchor_confidence": max((c["score"] for c in by_anchor.get(i, [])), default=0.),
                                "target_phrase": phrases[i]})
    write_json(args.save, {"meta": {**data.get("meta", {}), "source_sha256": fingerprint(data),
                                    "evidence_schema": DINO_EVIDENCE_SCHEMA,
                                    "complete": True, "gamma1": args.agree_skip_iou,
                                    "total_samples": len(rows), "processed": len(output_rows),
                                    "options": vars(args)}, "results": output_rows})


if __name__ == "__main__":
    main()
