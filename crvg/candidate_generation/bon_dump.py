"""Shared backbone candidate generation. No ground truth enters model prompts."""
import argparse
import re
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from crvg.utils.bbox import norm1000_to_pixel_xywh, xyxy_to_xywh, iou_xywh, valid_box
from crvg.utils.data import normalize_sample, read_json, write_json, fingerprint, index_rows

INTERNVL_PROMPT = "Please provide the bounding box coordinate of the region this sentence describes: <ref>{expr}</ref>."
VLMR1_PROMPT = ('Please provide the bounding box coordinate of the region this sentence describes: {expr}. '
               'First output the thinking process in <think> </think> tags and then output the final answer '
               'in <answer> </answer> tags. Output the final answer in JSON format.')


def extract_bbox(text):
    answer = re.search(r"<answer>(.*?)</answer>", text or "", re.S)
    text = answer.group(1) if answer else (text or "")
    number = r"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    match = re.search(r"\[\s*\[?\s*" + r"\s*,\s*".join([number] * 4) + r"\s*\]?", text)
    return list(map(float, match.groups())) if match else [0., 0., 0., 0.]


class BackboneEngine:
    """Greedy and stochastic decoding for a grounding backbone via Transformers."""

    def __init__(self, model_path, backbone, max_tokens=256, seed=42):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        torch.manual_seed(seed)
        self.kind = backbone
        self.max_tokens = max_tokens
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.processor.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
            device_map={"": 0}).eval()

    def prompt(self, expression):
        template = INTERNVL_PROMPT if self.kind == "internvl" else VLMR1_PROMPT
        messages = [{"role": "user", "content": [{"type": "image"},
                                                 {"type": "text", "text": template.format(expr=expression)}]}]
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    @torch.inference_mode()
    def predict(self, images, expressions, n=1, temperature=0.):
        prompts = [self.prompt(expression) for expression in expressions]
        inputs = self.processor(text=prompts, images=list(images), return_tensors="pt").to(self.model.device)
        kwargs = {"do_sample": temperature > 0, "max_new_tokens": self.max_tokens,
                  "num_return_sequences": n}
        if temperature > 0:
            kwargs["temperature"] = temperature
        output = self.model.generate(**inputs, **kwargs)
        generated = output[:, inputs["input_ids"].shape[1]:]
        texts = self.processor.batch_decode(generated, skip_special_tokens=True)
        if len(texts) != len(images) * n:
            raise RuntimeError("Backbone output count mismatch")
        batches = []
        for image, index in zip(images, range(len(images))):
            entries = []
            for completion in texts[index * n:(index + 1) * n]:
                coords = extract_bbox(completion)
                box = (norm1000_to_pixel_xywh(coords, *image.size) if self.kind == "internvl"
                       else xyxy_to_xywh(coords))
                valid = valid_box(box)
                entries.append({"bbox": box if valid else [0., 0., 0., 0.],
                                "valid": valid, "score": 0., "response": completion})
            batches.append(entries)
        return batches


def add_engine_args(parser):
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)


def engine_from_args(args, kind):
    return BackboneEngine(args.model_path, kind, args.max_tokens, args.seed)


def main(backbone):
    parser = argparse.ArgumentParser(description=f"{backbone} greedy + sampled candidates")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bon-n", type=int, default=8, help="Additional stochastic completions")
    parser.add_argument("--bon-temperature", type=float, default=1.3)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=-1)
    add_engine_args(parser)
    args = parser.parse_args()
    if args.bon_n < 0 or args.chunk_size < 1:
        parser.error("bon-n must be nonnegative and chunk-size positive")
    engine = engine_from_args(args, backbone)
    for dataset in args.datasets:
        raw = read_json(Path(args.data_root) / f"{dataset}.json")
        raw = raw["results"] if isinstance(raw, dict) else raw
        if args.max_samples > 0:
            raw = raw[:args.max_samples]
        samples = [normalize_sample(row, i, args.image_root) for i, row in enumerate(raw)]
        index_rows(samples)
        rows = []
        for start in tqdm(range(0, len(samples), args.chunk_size), desc=dataset):
            chunk = samples[start:start + args.chunk_size]
            images = []
            for sample in chunk:
                with Image.open(sample["image_path"]) as image:
                    images.append(image.convert("RGB"))
            exprs = [s["expr"] for s in chunk]
            greedy = engine.predict(images, exprs)
            sampled = engine.predict(images, exprs, args.bon_n, args.bon_temperature) if args.bon_n else [[] for _ in chunk]
            for sample, image, first, extras in zip(chunk, images, greedy, sampled):
                candidates = first + extras
                gt = sample["gt_bbox_xywh"]
                for j, candidate in enumerate(candidates):
                    candidate["source"] = "greedy" if j == 0 else "sampled"
                    candidate["iou"] = iou_xywh(candidate["bbox"], gt) if gt else None
                rows.append({"dataset_index": sample["dataset_index"], "dataset": dataset,
                             "split": "train" if dataset.endswith("_train") else dataset.rsplit("_", 1)[-1],
                             "image": sample["image_path"], "image_size": list(image.size),
                             "text": sample["expr"], "gt_bbox": gt, "pred_bbox": first[0]["bbox"],
                             "iou": first[0]["iou"], "iou_greedy": first[0]["iou"],
                             "candidates": candidates, "num_candidates": len(candidates)})
        out = Path(args.output_dir) / f"rec_results_{dataset}.json"
        write_json(out, {"meta": {"backbone": backbone, "model": args.model_path, "dataset": dataset,
                                  "annotation_sha256": fingerprint(raw), "options": vars(args),
                                  "complete": True, "total": len(rows)}, "results": rows})
        print(f"Saved {len(rows)} rows: {out}")
