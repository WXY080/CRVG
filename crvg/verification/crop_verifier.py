"""Score current and expanded candidates with a frozen Yes/No crop prompt."""
import argparse

import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from crvg.utils.bbox import min_pairwise_iou, valid_box
from crvg.utils.data import read_json, write_json, results, current_bbox, row_image, fingerprint, extract_expression
from crvg.verification.qwen import load_qwen, token_variants, last_logits, add_qwen_args

PROMPT = 'Look at the red box in this image. Does the red box correctly mark "{expr}"? Answer with only Yes or No.'


def crop_with_context(image, box, context=.3):
    x, y, w, h = box
    if not valid_box(box):
        return image.copy()
    left, top = max(0, int(x - w * context)), max(0, int(y - h * context))
    right, bottom = min(image.width, int(x + w * (1 + context))), min(image.height, int(y + h * (1 + context)))
    if right <= left or bottom <= top:
        return image.copy()
    crop = image.crop((left, top, right, bottom))
    ImageDraw.Draw(crop).rectangle((x-left, y-top, x+w-left, y+h-top),
                                   outline=(255, 0, 0), width=max(2, int(min(crop.size)*.012)))
    return crop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--save", required=True)
    parser.add_argument("--agree-skip-iou", type=float, default=.5)
    parser.add_argument("--context-frac", type=float, default=.3)
    add_qwen_args(parser)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    source = read_json(args.input)
    model, processor = load_qwen(args.verifier_model, args.device_map, args.attn_implementation)
    yes = token_variants(processor.tokenizer, ["Yes", "yes", "YES"])
    no = token_variants(processor.tokenizer, ["No", "no", "NO"])
    if set(yes) & set(no):
        raise ValueError("Overlapping Yes/No tokens")
    records = []
    for row in tqdm(results(source), desc="Qwen crops"):
        boxes = [c["bbox"] for c in row["candidates"]]
        gamma = row.get("crvg", {}).get("b0_min_iou", min_pairwise_iou(boxes))
        count = row.get("crvg", {}).get("b0_count", len(boxes))
        record = {"dataset_index": row["dataset_index"], "current_bbox": current_bbox(row),
                  "status": "skipped", "candidates": []}
        if count >= 2 and gamma < args.agree_skip_iou:
            with Image.open(row_image(row, args.image_root)) as raw:
                image = raw.convert("RGB")
            scoring_boxes = [current_bbox(row)] + boxes
            scores = []
            for start in range(0, len(scoring_boxes), args.batch_size):
                batch = scoring_boxes[start:start + args.batch_size]
                crops = [crop_with_context(image, box, args.context_frac) for box in batch]
                logits = last_logits(model, processor, crops, [PROMPT.format(expr=extract_expression(row))]*len(batch))
                # Equivalent to summing full-vocabulary probabilities and renormalizing over Yes/No.
                labels = torch.stack((torch.logsumexp(logits[:, no], -1), torch.logsumexp(logits[:, yes], -1)), -1)
                scores.extend(labels.softmax(-1)[:, 1].cpu().tolist())
            record.update(status="scored", current_probability=scores[0],
                          candidates=[{"bbox": b, "p_yes": p} for b, p in zip(boxes, scores[1:])])
        records.append(record)
    write_json(args.save, {"meta": {"source_sha256": fingerprint(source), "complete": True,
                                    "model": args.verifier_model, "gamma0": args.agree_skip_iou,
                                    "prompt": PROMPT, "yes_token_ids": yes, "no_token_ids": no,
                                    "options": vars(args),
                                    "scoring_backend": "base_qwen_next_token"}, "picks": records})


if __name__ == "__main__":
    main()
