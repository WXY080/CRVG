"""Frozen full-image pairwise 1/2 next-token logits with reversed-order alignment."""
import argparse

import torch
from PIL import Image
from tqdm import tqdm

from crvg.utils.data import read_json, write_json, results, current_candidate, row_image, extract_expression, fingerprint
from crvg.verification.qwen import add_qwen_args, load_qwen, token_variants, last_logits
from crvg.verification.render import render_pairwise_relation_montage, build_pairwise_relation_prompt


def aligned_scores(forward, reverse):
    reverse = reverse.flip(-1)
    mean_logits = .5 * (forward + reverse)
    probability = mean_logits.softmax(-1)
    fmargin, rmargin = float(forward[1]-forward[0]), float(reverse[1]-reverse[0])
    return {"current_probability": float(probability[0]), "alternative_probability": float(probability[1]),
            "alternative_advantage": float(probability[1]-probability[0]),
            "forward_alternative_margin": fmargin, "reverse_alternative_margin": rmargin,
            "mean_alternative_margin": .5*(fmargin+rmargin),
            "permutation_agree": (fmargin > 0 and rmargin > 0) or (fmargin < 0 and rmargin < 0),
            "permutation_logit_gap": abs(fmargin-rmargin)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence")
    parser.add_argument("--save", required=True)
    parser.add_argument("--max-challengers", type=int, default=8)
    parser.add_argument("--context-scale", type=float, default=1.8)
    add_qwen_args(parser)
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_challengers < 1:
        parser.error("batch-size and max-challengers must be positive")
    data = read_json(args.evidence)
    model, processor = load_qwen(args.verifier_model, args.device_map, args.attn_implementation)
    labels = [token_variants(processor.tokenizer, [str(i)]) for i in (1, 2)]
    records = []
    for row in tqdm(results(data), desc="Qwen pairwise"):
        current = current_candidate(row)
        challengers = sorted([c for c in row["candidates"] if c.get("source") == "grounding_dino_phrase"],
                              key=lambda c: c.get("dino_phrase_score", c.get("score", 0)), reverse=True)[:args.max_challengers]
        record = {"dataset_index": row["dataset_index"], "current": current, "challengers": [],
                  "status": "scored" if challengers else "no_dino_challenger"}
        if challengers:
            with Image.open(row_image(row, args.image_root)) as raw:
                image = raw.convert("RGB")
            for start in range(0, len(challengers), args.batch_size):
                batch = challengers[start:start+args.batch_size]
                views = []
                for challenger in batch:
                    views.extend([render_pairwise_relation_montage(image, pair, context_fraction=args.context_scale)
                                  for pair in ([current, challenger], [challenger, current])])
                logits = last_logits(model, processor, views, [build_pairwise_relation_prompt(extract_expression(row))]*len(views))
                scores = torch.stack([logits[:, ids].max(-1).values for ids in labels], -1)
                for i, challenger in enumerate(batch):
                    record["challengers"].append({**challenger, **aligned_scores(scores[2*i], scores[2*i+1])})
        records.append(record)
    write_json(args.save, {"meta": {"source_sha256": fingerprint(data), "complete": True,
                                    "model": args.verifier_model, "scoring_backend": "base_qwen_next_token",
                                    "prompt": build_pairwise_relation_prompt("{expression}"),
                                    "options": vars(args),
                                    "label_token_ids": labels, "context_scale": args.context_scale},
                           "picks": records})


if __name__ == "__main__":
    main()
