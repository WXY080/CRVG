"""Frozen Qwen3-VL next-token scoring for Eqs. (5) and (6)."""
import torch


def load_qwen(model_path, device_map="auto", attn_implementation="sdpa"):
    from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
    config = AutoConfig.from_pretrained(model_path)
    if config.model_type != "qwen3_vl":
        raise ValueError("The manuscript verifier requires Qwen3-VL-Instruct")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=dtype, device_map=device_map, attn_implementation=attn_implementation).eval()
    model.requires_grad_(False)
    return model, AutoProcessor.from_pretrained(model_path)


def token_variants(tokenizer, words):
    tokens = set()
    for word in words:
        for variant in (word, " " + word):
            encoded = tokenizer.encode(variant, add_special_tokens=False)
            if len(encoded) == 1:
                tokens.add(encoded[0])
    if not tokens:
        raise ValueError(f"No single-token label variants: {words}")
    return sorted(tokens)


@torch.inference_mode()
def last_logits(model, processor, images, prompts):
    if not images or len(images) != len(prompts):
        raise ValueError("Provide one prompt per image in a nonempty batch")
    messages = [[{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        for image, prompt in zip(images, prompts)]
    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True,
                       padding_side="left", add_special_tokens=False).to(model.device)
    inputs.pop("token_type_ids", None)
    output = model(**inputs, use_cache=False, logits_to_keep=1)
    return output.logits[:, -1, :].float()


def add_qwen_args(parser):
    parser.add_argument("--verifier-model", required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", choices=("sdpa", "flash_attention_2"), default="sdpa")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-root", default=None)
