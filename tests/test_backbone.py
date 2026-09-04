"""Backbone loading, variable-length batches, and multi-view generation contracts."""
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image
import torch

from crvg.candidate_generation.bon_dump import BackboneEngine
from crvg.candidate_generation.ece import build_views
from crvg.candidate_generation.internvl import image_tiles, pixel_batch
from tests.test_workflow import cli


class TensorBatch(dict):
    def to(self, device):
        return TensorBatch({key: value.to(device) for key, value in self.items()})


class Processor:
    def __init__(self):
        self.tokenizer = SimpleNamespace(padding_side="right")
        self.batches = []

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"][1]["text"]

    def __call__(self, text, images, padding=False, return_tensors=None):
        if not padding or self.tokenizer.padding_side != "left":
            raise ValueError("Variable-length decoder batches require left padding")
        ids = [torch.tensor([9] * (len(prompt) % 5 + image.width // 28 + 2) +
                            [image.getpixel((0, 0))[0]]) for prompt, image in zip(text, images)]
        width = max(len(row) for row in ids)
        input_ids = torch.stack([torch.nn.functional.pad(row, (width - len(row), 0)) for row in ids])
        batch = TensorBatch(input_ids=input_ids, attention_mask=(input_ids != 0).long())
        self.batches.append(batch)
        return batch

    def batch_decode(self, rows, **kwargs):
        if rows.shape[1] != 1:
            raise ValueError("Prompt tokens were not removed before decoding")
        return [f"<answer>[{int(row[0])}, 0, {int(row[0]) + 10}, 10]</answer>" for row in rows]


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(force_image_size=8, vision_config=SimpleNamespace(image_size=8),
                                      dynamic_image_size=True, use_thumbnail=True)
        self.generation_config = SimpleNamespace(bos_token_id=1, eos_token_id=2, pad_token_id=0,
                                                 top_p=.2, num_beams=5, forced_bos_token_id=99)
        self.calls = []

    @property
    def device(self):
        return self.weight.device

    @property
    def dtype(self):
        return self.weight.dtype

    @property
    def language_model(self):
        return self

    def generate(self, input_ids, attention_mask, **kwargs):
        self.calls.append(kwargs)
        n = kwargs["num_return_sequences"]
        expanded = input_ids.repeat_interleave(n, dim=0)
        completion = expanded[:, -1] * 10 + torch.arange(n).repeat(len(input_ids))
        return torch.cat([expanded, completion[:, None]], dim=1)

    def batch_chat(self, tokenizer, pixels, questions, generation_config, num_patches_list):
        self.calls.append((pixels, questions, generation_config, num_patches_list))
        return [f"[{100 + j * 10}, 200, 300, 400]" for _ in questions
                for j in range(generation_config["num_return_sequences"])]


def factories(version="4.57.6"):
    model, processor = Model(), Processor()
    module = SimpleNamespace(__version__=version, GenerationConfig=SimpleNamespace, AutoProcessor=MagicMock(),
                             AutoModelForImageTextToText=MagicMock(),
                             AutoModel=MagicMock(), AutoTokenizer=MagicMock())
    module.AutoProcessor.from_pretrained.return_value = processor
    module.AutoTokenizer.from_pretrained.return_value = processor.tokenizer
    module.AutoModelForImageTextToText.from_pretrained.return_value = model
    module.AutoModel.from_pretrained.return_value = model
    return module, model, processor


class BackboneTests(unittest.TestCase):
    def test_qwen_left_padding_and_sample_grouping(self):
        module, model, processor = factories()
        with patch.dict(sys.modules, {"transformers": module}):
            engine = BackboneEngine("synthetic", "vlmr1", batch_size=2)
        images = [Image.new("RGB", (28, 28), (1, 0, 0)),
                  Image.new("RGB", (84, 56), (2, 0, 0))]
        output = engine.predict(images, ["short", "a longer expression"], n=3, temperature=1.3)
        self.assertEqual([[entry["bbox"][0] for entry in row] for row in output], [[10, 11, 12], [20, 21, 22]])
        self.assertEqual(len(processor.batches), 1)
        batch = processor.batches[0]
        self.assertTrue((batch["attention_mask"][:, -1] == 1).all())
        self.assertTrue((batch["attention_mask"][:, 0] == 0).any())
        self.assertEqual(model.calls[0]["top_p"], 1.)
        self.assertEqual(model.calls[0]["top_k"], 0)
        config = vars(model.calls[0]["generation_config"])
        self.assertEqual(config, {"bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0})
        self.assertFalse(model.weight.requires_grad)
        self.assertFalse(model.training)
        module.AutoModel.from_pretrained.assert_not_called()

    def test_backbone_batch_limit_applies_to_ece_views(self):
        module, model, processor = factories()
        with patch.dict(sys.modules, {"transformers": module}):
            engine = BackboneEngine("synthetic", "vlmr1", batch_size=2)
        views = build_views(Image.new("RGB", (56, 28), "white"), "the left object", 1.25)
        result = engine.predict([v["image"] for v in views], [v["expr"] for v in views])
        self.assertEqual(len(result), 4)
        self.assertEqual([len(batch["input_ids"]) for batch in processor.batches], [2, 2])
        self.assertTrue(all(not call["do_sample"] and call["num_return_sequences"] == 1 for call in model.calls))

    def test_original_internvl_uses_native_loader_and_tiled_batch_chat(self):
        module, model, processor = factories("4.44.2")
        with patch.dict(sys.modules, {"transformers": module}):
            engine = BackboneEngine("synthetic", "internvl", batch_size=2, internvl_max_tiles=2)
        output = engine.predict([Image.new("RGB", (16, 8)), Image.new("RGB", (8, 8))],
                                ["object", "another object"], n=2, temperature=1.3)
        module.AutoProcessor.from_pretrained.assert_not_called()
        module.AutoModelForImageTextToText.from_pretrained.assert_not_called()
        self.assertFalse(module.AutoModel.from_pretrained.call_args.kwargs["use_flash_attn"])
        self.assertFalse(module.AutoTokenizer.from_pretrained.call_args.kwargs["use_fast"])
        pixels, questions, generation, counts = model.calls[0]
        self.assertEqual(counts, [3, 1])
        self.assertEqual(tuple(pixels.shape), (4, 3, 8, 8))
        self.assertTrue(all(question.startswith("<image>\n") for question in questions))
        self.assertEqual(generation["num_return_sequences"], 2)
        self.assertAlmostEqual(output[0][0]["bbox"][0], 1.6)
        self.assertAlmostEqual(output[1][1]["bbox"][0], .88)
        self.assertEqual(processor.tokenizer.padding_side, "left")

    def test_internvl_version_error_precedes_weight_loading(self):
        module, _, _ = factories()
        with patch.dict(sys.modules, {"transformers": module}):
            with self.assertRaisesRegex(RuntimeError, "CRVG_BACKBONE_PYTHON"):
                BackboneEngine("synthetic", "internvl")
        module.AutoModel.from_pretrained.assert_not_called()

    def test_tiling_limits_normalization_and_static_config(self):
        image = Image.new("RGB", (80, 16), "white")
        self.assertLessEqual(len(image_tiles(image, image_size=8, max_tiles=3)), 4)
        config = SimpleNamespace(force_image_size=None, vision_config=SimpleNamespace(image_size=8),
                                 dynamic_image_size=False, use_thumbnail=True)
        pixels, counts = pixel_batch([image], config)
        self.assertEqual(counts, [1])
        self.assertEqual(tuple(pixels.shape), (1, 3, 8, 8))
        self.assertAlmostEqual(float(pixels[0, 0, 0, 0]), (1 - .485) / .229, places=5)

    def test_invalid_generation_inputs(self):
        module, model, _ = factories()
        with patch.dict(sys.modules, {"transformers": module}):
            engine = BackboneEngine("synthetic", "vlmr1")
        image = Image.new("RGB", (28, 28))
        for images, exprs, n, temperature in (([], [], 1, 0), ([image], [], 1, 0),
                ([image], ["x"], 0, 0), ([image], ["x"], 2, 0),
                ([image], ["x"], 1, float("nan")), ([image], ["x"], 1, -1)):
            with self.subTest(n=n, temperature=temperature):
                with self.assertRaises(ValueError):
                    engine.predict(images, exprs, n=n, temperature=temperature)
        self.assertEqual(model.calls, [])

    def test_missing_completions_rejected_per_microbatch(self):
        module, model, _ = factories("4.44.2")
        with patch.dict(sys.modules, {"transformers": module}):
            engine = BackboneEngine("synthetic", "internvl")
        model.batch_chat = MagicMock(return_value=[])
        with self.assertRaisesRegex(RuntimeError, "count mismatch"):
            engine.predict([Image.new("RGB", (8, 8))], ["object"])

    def test_pipeline_forwards_backbone_controls_to_bon_and_ece(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("crvg.pipeline.run_stage") as stage:
                cli("crvg.pipeline", ["--backbone", "internvl", "--model-path", "backbone", "--data-root", tmp,
                    "--image-root", tmp, "--qwen-model", "qwen", "--dino-model", "dino", "--log-dir", tmp,
                    "--stop-after", "ece", "--batch-size", "7", "--backbone-batch-size", "2",
                    "--backbone-max-tokens", "100", "--backbone-seed", "123", "--internvl-max-tiles", "6",
                    "--backbone-python", "native-python"])
        for call in stage.call_args_list[:2]:
            argv = call.args[1]
            for key, value in (("--backbone-batch-size", 2), ("--max-tokens", 100),
                               ("--seed", 123), ("--internvl-max-tiles", 6)):
                self.assertEqual(argv[argv.index(key) + 1], value)
            self.assertEqual(call.args[4], "native-python")
        argv = stage.call_args_list[0].args[1]
        self.assertEqual(argv[argv.index("--chunk-size") + 1], 2)


if __name__ == "__main__":
    unittest.main()
