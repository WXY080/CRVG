"""Exercise model interfaces with deterministic synthetic outputs, without weights."""
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image
import torch

from crvg.controller.features import build_risk_examples
from crvg.settings import DINO_EVIDENCE_SCHEMA
from crvg.utils.data import read_json, write_json
from crvg.verification.qwen import last_logits, load_qwen
from tests.fixtures import A, B, C, case
from tests.test_workflow import cli


class InferenceTests(unittest.TestCase):
    def test_verifier_family_and_frozen_parameters(self):
        model = torch.nn.Linear(2, 2)
        factory = MagicMock()
        factory.from_pretrained.return_value = model
        config = MagicMock()
        config.from_pretrained.return_value = SimpleNamespace(model_type="qwen3_vl")
        processor = MagicMock()
        module = SimpleNamespace(AutoConfig=config, AutoProcessor=processor,
                                 Qwen3VLForConditionalGeneration=factory)
        with patch.dict(sys.modules, {"transformers": module}):
            with patch("torch.cuda.is_available", return_value=False):
                actual, _ = load_qwen("synthetic-model", device_map="cpu")
            self.assertIs(actual, model)
            self.assertFalse(model.training)
            self.assertTrue(all(not p.requires_grad for p in model.parameters()))
            config.from_pretrained.return_value.model_type = "another_model_family"
            with self.assertRaisesRegex(ValueError, "Qwen3"):
                load_qwen("synthetic-model")
        self.assertEqual(factory.from_pretrained.call_count, 1)

    def test_next_token_batch_requires_matching_prompts(self):
        with self.assertRaisesRegex(ValueError, "one prompt"):
            last_logits(None, None, [object()], [])
        with self.assertRaisesRegex(ValueError, "nonempty"):
            last_logits(None, None, [], [])

    def test_dino_to_comparison_preserves_and_scores_all_challengers(self):
        row, _ = case()
        row["candidates"] = [{"bbox": A, "source": "current_system"},
                             {"bbox": C, "source": "sampled"}]
        row["crvg"].update(b1_count=2, b1_min_iou=0.)
        row["text"] = "object"
        close = [.6, 0., 10., 10.]  # Nonduplicate, but overlaps the current box above 0.85.
        proposals = [{"bbox": close, "score": .9}, {"bbox": B, "score": .8}]
        model = MagicMock()
        model.to.return_value = model
        model.eval.return_value = model
        factory = MagicMock()
        factory.from_pretrained.return_value = model
        module = SimpleNamespace(AutoProcessor=MagicMock(),
                                 AutoModelForZeroShotObjectDetection=factory)
        tokenizer = SimpleNamespace(encode=lambda word, **kw: [int(word.strip())])
        processor = SimpleNamespace(tokenizer=tokenizer)

        def logits(_model, _processor, views, prompts):
            self.assertEqual(len(views), len(prompts))
            scores = torch.zeros(len(views), 3)
            scores[0::2, 2] = 2.
            scores[1::2, 1] = 2.
            return scores

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row["image"] = str(root / "scene.png")
            Image.new("RGB", (100, 100), "white").save(row["image"])
            original = copy.deepcopy(row)
            write_json(root / "source.json", {"meta": {}, "results": [row]})
            with patch.dict(sys.modules, {"transformers": module}):
                with patch("crvg.verification.dino_detector.detect", side_effect=[[proposals], []]):
                    cli("crvg.verification.dino_detector", [root / "source.json",
                        "--grounding-model", "synthetic-model", "--device", "cpu",
                        "--save", root / "evidence.json"])
            data = read_json(root / "evidence.json")
            self.assertEqual(data["meta"]["evidence_schema"], DINO_EVIDENCE_SCHEMA)
            self.assertEqual(data["results"][0]["candidates"][:2][0]["bbox"], A)
            self.assertEqual(data["results"][0]["crvg"], original["crvg"])
            self.assertEqual(data["results"][0]["target_detections"], proposals)
            model.requires_grad_.assert_called_once_with(False)
            with patch("crvg.verification.pairwise.load_qwen", return_value=(None, processor)):
                with patch("crvg.verification.pairwise.last_logits", side_effect=logits):
                    cli("crvg.verification.pairwise", [root / "evidence.json",
                        "--verifier-model", "synthetic-model", "--save", root / "picks.json"])
            picks = read_json(root / "picks.json")
            scored = picks["picks"][0]["challengers"]
            self.assertEqual([c["bbox"] for c in scored], [close, B])
            self.assertTrue(all(c["permutation_agree"] for c in scored))
            examples, _ = build_risk_examples(data, picks, "synthetic", require_train=True)
            self.assertEqual(len(examples), 2)

    def test_dino_empty_route_does_not_load_weights(self):
        row, _ = case()
        row["crvg"]["cascade_entered"] = False
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "source.json", {"meta": {}, "results": [row]})
            with patch.dict(sys.modules, {"transformers": None}):
                cli("crvg.verification.dino_detector", [root / "source.json",
                    "--grounding-model", "unavailable", "--save", root / "evidence.json"])
            self.assertEqual(read_json(root / "evidence.json")["results"], [])


if __name__ == "__main__":
    unittest.main()
