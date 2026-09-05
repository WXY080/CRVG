# Third-Party Components

CRVG uses the public interfaces of:

- [PyTorch](https://github.com/pytorch/pytorch)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Pillow](https://github.com/python-pillow/Pillow)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
- [InternVL](https://github.com/OpenGVLab/InternVL)
- [VLM-R1](https://github.com/om-ai-lab/VLM-R1)
- [RefCOCO and RefCOCO+](https://arxiv.org/abs/1608.00272)
- [RefCOCOg](https://arxiv.org/abs/1511.02283)
- [REFER dataset loader and annotations](https://github.com/lichengunc/refer)

InternVL dynamic tiling in `crvg/candidate_generation/internvl.py` follows OpenGVLab's [model-card preprocessing](https://huggingface.co/OpenGVLab/InternVL3-9B), released under the MIT License (Copyright 2024 OpenGVLab).

The consensus routing, spatial evidence, comparison rendering, and risk-controller implementations are provided in this repository. External model weights, images, and annotations are obtained from their official sources; their own licenses and access conditions apply.
