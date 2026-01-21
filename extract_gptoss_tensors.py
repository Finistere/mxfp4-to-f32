#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "gpt-oss>=0.0.9",
#     "numpy>=2.4.1",
#     "safetensors>=0.7.0",
#     "torch>=2.9.1",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = [
#   { index = "pytorch-cpu" },
# ]
# ///

"""
Extract a MXFP4-encoded tensor (blocks/scales) and a dequantized f32 dump from gpt-oss-20b.

The script assumes:
- The https://huggingface.co/openai/gpt-oss-20b model to be in the gpt-oss-20b/ directory
- the data will be written to data/ directory, overwriting any existing files.
"""

import os
import sys
from pathlib import Path

import torch


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    from gpt_oss.torch.weights import Checkpoint

    checkpoint_dir = os.path.join("gpt-oss-20b", "original")
    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)

    blocks_key = "block.0.mlp.mlp1_weight.blocks"
    scales_key = "block.0.mlp.mlp1_weight.scales"

    device = torch.device("cpu")
    checkpoint = Checkpoint(checkpoint_dir, device)
    blocks = checkpoint.get(blocks_key).cpu().contiguous()
    scales = checkpoint.get(scales_key).cpu().contiguous()

    blocks.numpy().tofile(os.path.join(out_dir, f"{blocks_key}.bin"))
    scales.numpy().tofile(os.path.join(out_dir, f"{scales_key}.bin"))

    dequant = checkpoint._get_mxfp4_tensor(blocks_key, scales_key, dtype=torch.float32)
    dequant = dequant.cpu().contiguous()
    dequant.numpy().tofile(os.path.join(out_dir, "block.0.mlp.mlp1_weight.f32.bin"))



if __name__ == "__main__":
    main()
