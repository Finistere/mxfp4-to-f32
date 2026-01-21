#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "numpy>=2.4.1",
#     "torch>=2.9.1",
# "setuptools>=40.8.0",
# "wheel",
# "cmake>=3.20,<4.0",
# "ninja>=1.11.1",
# "pybind11>=2.13.1",
# "lit",
#     "triton",
#     "triton_kernels",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv]
# no-build-isolation = true
# no-build-isolation-package = ["triton"]
#
# [tool.uv.extra-build-dependencies]
# triton = [
# "setuptools>=40.8.0",
# "wheel",
# "cmake>=3.20,<4.0",
# "ninja>=1.11.1",
# "pybind11>=2.13.1",
# "lit"
# ]
#
# [tool.uv.sources]
# torch = [
#   { index = "pytorch-cpu" },
# ]
# triton_kernels = { path = "triton/python/triton_kernels" }
# triton = { path = "triton" }
# ///

"""Generate MXFP4 quantize fixtures using Triton's MXFP4 quantizer."""

import math
import sys
import json
from pathlib import Path

import numpy as np
import torch


OUT_DIR = Path("cases")
MXFP_AXIS = -1


def _write_bin(path: Path, array: np.ndarray) -> None:
    path.write_bytes(array.tobytes())


def _case_data() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    return {
        "const_pi": np.full(32, 3.14, dtype=np.float32),
        "const_one": np.ones(32, dtype=np.float32),
        "ramp_neg_pos": np.linspace(-2.0, 2.0, 32, dtype=np.float32),
        "alt_sign_small": (
            ((-1.0) ** np.arange(32)) * np.float32(0.25)
        ).astype(np.float32),
        "pattern_steps": np.tile(
            np.array([0.0, 0.5, 1.0, 2.0, -1.0, -0.5, 4.0, -3.0], dtype=np.float32), 4
        ),
        "gaussian_0p5": rng.normal(loc=0.0, scale=0.5, size=32).astype(np.float32),
        "tiny_magnitude": np.full(32, 1e-3, dtype=np.float32),
        "large_magnitude": np.full(32, 1e3, dtype=np.float32),
        "mixed_wide": np.array(
            np.concatenate(
                [
                    np.linspace(-1e3, -1e-3, 16, dtype=np.float32),
                    np.linspace(1e-3, 1e3, 16, dtype=np.float32),
                ]
            ),
            dtype=np.float32,
        ),
        "multiple_blocks": np.array(
            np.concatenate(
                [
                    np.full(32, 1e-2, dtype=np.float32),
                    np.full(32, 1e2, dtype=np.float32),
                ]
            ),
            dtype=np.float32,
        ),
    }


def main() -> None:
    from triton_kernels.numerics_details.mxfp import downcast_to_mxfp_torch

    OUT_DIR.mkdir(exist_ok=True)
    manifest: dict[str, list[float]] = {}

    for name, data in _case_data().items():
        x = torch.from_numpy(data)
        blocks, scales = downcast_to_mxfp_torch(x, torch.uint8, axis=MXFP_AXIS)

        blocks = blocks.cpu().contiguous()
        scales = scales.cpu().contiguous()
        _write_bin(OUT_DIR / f"{name}.blocks.bin", blocks.numpy())
        _write_bin(OUT_DIR / f"{name}.scales.bin", scales.numpy())

        # Re-shaping block to match what the dequantize_mxfp4_blocks expects:
        # - blocks shape: (..., G, B) where B is bytes per block (16) and G is the number of blocks.
        # - scales shape: (..., G) — one scale per block. 
        blocks = blocks.view(scales.numel(), -1)
        dequant = dequantize_mxfp4_blocks(blocks, scales).cpu().contiguous()
        _write_bin(OUT_DIR / f"{name}.f32.bin", dequant.numpy())
        manifest[name] = dequant.numpy().reshape(-1).tolist()

    (OUT_DIR / "expected_values.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

# Copied from https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/weights.py
FP4_VALUES = [
    +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]

def dequantize_mxfp4_blocks(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    *,
    rows_per_chunk: int = 16384 * 512,
) -> torch.Tensor:
    scales = scales.to(torch.int32) - 127
    if blocks.shape[:-1] != scales.shape:
        raise ValueError(f"{blocks.shape=} does not match {scales.shape=}")

    lut = torch.tensor(FP4_VALUES, dtype=torch.float32, device=blocks.device)

    *prefix_shape, G, B = blocks.shape
    rows_total = math.prod(prefix_shape) * G

    blocks = blocks.reshape(rows_total, B)
    scales = scales.reshape(rows_total, 1)

    out = torch.empty(rows_total, B * 2, dtype=torch.float32, device=blocks.device)

    for r0 in range(0, rows_total, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, rows_total)

        blk = blocks[r0:r1]
        exp = scales[r0:r1]

        idx_lo = (blk & 0x0F).to(torch.long)
        idx_hi = (blk >> 4).to(torch.long)

        sub = out[r0:r1]
        sub[:, 0::2] = lut[idx_lo]
        sub[:, 1::2] = lut[idx_hi]

        torch.ldexp(sub, exp, out=sub)
        del idx_lo, idx_hi, blk, exp

    return out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)


if __name__ == "__main__":
    main()
