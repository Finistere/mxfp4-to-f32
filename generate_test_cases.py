#!/usr/bin/env python3
"""Generate MXFP4 quantize/dequantize fixtures for a Zig dequantizer."""
from pathlib import Path

import numpy as np
from gguf.constants import GGMLQuantizationType
from gguf.quants import dequantize, quantize


OUT_DIR = Path("cases")
QTYPE = GGMLQuantizationType.MXFP4


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
    }


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    for name, data in _case_data().items():
        q = quantize(data, QTYPE)
        dq = dequantize(q, QTYPE)

        q_path = OUT_DIR / f"{name}.mxfp4.bin"
        dq_path = OUT_DIR / f"{name}.f32.bin"

        _write_bin(q_path, q)
        _write_bin(dq_path, dq)


if __name__ == "__main__":
    main()
