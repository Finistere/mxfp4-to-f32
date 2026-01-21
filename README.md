# mxfp4-to-f32

Zig `std.io.Reader` for MXFP4 quantized F32 [gpt-oss](https://github.com/openai/gpt-oss) tensors.

The specification for MXFP4 can be found here: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf.
It does _not_ specify how data is stored, so this implementation is specific to GPT-OSS tensor layout.

The benchmark shows a throughput of ~3.4 GB/s but it's very dependent on buffer sizes. In my `std.io.Reader` implementation I'm relying on the nested Reader buffer and output buffer to be big enough that I can decode multiple blocks at once for best performance in the SSSE3 variant. It's a debatable choice, depends who is the consumer (internal lib for one project vs public lib) and how it's meant to be used.

Today the implementation spends a lot of time in `@memcpy`, so that might be improvable, a bit unclear with `std.dio.Reader`.

## Usage

```zig
const std = @import("std");
const mxfp4 = @import("mxfp4");

var scale_reader = std.io.Reader.fixed(scales_bytes);
var block_reader = std.io.Reader.fixed(blocks_bytes);
var reader_buffer: [1024]u8 = undefined;
var reader = mxfp4.io.GptOssReader.init(
    &block_reader,
    &scale_reader,
    &reader_buffer,
    .little,
);

var out: [16 * 1024]u8 = undefined;
try reader.interface.readSliceAll(&out);
```

## Requirements

- Native endianness only (pass the host endianness to `GptOssReader.init`).
- SSSE3 is used when available on x86/x86_64; scalar fallback otherwise.

## Setup

There is a devenv (Nix) but in short you'll need `zig`, `python` and `uv` and an environment that works for Python packages (can compile C libs).

Download the gpt-oss-20b tensors:

```sh
huggingface-cli download openai/gpt-oss-20b --include "original/*" --local-dir gpt-oss-20b/
```

Now you can run:

```sh
# Creates the data/ dir with one extracted GPT-OSS tensor.
./extract_gptoss_tensors.py
```

The test cases are committed to git, they're small enough and running `./generated_test_cases.py` is painful because of Triton and Python. I only get it to work with `uv`, when first installing all dependencies except triton ones and only then re-adding them. `uv` creates a dedicated venv for the script, so `no-build-isolation` applies to the venv and `extra-build-dependencies` doesn't seem to work. Beware that the installation of `triton-kenerls` requires some time compiling.

## Tests

There are 2 kinds of tests, both inspired from the GPT-OSS code:

- simple generated ones with `generated_test_cases.py` that uses Triton's MXFP4 quantizer.
- one of the tensors from GPT-OSS extracted with `extract_gptoss_tensors.py` and dequantized with the Torch specific code in gpt-oss.

Run the tests:

```sh
zig build test
```

## Data layout

The GPT-OSS test expects `data/` to contain:

- `block.0.mlp.mlp1_weight.scales.bin`
- `block.0.mlp.mlp1_weight.blocks.bin`
- `block.0.mlp.mlp1_weight.f32.bin`

## Benchmarks

The first bench loads 1MB of blocks and 64KB of scales into memory, which fits into L3. The second reads 265MB of values and 17MB of scales from disk which ends up generating 2.1GB of floats. In practice it likely reads from RAM though. Both use 16KB buffers. `Scalar` and `SSSE3` both correspond to 1 million blocks dequantization.

In terms of throughput, for the cpu boost version:

- From disk/ram: 3.39 GB/s
- From L3: 3.92 GB/s
- Scalar dequantization: 2.71 GB/s
- SSSE3 dequantization: 58.2 GB/s

Take those numbers with a big grain of salt though. I'm not super confident in the benchmarks, as in I'm not entirely sure what Zig optimized away despite my best effort with `std.mem.doNotOptimizeAway`. The latter can have a significant impact on performance when placed inside the hot loop, so it's not that easy to interpret those numbers.

However the SSSE3 variant was definitely faster, dequantization many blocks at once helped a lot, limiting the `@memcpy` between buffers also. Given the small difference between the L3 and disk/ram benches, it very likely means that it's the Reader implementation that's slowing all of this down.

```sh
zig build -Doptimize=ReleaseFast benchmark
```

```text
AMD Ryzen 9 7950X3D
Linux 6.18.5

** WITH CPU BOOST **
benchmark              runs     total time     time/run (avg ± σ)    (min ... max)                p75        p99        p995
-----------------------------------------------------------------------------------------------------------------------------
2M floats (L3 cache)   15491    3.955s         255.314us ± 3.149us   (250.834us ... 329.583us)    257.216us  265.632us  267.857us
530M floats            51       3.904s         76.55ms ± 1.229ms     (75.499ms ... 80.102ms)      77.134ms   80.102ms   80.102ms
Scalar                 672      3.986s         5.932ms ± 3.316us     (5.925ms ... 5.956ms)        5.932ms    5.946ms    5.95ms
SSSE3                  14377    3.961s         275.536us ± 2.104us   (274.493us ... 362.051us)    277.469us  278.932us  280.445us


** WITHOUT CPU BOOST **
benchmark              runs     total time     time/run (avg ± σ)    (min ... max)                p75        p99        p995
-----------------------------------------------------------------------------------------------------------------------------
2M floats (L3 cache)   12203    3.997s         327.592us ± 3.087us   (323.512us ... 383.735us)    329.914us  335.525us  337.298us
530M floats            41       3.99s          97.321ms ± 404.725us  (96.792ms ... 98.107ms)      97.718ms   98.107ms   98.107ms
Scalar                 518      3.996s         7.714ms ± 8.064us     (7.706ms ... 7.803ms)        7.715ms    7.75ms     7.757ms
SSSE3                  11058    3.998s         361.601us ± 2.255us   (359.976us ... 390.835us)    363.833us  365.928us  367.951us
```

To disable CPU boost on Linux:

```sh
echo "0" | sudo tee /sys/devices/system/cpu/cpufreq/boost
```

## Profiling

Be sure `strip = false` in `build.zig`.

```
zig build -Doptimize=ReleaseFast -Dtarget=x86_64-linux -Dcpu=x86_64_v3 benchmark
valgrind --tool=callgrind --dump-instr=yes --collect-jumps=yes ./zig-out/bin/mxfp4
```
