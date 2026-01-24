# mxfp4-to-f32

Zig 0.15 `std.io.Reader` for MXFP4 quantized F32 [gpt-oss](https://github.com/openai/gpt-oss) tensors.

The specification for MXFP4 can be found here: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf. It does _not_ specify how data is stored, so this implementation is specific to GPT-OSS tensor layout.

The implementation achieves an input throughput of 6.5 GB/s on recent x86 processors with AVX-512BW on 280MB of data and 2.7 GB/s without SIMD. More details in the benchmark section. SIMD detection is done at runtime.

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

## Benchmarks

The first bench loads 1MB of blocks and 64KB of scales into memory, which fits into L3. The second reads 265MB of values and 17MB of scales from disk which ends up generating 2.1GB of floats. In practice it will surely read from RAM though as the machine had plenty. Both use 16KB buffers.

- x86_64 -> scalar
- x86_64_v2 -> SSSE3: 1 block at a time.
- x86_64_v3 -> AVX2: 2 blocks at a time.
- x86_64_v4 -> AVX-512BW: 4 blocks at a time.

Here are the results in terms of input throughput:

| arch                                    | 2M (input GB/s) | 530M (input GB/s) |
| --------------------------------------- | --------------: | ----------------: |
| native with cpu boost                   |            11.6 |               6.4 |
| x86_64 (no simd) with cpu boost         |             3.3 |               2.7 |
| native without cpu boost                |             8.8 |               5.0 |
| x86_64_v4 (avx-512bw) without cpu boost |             8.3 |               4.9 |
| x86_64_v3 (avx2) without cpu boost      |             6.6 |               4.7 |
| x86_64_v2 (ssse3) without cpu boost     |             4.2 |               3.2 |
| x86_64 (no simd) without cpu boost      |             2.5 |               2.1 |

```sh
zig build -Doptimize=ReleaseFast benchmark
```

```text
AMD Ryzen 9 7950X3D
Linux 6.18.5

** WITH CPU BOOST | NATIVE **
benchmark              runs     total time     time/run (avg ± σ)    (min ... max)                p75        p99        p995
-----------------------------------------------------------------------------------------------------------------------------
2M floats (L3 cache)   42986    3.927s         91.365us ± 1.892us    (89.84us ... 121.741us)      90.972us   98.417us   99.608us
530M floats            91       3.989s         43.836ms ± 336.29us   (43.276ms ... 45.804ms)      43.781ms   45.804ms   45.804ms


** WITH CPU BOOST | x86_64 (no simd) **
benchmark              runs     total time     time/run (avg ± σ)    (min ... max)                p75        p99        p995
-----------------------------------------------------------------------------------------------------------------------------
2M floats (L3 cache)   12341    4.004s         324.487us ± 19.961us  (308.574us ... 2.107ms)      325.125us  400.899us  425.265us
530M floats            38       3.927s         103.344ms ± 817.229us (101.434ms ... 104.155ms)    103.854ms  104.155ms  104.155ms


** WITHOUT CPU BOOST | NATIVE **
benchmark              runs     total time     time/run (avg ± σ)    (min ... max)                p75        p99        p995
-----------------------------------------------------------------------------------------------------------------------------
2M floats (L3 cache)   32979    3.995s         121.163us ± 1.94us    (119.766us ... 149.032us)    120.829us  127.25us   127.872us
530M floats            70       3.955s         56.513ms ± 215.626us  (56.285ms ... 57.207ms)      56.557ms   57.207ms   57.207ms
benchmark              runs     total time     time/run (avg ± σ)    (min ... max)                p75        p99        p995


** WITHOUT CPU BOOST | x86_64_v4 (avx-512bw) **
benchmark              runs     total time     time/run (avg ± σ)    (min ... max)                p75        p99        p995
-----------------------------------------------------------------------------------------------------------------------------
2M floats (L3 cache)   31090    3.996s         128.554us ± 2.047us   (126.71us ... 161.366us)     128.663us  135.006us  135.656us
530M floats            69       3.984s         57.74ms ± 245.604us   (57.431ms ... 58.428ms)      57.902ms   58.428ms   58.428ms


** WITHOUT CPU BOOST | x86_64_v3 (avx2) **
benchmark              runs     total time     time/run (avg ± σ)    (min ... max)                p75        p99        p995
-----------------------------------------------------------------------------------------------------------------------------
2M floats (L3 cache)   24931    3.991s         160.1us ± 2.284us     (158.18us ... 199.638us)     160.344us  166.886us  167.627us
530M floats            66       3.955s         59.938ms ± 47.038us   (59.845ms ... 60.046ms)      59.97ms    60.046ms   60.046ms


** WITHOUT CPU BOOST | x86_64_v2 (ssse3) **
benchmark              runs     total time     time/run (avg ± σ)    (min ... max)                p75        p99        p995
-----------------------------------------------------------------------------------------------------------------------------
2M floats (L3 cache)   15753    3.999s         253.877us ± 3.389us   (251.305us ... 339.483us)    256.525us  262.637us  265.102us
530M floats            45       3.964s         88.11ms ± 85.221us    (87.97ms ... 88.363ms)       88.145ms   88.363ms   88.363ms


** WITHOUT CPU BOOST | x86_64 (no simd) **
benchmark              runs     total time     time/run (avg ± σ)    (min ... max)                p75        p99        p995
-----------------------------------------------------------------------------------------------------------------------------
2M floats (L3 cache)   9461     3.998s         422.64us ± 3.372us    (419.373us ... 543.248us)    425.355us  430.655us  433.421us
530M floats            30       3.964s         132.142ms ± 178.454us (131.89ms ... 132.527ms)     132.303ms  132.527ms  132.527ms

```

To disable CPU boost on Linux:

```sh
echo "0" | sudo tee /sys/devices/system/cpu/cpufreq/boost
```

## Profiling

Be sure to have `strip = false` in `build.zig`.

```
zig build -Doptimize=ReleaseFast -Dtarget=x86_64-linux -Dcpu=x86_64_v3 benchmark
valgrind --tool=callgrind --dump-instr=yes --collect-jumps=yes ./zig-out/bin/mxfp4
```
