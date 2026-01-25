# mxfp4-to-f32

Zig 0.15 `std.io.Reader` for MXFP4 quantized F32 [gpt-oss](https://github.com/openai/gpt-oss) tensors.

The specification for MXFP4 can be found here: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf. It does _not_ specify how data is stored, so this implementation is specific to GPT-OSS tensor layout.

SIMD support for SSE3, AVX2 and AVX-512BW on x86 and NEON on aarch64.

The implementation processes 280MB of data (scales + blocks) on:

- AMD 7950X3D in 44ms at 6.4GB/s in | 48.5GB/s out
- Apple M2 in 72ms at 3.9GB/s in | 29.7GB/s out

For AVX support I had to use the LLVM backend of Zig.

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

Here are the results in terms of throughput:

| arch                                    | 2M (input GB/s) | 2M (output GB/s) | 530M (input GB/s) | 530M (output GB/s) |
| --------------------------------------- | --------------: | ---------------: | ----------------: | -----------------: |
| native with cpu boost                   |            11.8 |             89.0 |               6.4 |               48.5 |
| x86_64 (no simd) with cpu boost         |             3.3 |             25.1 |               2.8 |               20.8 |
| native without cpu boost                |             9.2 |             69.2 |               5.1 |               38.1 |
| x86_64_v4 (avx-512bw) without cpu boost |             8.7 |             65.5 |               5.0 |               37.7 |
| x86_64_v3 (avx2) without cpu boost      |             7.2 |             54.4 |               4.9 |               36.8 |
| x86_64_v2 (ssse3) without cpu boost     |             4.4 |             33.2 |               3.2 |               24.4 |
| x86_64 (no simd) without cpu boost      |             2.5 |             19.1 |               2.1 |               16.0 |
| Apple M2 Max (native)                   |             5.8 |             43.5 |               3.9 |               29.7 |

```sh
zig build -Doptimize=ReleaseFast benchmark
```

```text
AMD Ryzen 9 7950X3D
Linux 6.18.5

** WITH CPU BOOST | NATIVE **
benchmark              runs     total time     time/run (avg ± σ)    (min ... max)                p75        p99        p995
-----------------------------------------------------------------------------------------------------------------------------
2M floats (L3 cache)   67936    6.104s         89.858us ± 2.024us    (85.912us ... 173.118us)     89.369us   95.561us   98.115us
530M floats            137      5.989s         43.719ms ± 30.649us   (43.662ms ... 43.829ms)      43.735ms   43.804ms   43.829ms


** WITH CPU BOOST | x86_64 (no simd) **
benchmark              runs     total time     time/run (avg ± σ)    (min ... max)                p75        p99        p995
-----------------------------------------------------------------------------------------------------------------------------
2M floats (L3 cache)   18768    5.991s         319.258us ± 2.427us   (316.008us ... 355.352us)    321.588us  325.997us  330.596us
530M floats            58       5.923s         102.129ms ± 629.35us  (101.573ms ... 105.33ms)     102.164ms  105.33ms   105.33ms


** WITHOUT CPU BOOST | NATIVE **
benchmark              runs     total time     time/run (avg ± σ)    (min ... max)                p75        p99        p995
-----------------------------------------------------------------------------------------------------------------------------
2M floats (L3 cache)   51895    6s             115.633us ± 2.794us   (114.326us ... 177.005us)    114.977us  122.722us  132.26us
530M floats            107      5.956s         55.667ms ± 123.448us  (55.541ms ... 56.177ms)      55.693ms   56.059ms   56.177ms


** WITHOUT CPU BOOST | x86_64_v4 (avx-512bw) **
benchmark              runs     total time     time/run (avg ± σ)    (min ... max)                p75        p99        p995
-----------------------------------------------------------------------------------------------------------------------------
2M floats (L3 cache)   49068    5.997s         122.23us ± 1.926us    (120.568us ... 160.644us)    122.271us  128.052us  128.503us
530M floats            106      5.968s         56.304ms ± 130.04us   (56.1ms ... 57.028ms)        56.359ms   56.646ms   57.028ms


** WITHOUT CPU BOOST | x86_64_v3 (avx2) **
benchmark              runs     total time     time/run (avg ± σ)    (min ... max)                p75        p99        p995
-----------------------------------------------------------------------------------------------------------------------------
2M floats (L3 cache)   40814    5.997s         146.951us ± 1.965us   (145.736us ... 172.427us)    146.276us  152.378us  153.029us
530M floats            103      5.941s         57.68ms ± 187.41us    (57.454ms ... 58.312ms)      57.739ms   58.138ms   58.312ms


** WITHOUT CPU BOOST | x86_64_v2 (ssse3) **
benchmark              runs     total time     time/run (avg ± σ)    (min ... max)                p75        p99        p995
-----------------------------------------------------------------------------------------------------------------------------
2M floats (L3 cache)   24874    5.999s         241.202us ± 2.603us   (239.363us ... 311.119us)    240.896us  246.967us  249.462us
530M floats            69       5.988s         86.79ms ± 93.059us    (86.678ms ... 87.126ms)      86.848ms   87.126ms   87.126ms


** WITHOUT CPU BOOST | x86_64 (no simd) **
benchmark              runs     total time     time/run (avg ± σ)    (min ... max)                p75        p99        p995
-----------------------------------------------------------------------------------------------------------------------------
2M floats (L3 cache)   14295    6s             419.727us ± 3.308us   (416.718us ... 564.959us)    422.56us   424.974us  429.473us
530M floats            45       5.946s         132.154ms ± 316.676us (131.661ms ... 133.302ms)    132.271ms  133.302ms  133.302ms

```

```
Apple M2 Max

benchmark              runs     total time     time/run (avg ± σ)    (min ... max)                p75        p99        p995
-----------------------------------------------------------------------------------------------------------------------------
2M floats (L3 cache)   32395    5.963s         184.075us ± 6.048us   (179.875us ... 283.042us)    187.625us  205.125us  209.916us
530M floats            84       6.005s         71.496ms ± 885.576us  (69.92ms ... 75.591ms)       71.875ms   75.591ms   75.591ms
```

To disable CPU boost on Linux:

```sh
echo "0" | sudo tee /sys/devices/system/cpu/cpufreq/boost
```

## Profiling

Be sure to have `strip = false` in `build.zig`.

```sh
# My Valgrind doesn't support more recent instructions.
zig build -Doptimize=ReleaseFast -Dtarget=x86_64-linux -Dcpu=x86_64_v3 benchmark
valgrind --tool=callgrind --dump-instr=yes --collect-jumps=yes ./zig-out/bin/mxfp4
```
