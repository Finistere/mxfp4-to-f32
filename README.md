# mxfp4-to-f32

Zig 0.15 `std.io.Reader` for MXFP4 quantized F32 [gpt-oss](https://github.com/openai/gpt-oss) tensors.

The specification for MXFP4 can be found here: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf. It does _not_ specify how data is stored, so this implementation is specific to GPT-OSS tensor layout.

SIMD support for SSSE3, AVX2, and AVX-512BW on x86, NEON on aarch64, and scalar fallback.

For GPT-OSS tensor data (scales + blocks):

- 282MB on:
  - AMD 7950X3D takes 44ms at 6.0GiB/s in | 45.3GiB/s out
  - Apple M2 Max takes 72ms at 3.7GiB/s in | 27.6GiB/s out

- 1MB (fits in L3) on:
  - AMD 7950X3D takes 84us at 11.7GiB/s in | 88.5GiB/s out
  - Apple M2 Max takes 184us at 5.4GiB/s in | 40.5GiB/s out

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

| arch                                    | 2M (input GiB/s) | 2M (output GiB/s) | 530M (input GiB/s) | 530M (output GiB/s) |
| --------------------------------------- | ---------------: | ----------------: | -----------------: | ------------------: |
| native with cpu boost                   |             11.7 |              88.5 |                6.0 |                45.3 |
| x86_64 (no simd) with cpu boost         |              3.1 |              23.3 |                2.5 |                19.1 |
| native without cpu boost                |              8.8 |              66.2 |                4.7 |                35.4 |
| x86_64_v4 (avx-512bw) without cpu boost |              8.4 |              63.3 |                4.6 |                35.0 |
| x86_64_v3 (avx2) without cpu boost      |              8.3 |              62.6 |                4.6 |                34.3 |
| x86_64_v2 (ssse3) without cpu boost     |              4.1 |              31.1 |                3.0 |                22.7 |
| x86_64 (no simd) without cpu boost      |              2.4 |              17.8 |                2.0 |                15.0 |
| Apple M2 Max (native)                   |              5.4 |              40.5 |                3.7 |                27.6 |

```sh
zig build -Doptimize=ReleaseFast benchmark
```

```text
AMD Ryzen 9 7950X3D
Linux 6.18.5

** WITH CPU BOOST | NATIVE **
benchmark              runs     total time     time/run (avg ± σ)    (min ... max)                p75        p99        p995
-----------------------------------------------------------------------------------------------------------------------------
2M floats (L3 cache)   69975    5.893s         84.217us ± 2.01us     (81.664us ... 148.942us)     84.44us    89.61us    91.923us
530M floats            137      5.972s         43.597ms ± 115.188us  (43.473ms ... 44.246ms)      43.623ms   44.165ms   44.246ms


** WITH CPU BOOST | x86_64 (no simd) **
benchmark              runs     total time     time/run (avg ± σ)    (min ... max)                p75        p99        p995
-----------------------------------------------------------------------------------------------------------------------------
2M floats (L3 cache)   18691    5.987s         320.36us ± 3.341us    (317.241us ... 525.454us)    322.31us   327.509us  328.231us
530M floats            57       5.892s         103.373ms ± 620.55us  (102.79ms ... 106.991ms)     103.508ms  106.991ms  106.991ms


** WITHOUT CPU BOOST | NATIVE **
benchmark              runs     total time     time/run (avg ± σ)    (min ... max)                p75        p99        p995
-----------------------------------------------------------------------------------------------------------------------------
2M floats (L3 cache)   53270    5.998s         112.608us ± 1.89us    (111.601us ... 143.201us)    112.032us  118.214us  119.106us
530M floats            107      5.971s         55.811ms ± 158.272us  (55.622ms ... 56.458ms)      55.945ms   56.175ms   56.458ms


** WITHOUT CPU BOOST | x86_64_v4 (avx-512bw) **
benchmark              runs     total time     time/run (avg ± σ)    (min ... max)                p75        p99        p995
-----------------------------------------------------------------------------------------------------------------------------
2M floats (L3 cache)   51011    6.004s         117.714us ± 2.464us   (116.28us ... 234.704us)     117.162us  124.466us  132.18us
530M floats            106      5.982s         56.442ms ± 313.54us   (56.186ms ... 58.196ms)      56.498ms   57.737ms   58.196ms


** WITHOUT CPU BOOST | x86_64_v3 (avx2) **
benchmark              runs     total time     time/run (avg ± σ)    (min ... max)                p75        p99        p995
-----------------------------------------------------------------------------------------------------------------------------
2M floats (L3 cache)   50438    5.999s         118.95us ± 1.959us    (117.833us ... 193.987us)    118.524us  124.456us  125.297us
530M floats            103      5.935s         57.624ms ± 181.38us   (57.416ms ... 58.206ms)      57.664ms   58.204ms   58.206ms


** WITHOUT CPU BOOST | x86_64_v2 (ssse3) **
benchmark              runs     total time     time/run (avg ± σ)    (min ... max)                p75        p99        p995
-----------------------------------------------------------------------------------------------------------------------------
2M floats (L3 cache)   25050    5.999s         239.52us ± 2.568us    (236.477us ... 283.416us)    239.503us  245.264us  246.507us
530M floats            68       5.911s         86.938ms ± 107.676us  (86.785ms ... 87.241ms)      87.014ms   87.241ms   87.241ms


** WITHOUT CPU BOOST | x86_64 (no simd) **
benchmark              runs     total time     time/run (avg ± σ)    (min ... max)                p75        p99        p995
-----------------------------------------------------------------------------------------------------------------------------
2M floats (L3 cache)   14317    5.997s         418.901us ± 3.293us   (415.727us ... 491.38us)     421.708us  423.962us  428.631us
530M floats            45       5.942s         132.063ms ± 201.106us (131.678ms ... 132.593ms)    132.181ms  132.593ms  132.593ms

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

## PGO / BOLT

Can not use PGO with Zig code AFAIK.

I've tried using BOLT 21 but without any success. Benchmarks results were the same. Here are the steps I followed:

```zig
// build.zig
exe.link_emit_relocs = true;
```

```sh
perf record -e cycles:u -j any,u -o perf.data ./zig-out/bin/mxfp4
perf2bolt ./zig-out/bin/mxfp4 -perfdata=perf.data -o app.fdata
llvm-bolt ./zig-out/bin/mxfp4 -o mxfp4.bolt -data=app.fdata -reorder-blocks=ext-tsp -reorder-functions=cdsort -split-functions -split-all-cold -split-eh -dyno-stats
```
