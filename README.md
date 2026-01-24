# mxfp4-to-f32

Zig 0.15 `std.io.Reader` for MXFP4 quantized F32 [gpt-oss](https://github.com/openai/gpt-oss) tensors.

The specification for MXFP4 can be found here: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf. It does _not_ specify how data is stored, so this implementation is specific to GPT-OSS tensor layout.

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
- SSSE3 is used when available on x86/x86_64 (runtime detection); scalar fallback otherwise.

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

** WITH CPU BOOST | NATIVE **
benchmark              runs     total time     time/run (avg ± σ)    (min ... max)                p75        p99        p995
-----------------------------------------------------------------------------------------------------------------------------
2M floats (L3 cache)   30352    3.989s         131.425us ± 6.67us    (127.972us ... 898.19us)     130.757us  138.963us  146.147us
530M floats            79       3.933s         49.788ms ± 381.681us  (49.13ms ... 50.826ms)       50.015ms   50.826ms   50.826ms
Scalar                 675      3.99s          5.911ms ± 24.294us    (5.894ms ... 6.17ms)         5.913ms    5.987ms    6.168ms
SIMD1                  2680     3.998s         1.491ms ± 12.969us    (1.487ms ... 1.757ms)        1.492ms    1.503ms    1.513ms
SIMD2                  2410     3.998s         1.659ms ± 20.299us    (1.649ms ... 2.141ms)        1.66ms     1.729ms    1.742ms
SIMD3                  1970     3.998s         2.029ms ± 18.347us    (2.018ms ... 2.294ms)        2.031ms    2.09ms     2.11ms


** WITHOUT CPU BOOST | NATIVE **
benchmark              runs     total time     time/run (avg ± σ)    (min ... max)                p75        p99        p995
-----------------------------------------------------------------------------------------------------------------------------
2M floats (L3 cache)   24034    3.999s         166.407us ± 3.616us   (164.301us ... 435.644us)    166.205us  172.878us  174.11us
530M floats            62       3.973s         64.087ms ± 193.429us  (63.844ms ... 64.619ms)      64.227ms   64.619ms   64.619ms
Scalar                 517      3.994s         7.725ms ± 29.71us     (7.705ms ... 7.979ms)        7.73ms     7.954ms    7.978ms
SIMD1                  2025     4s             1.975ms ± 12.941us    (1.962ms ... 2.233ms)        1.978ms    2.001ms    2.01ms
SIMD2                  1841     4.001s         2.173ms ± 17.134us    (2.165ms ... 2.438ms)        2.173ms    2.2ms      2.25ms
SIMD3                  1506     3.999s         2.655ms ± 17.848us    (2.646ms ... 2.92ms)         2.655ms    2.677ms    2.745ms


** WITHOUT CPU BOOST | x86_64_v4 **
benchmark              runs     total time     time/run (avg ± σ)    (min ... max)                p75        p99        p995
-----------------------------------------------------------------------------------------------------------------------------
2M floats (L3 cache)   24511    4s             163.227us ± 4.723us   (161.125us ... 461.744us)    163.109us  171.595us  180.452us
530M floats            62       3.966s         63.981ms ± 135.033us  (63.839ms ... 64.467ms)      63.994ms   64.467ms   64.467ms
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
