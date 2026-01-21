# mxfp4-to-f32

Zig `std.io.Reader` for MXFP4 encoded F32 gpt-oss tensors.

## Setup

There is a devenv (Nix) but in short you'll need `zig`, `python` and `uv` and a environment that works for python packages (can compile C libs).

## Tests

There 2 kinds of tests, both inspired from the GPT-OSS code:

- simple generated ones with `generated_test_cases.py` that uses Triton's MXFP4 quantizer.
- one of the tensors from GPT-OSS extracted with `extract_gptoss_tensors.py` and dequantized with the Torch specific code in gpt-oss.

## Benchmark

```sh
zig build -Doptimize=ReleaseFast benchmark
```

```text
AMD Ryzen 9 7950X3D
Linux 6.18.5

** WITH CPU BOOST **
benchmark              runs     total time     time/run (avg ± σ)    (min ... max)                p75        p99        p995
-----------------------------------------------------------------------------------------------------------------------------
2M floats (L3 cache)   10254    4.023s         392.376us ± 7.494us   (375.901us ... 515.245us)    396.37us   414.263us  416.808us
530M floats            50       3.925s         78.517ms ± 1.562ms    (76.602ms ... 81.989ms)      79.326ms   81.989ms   81.989ms
Scalar                 459      3.97s          8.649ms ± 271.575us   (8.29ms ... 9.846ms)         8.789ms    9.393ms    9.598ms
SSSE3                  2894     4s             1.382ms ± 6.362us     (1.372ms ... 1.585ms)        1.385ms    1.393ms    1.394ms


** WITHOUT CPU BOOST **
benchmark              runs     total time     time/run (avg ± σ)    (min ... max)                p75        p99        p995
-----------------------------------------------------------------------------------------------------------------------------
2M floats (L3 cache)   9429     3.997s         423.963us ± 4.22us    (416.538us ... 459.149us)    426.577us  436.095us  438.971us
530M floats            40       3.934s         98.356ms ± 889.807us  (97.737ms ... 101.381ms)     98.188ms   101.381ms  101.381ms
Scalar                 367      3.989s         10.87ms ± 299.144us   (10.534ms ... 12.029ms)      10.979ms   11.85ms    11.987ms
SSSE3                  2206     3.999s         1.813ms ± 5.267us     (1.799ms ... 1.853ms)        1.816ms    1.825ms    1.826ms
```

To disable CPU boost on Linux:

```sh
echo "0" | sudo tee /sys/devices/system/cpu/cpufreq/boost
```
