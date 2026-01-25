#!/usr/bin/env bash

# Enable CPU boost
echo "1" | sudo tee /sys/devices/system/cpu/cpufreq/boost

echo "** WITH CPU BOOST | NATIVE **"
zig build -Doptimize=ReleaseFast benchmark

echo -e "\n\n** WITH CPU BOOST | x86_64 (no simd) **"
zig build -Doptimize=ReleaseFast -Dcpu=x86_64 benchmark

# Disable CPU boost
echo "0" | sudo tee /sys/devices/system/cpu/cpufreq/boost

echo -e "\n\n** WITHOUT CPU BOOST | NATIVE **"
zig build -Doptimize=ReleaseFast benchmark

echo -e "\n\n** WITHOUT CPU BOOST | x86_64_v4 (avx-512bw) **"
zig build -Doptimize=ReleaseFast -Dcpu=x86_64_v4 benchmark

echo -e "\n\n** WITHOUT CPU BOOST | x86_64_v3 (avx2) **"
zig build -Doptimize=ReleaseFast -Dcpu=x86_64_v3 benchmark

echo -e "\n\n** WITHOUT CPU BOOST | x86_64_v2 (ssse3) **"
zig build -Doptimize=ReleaseFast -Dcpu=x86_64_v2 benchmark

echo -e "\n\n** WITHOUT CPU BOOST | x86_64 (no simd) **"
zig build -Doptimize=ReleaseFast -Dcpu=x86_64 benchmark
