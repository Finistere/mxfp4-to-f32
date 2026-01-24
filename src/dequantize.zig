const std = @import("std");
const builtin = @import("builtin");
const mxfp4 = @import("root.zig");

const E2M1_LUT = [_]f32{
    0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
};

/// Fastest non-vectorized implementation on x86_64 I've tested.
/// Inspired from https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/weights.py
pub fn gpt_oss_one_block(scale_e8m0: u8, block_e2m1: [mxfp4.BLOCK_BYTES_SIZE]u8, output: *[mxfp4.VALUES_PER_BLOCK]f32) void {
    const scale = e8m0_to_fp32(scale_e8m0);

    var i: usize = 0;
    while (i < 16) : (i += 1) {
        // Each following byte contains two 4-bit indices into kvalues.
        const byte = block_e2m1[i];
        const low = byte & 0x0F;
        const high = byte >> 4;

        output[i * 2] = scale * E2M1_LUT[@as(usize, low)];
        output[i * 2 + 1] = scale * E2M1_LUT[@as(usize, high)];
    }
}

fn e8m0_to_fp32(x: u8) f32 {
    const bits: u32 = if (x == 0)
        0x0040_0000
    else
        (@as(u32, x) << 23);

    return @bitCast(bits);
}

const E2M1_DOUBLED_LUT: @Vector(16, i8) = @bitCast([_]i8{ 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 });
const E2M1_DOUBLED_LUT_2: @Vector(32, i8) = @bitCast([_]i8{ 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 } ** 2);
const E2M1_DOUBLED_LUT_4: @Vector(64, i8) = @bitCast([_]i8{ 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 } ** 4);

/// Inspired from https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-impl.h#L467
/// Adapted for SIMD on amd64
pub fn gpt_oss_blocks_simd(comptime N: usize, scales_e8m0: [N]u8, blocks_e2m1: @Vector(N * 16, u8), output: *[N * mxfp4.BYTES_PER_F32_BLOCK]u8) void {
    comptime {
        if (N != 1 and N != 2 and N != 4)
            @compileError("blocks must be 1, 2, or 4");
    }

    const hinibble = blocks_e2m1 >> @as(@Vector(N * 16, u3), @splat(@as(u3, 4)));
    const lonibble = blocks_e2m1 & @as(@Vector(N * 16, u8), @splat(@as(u8, 0x0F)));

    const lut = switch (N) {
        1 => E2M1_DOUBLED_LUT,
        2 => E2M1_DOUBLED_LUT_2,
        4 => E2M1_DOUBLED_LUT_4,
        else => unreachable,
    };
    const hivalues = pack_shuffle_bytes(lut, hinibble);
    const lovalues = pack_shuffle_bytes(lut, lonibble);
    const values_i8: @Vector(N * 32, i8) = std.simd.interlace(.{ lovalues, hivalues });
    const arr_i8: [N * 32]u8 = @bitCast(values_i8);

    //
    // Even with AVX-512 we can multiply at most 16 f32, so process values mxfp4 block by block to have a single scale
    // and let Zig figure out the best way to do all of this with SIMD.
    for (0..N) |i| {
        const block_i8_ptr: *const [32]i8 = @ptrCast(arr_i8[i * 32 .. (i + 1) * 32].ptr);
        const block_i8: @Vector(32, i8) = @bitCast(block_i8_ptr.*);
        const values_f32: @Vector(32, f32) = @floatFromInt(block_i8);
        const scale: @Vector(32, f32) = @splat(e8m0_to_fp32_half(scales_e8m0[i]));
        const result: @Vector(32, f32) = values_f32 * scale;
        const arr: [128]u8 = @bitCast(result);
        @memcpy(output[i * 128 .. (i + 1) * 128], arr[0..]);
    }
}

fn e8m0_to_fp32_half(x: u8) f32 {
    if (x < 2) {
        // Denorm/low exponent path used by ggml_e8m0_to_fp32_half.
        const bits: u32 = @as(u32, 0x0020_0000) << @as(u5, @intCast(x));
        return @bitCast(bits);
    }
    // Normal path: exponent stored with an implicit -1 shift.
    const bits: u32 = @as(u32, x - 1) << 23;
    return @bitCast(bits);
}

pub fn simdBlockWidth() u8 {
    const target = builtin.target;
    if (target.cpu.has(.x86, .avx512bw)) return 4;
    if (target.cpu.has(.x86, .avx2)) return 2;
    if (target.cpu.has(.x86, .ssse3)) return 1;
    return 0;
}

/// https://www.felixcloutier.com/x86/pshufb
/// Performs a byte-wise shuffle of the first operand (table) according to the indices specified in the second operand (mask).
/// This function required the LLVM backend for the AVX variants.
fn pack_shuffle_bytes(table: anytype, mask: anytype) @TypeOf(table) {
    const T = @TypeOf(table);
    comptime {
        switch (@typeInfo(T)) {
            .vector => {},
            else => @compileError("pack_shuffle_bytes only works with vector types"),
        }
        if (@TypeOf(mask) != @Vector(@typeInfo(T).vector.len, u8))
            @compileError("mask must be same length vector of u8");
    }

    const lanes = @typeInfo(T).vector.len;
    const cpu = builtin.target.cpu;

    if (lanes == 64 and cpu.has(.x86, .avx512bw)) {
        var dst = table;
        asm volatile ("vpshufb %[mask], %[src], %[dst]"
            : [dst] "=x" (dst),
            : [src] "x" (table),
              [mask] "x" (mask),
            : .{});
        return dst;
    } else if (lanes == 32 and cpu.has(.x86, .avx2)) {
        var dst = table;
        asm volatile ("vpshufb %[mask], %[src], %[dst]"
            : [dst] "=x" (dst),
            : [src] "x" (table),
              [mask] "x" (mask),
            : .{});
        return dst;
    } else if (lanes == 16 and cpu.has(.x86, .ssse3)) {
        var dst = table;
        asm volatile ("pshufb %[mask], %[dst]"
            : [dst] "+x" (dst),
            : [mask] "x" (mask),
            : .{});
        return dst;
    } else if (lanes % 16 == 0) {
        // FIXME: Fallback implementation that shouldn't be used, I'm just not really sure how to properly handle
        // runtime detection to be honest. This function should only be called if SIMD is available in the first place.
        var dst: @TypeOf(table) = undefined;
        for (0..(lanes / 16)) |v| {
            const offset = v * 16;
            for (0..16) |i| {
                const index = mask[offset + i];
                if (index & 0x80 != 0) {
                    dst[offset + i] = 0;
                } else {
                    dst[offset + i] = table[offset + @as(usize, index)];
                }
            }
        }
        return dst;
    } else {
        @compileError("Unsupported vector length or missing CPU feature");
    }
}
