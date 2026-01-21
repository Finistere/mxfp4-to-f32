const std = @import("std");
const builtin = @import("builtin");
const mxfp4 = @import("root.zig");

const FP4_VALUES = [_]f32{
    0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
};

// Fastest non-vectorized implementation on x86_64 I've tested.
// Inspired from https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/weights.py
pub fn gpt_oss_one_block(scale_e8m0: u8, block: [mxfp4.BLOCK_BYTES_SIZE]u8, output: []f32) void {
    const scale = e8m0_to_fp32(scale_e8m0);

    var i: usize = 0;
    while (i < 16) : (i += 1) {
        // Each following byte contains two 4-bit indices into kvalues.
        const byte = block[i];
        const low = byte & 0x0F;
        const high = byte >> 4;

        output[i * 2] = scale * FP4_VALUES[@as(usize, low)];
        output[i * 2 + 1] = scale * FP4_VALUES[@as(usize, high)];
    }
}

fn e8m0_to_fp32(x: u8) f32 {
    const bits: u32 = if (x == 0)
        0x0040_0000
    else
        (@as(u32, x) << 23);

    return @bitCast(bits);
}

//
// Inspired from https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-impl.h#L467
// Adjusted for SSE3
//
pub fn gpt_oss_one_block_ssse3(scale_e8m0: u8, block: @Vector(16, u8)) @Vector(32, f32) {
    const scale_half = e8m0_to_fp32_half(scale_e8m0);

    const hinibble = block >> HIGH_SHIFT;
    const lonibble = block & LOW_NIBBLE_MASK;
    const hivalues = pshufb(KVALUES, hinibble);
    const lovalues = pshufb(KVALUES, lonibble);
    const values_i8: @Vector(32, i8) = std.simd.interlace(.{ lovalues, hivalues });
    const values_f32: @Vector(32, f32) = @floatFromInt(values_i8);

    const scale_vector: @Vector(32, f32) = @splat(scale_half);
    return values_f32 * scale_vector;
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

const KVALUES = [_]i8{ 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };
const LOW_NIBBLE_MASK: @Vector(16, u8) = @splat(0x0F);
const HIGH_SHIFT: @Vector(16, u3) = @splat(4);

// https://www.felixcloutier.com/x86/pshufb (SSE3)
// Performs a byte-wise shuffle of the first operand (table) according to the indices specified in the second operand (mask).
fn pshufb(table: @Vector(16, i8), mask: @Vector(16, u8)) @Vector(16, i8) {
    var dst = table;
    asm volatile ("pshufb %[mask], %[dst]"
        : [dst] "+x" (dst), // input/output
        : [mask] "x" (mask),
        : .{});
    return dst;
}
