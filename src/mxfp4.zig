const std = @import("std");

pub const BLOCK_BYTES_SIZE = 16;
pub const VALUES_PER_BLOCK = 32;

fn e8m0_to_fp32_half(x: u8) f32 {
    if (x < 2) {
        // Denorm/low exponent path used by ggml_e8m0_to_fp32_half.
        const bits: u32 = @as(u32, 0x00200000) << @as(u5, @intCast(x));
        return @bitCast(bits);
    }
    // Normal path: exponent stored with an implicit -1 shift.
    const bits: u32 = @as(u32, x - 1) << 23;
    return @bitCast(bits);
}

const KVALUES = [_]i8{ 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };

pub fn dequantize_block(scale: u8, block: []const u8, output: []f32) void {
    const scale_half = e8m0_to_fp32_half(scale);
    std.debug.assert(block.len >= BLOCK_BYTES_SIZE);

    var i: usize = 0;
    while (i < 16) : (i += 1) {
        // Each following byte contains two 4-bit indices into kvalues.
        const byte = block[i];
        const low = byte & 0x0F;
        const high = byte >> 4;

        // Low nibbles fill the first 16 values; high nibbles fill the last 16.
        output[i * 2] = scale_half * @as(f32, @floatFromInt(KVALUES[@as(usize, low)]));
        output[i * 2 + 1] = scale_half * @as(f32, @floatFromInt(KVALUES[@as(usize, high)]));
    }
}
