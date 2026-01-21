pub const dequantize = @import("dequantize.zig");
pub const io = @import("io.zig");

const BLOCK_BYTES_SIZE = 16;
const VALUES_PER_BLOCK = 32;
const BYTES_PER_F32_BLOCK = VALUES_PER_BLOCK * @sizeOf(f32);
