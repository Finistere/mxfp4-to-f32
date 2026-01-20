const std = @import("std");
const builtin = @import("builtin");

pub const MXFP4_BLOCK_BYTES_SIZE = 16;
pub const MXFP4_VALUES_PER_BLOCK = 32;

// https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
pub const SimpleMxfp4Reader = struct {
    blocks_reader: *std.io.Reader,
    scales_reader: *std.io.Reader,
    fp4_block: [MXFP4_BLOCK_BYTES_SIZE]u8,
    f32_block: [MXFP4_VALUES_PER_BLOCK]f32,
    interface: std.io.Reader,

    pub fn init(blocks_reader: *std.io.Reader, scales_reader: *std.io.Reader, buffer: []u8, comptime endianness: std.builtin.Endian) SimpleMxfp4Reader {
        if (buffer.len < MXFP4_VALUES_PER_BLOCK * @sizeOf(f32)) @panic("Buffer must be at least 128 bytes big to contain all values from a single block.");
        // We ensure with the parameter that the caller is aware of how we expose the f32 as bytes. Little-endian will be most likely what's needed
        // and it's used by most modern arch, but better safe than sorry.
        if (endianness != builtin.target.cpu.arch.endian()) @panic("Only native endianness is supported by this adapter.");

        const reader = std.io.Reader{
            .vtable = &.{ .stream = SimpleMxfp4Reader.stream },
            .buffer = buffer,
            .seek = 0,
            .end = 0,
        };
        return SimpleMxfp4Reader{ .blocks_reader = blocks_reader, .scales_reader = scales_reader, .fp4_block = undefined, .f32_block = undefined, .interface = reader };
    }

    fn stream(r: *std.io.Reader, w: *std.io.Writer, limit: std.io.Limit) std.io.Reader.StreamError!usize {
        const self: *SimpleMxfp4Reader = @fieldParentPtr("interface", r);

        const scale: u8 = try self.scales_reader.takeByte();
        try self.blocks_reader.readSliceAll(&self.fp4_block);
        dequantize(scale, &self.fp4_block, &self.f32_block);

        // Here we do assume that we want the current arch endianness which we checked in the init() function.
        // Otherwise we would need to use the writeInt function when it differs which likely has some extra cost.
        const bytes: []const u8 = std.mem.sliceAsBytes(&self.f32_block);

        const n = limit.minInt(bytes.len);
        try w.writeAll(bytes[0..n]);
        const remaining = bytes.len - n;

        if (remaining > 0) {
            if (remaining > r.buffer.len - r.end) {
                return error.ReadFailed;
            }
            @memcpy(r.buffer[r.seek .. r.seek + remaining], bytes[n..]);
            r.end += remaining;
        }

        return n;
    }
};

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

fn dequantize(scale: u8, block: []const u8, output: []f32) void {
    std.debug.assert(block.len == MXFP4_BLOCK_BYTES_SIZE);
    std.debug.assert(output.len == MXFP4_VALUES_PER_BLOCK);

    // First byte stores the shared scale exponent for the block.
    const d = e8m0_to_fp32_half(scale);

    var i: usize = 0;
    while (i < 16) : (i += 1) {
        // Each following byte contains two 4-bit indices into kvalues.
        const byte = block[i];
        const low = byte & 0x0F;
        const high = byte >> 4;

        // Low nibbles fill the first 16 values; high nibbles fill the last 16.
        output[i * 2] = d * @as(f32, @floatFromInt(KVALUES[@as(usize, low)]));
        output[i * 2 + 1] = d * @as(f32, @floatFromInt(KVALUES[@as(usize, high)]));
    }
}
