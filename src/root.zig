const std = @import("std");
const builtin = @import("builtin");
const mxfp4 = @import("mxfp4.zig");

// https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
pub const Mxfp4Reader = struct {
    blocks_reader: *std.io.Reader,
    scales_reader: *std.io.Reader,
    fp4_block: [mxfp4.BLOCK_BYTES_SIZE]u8,
    f32_block: [mxfp4.VALUES_PER_BLOCK]f32,
    interface: std.io.Reader,

    pub fn init(blocks_reader: *std.io.Reader, scales_reader: *std.io.Reader, buffer: []u8, comptime endianness: std.builtin.Endian) Mxfp4Reader {
        if (buffer.len < mxfp4.VALUES_PER_BLOCK * @sizeOf(f32)) @panic("Buffer must be at least 128 bytes big to contain all values from a single block.");
        // We ensure with the parameter that the caller is aware of how we expose the f32 as bytes. Little-endian will be most likely what's needed
        // and it's used by most modern arch, but better safe than sorry.
        if (endianness != builtin.target.cpu.arch.endian()) @panic("Only native endianness is supported by this adapter.");

        const reader = std.io.Reader{
            .vtable = &.{ .stream = Mxfp4Reader.stream },
            .buffer = buffer,
            .seek = 0,
            .end = 0,
        };
        return Mxfp4Reader{ .blocks_reader = blocks_reader, .scales_reader = scales_reader, .fp4_block = undefined, .f32_block = undefined, .interface = reader };
    }

    fn stream(r: *std.io.Reader, w: *std.io.Writer, limit: std.io.Limit) std.io.Reader.StreamError!usize {
        const self: *Mxfp4Reader = @fieldParentPtr("interface", r);

        const scale: u8 = try self.scales_reader.takeByte();
        try self.blocks_reader.readSliceAll(&self.fp4_block);
        mxfp4.dequantize_block(scale, &self.fp4_block, &self.f32_block);

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
