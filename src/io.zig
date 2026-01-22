const std = @import("std");
const builtin = @import("builtin");
const mxfp4 = @import("root.zig");

const SCALAR_VTABLE = std.io.Reader.VTable{ .stream = GptOssReader.stream_scalar };
const SSSE3_VTABLE = std.io.Reader.VTable{ .stream = GptOssReader.stream_ssse3_blocks };

/// https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
/// An io.Reader adapter that reads MXFP4 compressed data from two underlying readers for the scales and blocks.
///
/// This implementation is specific to GPT-OSS weights.
pub const GptOssReader = struct {
    blocks_reader: *std.io.Reader,
    scales_reader: *std.io.Reader,
    f32_block: [mxfp4.VALUES_PER_BLOCK]f32,
    interface: std.io.Reader,

    pub fn init(blocks_reader: *std.io.Reader, scales_reader: *std.io.Reader, buffer: []u8, comptime endianness: std.builtin.Endian) GptOssReader {
        if (buffer.len < mxfp4.BYTES_PER_F32_BLOCK) @panic("Buffer must be at least 128 bytes big to contain all values from a single block.");
        if (blocks_reader.buffer.len < mxfp4.BLOCK_BYTES_SIZE) @panic("blocks_reader buffer must be at least 16 bytes big to read a full block.");
        // We ensure with the parameter that the caller is aware of how we expose the f32 as bytes.
        // Little-endian will be most likely what's needed and it's used by most modern arch, but better safe than sorry.
        if (endianness != builtin.target.cpu.arch.endian()) @panic("Only native endianness is supported by this adapter.");

        const interface = std.io.Reader{
            .vtable = if (hasSsse3()) &SSSE3_VTABLE else &SCALAR_VTABLE,
            .buffer = buffer,
            .seek = 0,
            .end = 0,
        };
        return GptOssReader{ .blocks_reader = blocks_reader, .scales_reader = scales_reader, .f32_block = undefined, .interface = interface };
    }

    fn stream_scalar(r: *std.io.Reader, w: *std.io.Writer, limit: std.io.Limit) std.io.Reader.StreamError!usize {
        const self: *GptOssReader = @fieldParentPtr("interface", r);

        const scale: u8 = try self.scales_reader.takeByte();
        try self.blocks_reader.fill(mxfp4.BLOCK_BYTES_SIZE);
        // Here we use the blocks reader buffer directly avoiding a costly copy. I would love to do the same for the f32 block,
        // writing directly into the writer. But we have no guarantee on the alignment and it's not reasonable to put
        // expectations on it for a generic Reader.
        mxfp4.dequantize.gpt_oss_one_block(scale, self.blocks_reader.buffered()[0..mxfp4.BLOCK_BYTES_SIZE].*, &self.f32_block);

        try self.blocks_reader.discardAll(mxfp4.BLOCK_BYTES_SIZE);

        // Here we do assume that we want the current arch endianness which we checked in the init() function.
        // Otherwise we would need to use the writeInt function when it differs which likely has some extra cost.
        const bytes: []const u8 = std.mem.sliceAsBytes(&self.f32_block);

        const l: usize = @intFromEnum(limit);
        switch (std.math.order(l, bytes.len)) {
            .lt => {
                try w.writeAll(bytes[0..l]);
                const remaining = bytes.len - l;
                if (remaining > r.buffer.len - r.end) return error.ReadFailed;
                @memcpy(r.buffer[r.seek .. r.seek + remaining], bytes[l..]);
                r.end += remaining;
                return l;
            },
            else => {
                try w.writeAll(bytes);
                return bytes.len;
            },
        }
    }

    /// This function is kept as a reference for this exercise.
    fn stream_ssse3(r: *std.io.Reader, w: *std.io.Writer, limit: std.io.Limit) std.io.Reader.StreamError!usize {
        const self: *GptOssReader = @fieldParentPtr("interface", r);

        const scale: u8 = try self.scales_reader.takeByte();
        try self.blocks_reader.fill(mxfp4.BLOCK_BYTES_SIZE);
        // Here we use the blocks reader buffer directly avoiding a costly copy.
        const f32_vector: @Vector(mxfp4.VALUES_PER_BLOCK, f32) = mxfp4.dequantize.gpt_oss_one_block_ssse3(
            scale,
            self.blocks_reader.buffered()[0..mxfp4.BLOCK_BYTES_SIZE].*,
        );

        try self.blocks_reader.discardAll(mxfp4.BLOCK_BYTES_SIZE);

        // Here we do assume that we want the current arch endianness which we checked in the init() function.
        // Otherwise we would need to use the writeInt function when it differs which likely has some extra cost.
        const arr: *const [mxfp4.BYTES_PER_F32_BLOCK]u8 = @ptrCast(&f32_vector);
        const bytes: []const u8 = arr[0..];

        const l: usize = @intFromEnum(limit);
        switch (std.math.order(l, bytes.len)) {
            .lt => {
                try w.writeAll(bytes[0..l]);
                const remaining = bytes.len - l;
                if (remaining > r.buffer.len - r.end) return error.ReadFailed;
                @memcpy(r.buffer[r.seek .. r.seek + remaining], bytes[l..]);
                r.end += remaining;
                return l;
            },
            else => {
                try w.writeAll(bytes);
                return bytes.len;
            },
        }
    }

    /// To improve reader speed this function decodes as many blocks as requested at once
    fn stream_ssse3_blocks(r: *std.io.Reader, w: *std.io.Writer, limit: std.io.Limit) std.io.Reader.StreamError!usize {
        const self: *GptOssReader = @fieldParentPtr("interface", r);

        const blocks_limit = @intFromEnum(limit) / mxfp4.BYTES_PER_F32_BLOCK;

        // Here we assume that readers use a big enough buffer to at least retrieve one block of f32.
        if (blocks_limit == 0) return error.ReadFailed;

        var block_count = bufferedBlockCount(self.blocks_reader, blocks_limit) catch |err| switch (err) {
            error.EndOfStream => {
                // If we still have bytes left in the scales, there is a discrepancy between the blocks and the scales.
                return if (self.scales_reader.peekByte()) |_| error.ReadFailed else |_| err;
            },
            else => return err,
        };

        if (self.scales_reader.bufferedLen() < block_count) {
            self.scales_reader.fillMore() catch |err| switch (err) {
                error.EndOfStream => {
                    // we don't have enough scales for the blocks we loaded, there is discrepancy between the blocks and the scales.
                    return error.ReadFailed;
                },
                else => return err,
            };
            block_count = @min(block_count, self.scales_reader.bufferedLen());
        }
        if (block_count == 0) return 0;

        const scales_buffer = self.scales_reader.buffered();
        const blocks_buffer = self.blocks_reader.buffered();
        for (0..block_count) |i| {
            const scale: u8 = scales_buffer[i];
            const block = blocks_buffer[(i * mxfp4.BLOCK_BYTES_SIZE) .. (i + 1) * mxfp4.BLOCK_BYTES_SIZE];
            const block_ptr = @as(*const [16]u8, @ptrCast(block.ptr));

            // Here we use the blocks reader buffer directly avoiding a costly copy.
            const f32_vector: @Vector(mxfp4.VALUES_PER_BLOCK, f32) = mxfp4.dequantize.gpt_oss_one_block_ssse3(
                scale,
                @bitCast(block_ptr.*),
            );

            // Here we do assume that we want the current arch endianness which we checked in the init() function.
            // Otherwise we would need to use the writeInt function when it differs which likely has some extra cost.
            const arr: *const [mxfp4.BYTES_PER_F32_BLOCK]u8 = @ptrCast(&f32_vector);
            const bytes: []const u8 = arr[0..];

            try w.writeAll(bytes);
        }

        try self.scales_reader.discardAll(block_count);
        try self.blocks_reader.discardAll(block_count * mxfp4.BLOCK_BYTES_SIZE);
        return block_count * mxfp4.BYTES_PER_F32_BLOCK;
    }

    /// Returns the number of full blocks currently buffered, up to blocks_limit.
    fn bufferedBlockCount(reader: *std.io.Reader, blocks_limit: usize) std.io.Reader.StreamError!usize {
        var n = reader.bufferedLen() / mxfp4.BLOCK_BYTES_SIZE;
        if (n >= blocks_limit) return blocks_limit;

        reader.fillMore() catch |err| switch (err) {
            error.EndOfStream => {
                return switch (reader.bufferedLen()) {
                    0 => err,
                    // We have not enough data left for a full block.
                    1...mxfp4.BLOCK_BYTES_SIZE - 1 => error.ReadFailed,
                    else => n,
                };
            },
            else => return err,
        };

        n = reader.bufferedLen() / mxfp4.BLOCK_BYTES_SIZE;
        return @min(n, blocks_limit);
    }
};

/// Returns true if the CPU supports SSSE3 instructions at runtime.
/// Very likely to be useless as I would expect all somewhat recent CPUs to have SSSE3
/// But given the target is ZML, a library, runtime detection gives the most flexibility.
fn hasSsse3() bool {
    var query = std.Target.Query.fromTarget(&builtin.target);
    query.cpu_model = .native; // ask OS/compiler to resolve real CPU
    const native_target = std.zig.system.resolveTargetQuery(query) catch return false;
    return native_target.cpu.has(.x86, .ssse3);
}
