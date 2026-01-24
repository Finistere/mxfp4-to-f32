const std = @import("std");
const builtin = @import("builtin");
const mxfp4 = @import("root.zig");

const SCALAR_VTABLE = std.io.Reader.VTable{ .stream = GptOssReader.stream_scalar };
const SIMD1_VTABLE = std.io.Reader.VTable{ .stream = GptOssReader.stream_simd1 };
const SIMD2_VTABLE = std.io.Reader.VTable{ .stream = GptOssReader.stream_simd2 };
const SIMD4_VTABLE = std.io.Reader.VTable{ .stream = GptOssReader.stream_simd4 };

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
            .vtable = switch (mxfp4.dequantize.simdBlockWidth()) {
                4 => &SIMD4_VTABLE,
                2 => &SIMD2_VTABLE,
                1 => &SIMD1_VTABLE,
                else => &SCALAR_VTABLE,
            },
            .buffer = buffer,
            .seek = 0,
            .end = 0,
        };
        return GptOssReader{ .blocks_reader = blocks_reader, .scales_reader = scales_reader, .f32_block = undefined, .interface = interface };
    }

    fn stream_scalar(r: *std.io.Reader, w: *std.io.Writer, limit: std.io.Limit) std.io.Reader.StreamError!usize {
        const self: *GptOssReader = @fieldParentPtr("interface", r);

        // Ensure we read at least one block up to as many as possible up to the limit.
        const block_count = try self.fill_readers_up_to_complete_block(limit);

        // We're reading/writing directly from/into the reader/writer buffers.
        const scales_buffer = self.scales_reader.buffered();
        const blocks_buffer = self.blocks_reader.buffered();
        var output_slice = try w.writableSlice(block_count * mxfp4.BYTES_PER_F32_BLOCK);

        for (0..block_count) |i| {
            const scale: u8 = scales_buffer[i];
            const block: *const [16]u8 = @ptrCast(blocks_buffer[i * mxfp4.BLOCK_BYTES_SIZE .. (i + 1) * mxfp4.BLOCK_BYTES_SIZE].ptr);
            const output: *[mxfp4.BYTES_PER_F32_BLOCK]u8 = @ptrCast(output_slice[i * mxfp4.BYTES_PER_F32_BLOCK .. (i + 1) * mxfp4.BYTES_PER_F32_BLOCK].ptr);
            mxfp4.dequantize.gpt_oss_one_block(scale, @bitCast(block.*), output);
        }

        try self.discard_read_blocks(block_count);
        return block_count * mxfp4.BYTES_PER_F32_BLOCK;
    }

    fn stream_simd1(r: *std.io.Reader, w: *std.io.Writer, limit: std.io.Limit) std.io.Reader.StreamError!usize {
        return GptOssReader.stream_simd(1, r, w, limit);
    }

    fn stream_simd2(r: *std.io.Reader, w: *std.io.Writer, limit: std.io.Limit) std.io.Reader.StreamError!usize {
        return GptOssReader.stream_simd(2, r, w, limit);
    }

    fn stream_simd4(r: *std.io.Reader, w: *std.io.Writer, limit: std.io.Limit) std.io.Reader.StreamError!usize {
        return GptOssReader.stream_simd(4, r, w, limit);
    }

    /// To improve reader speed this function decodes as many blocks as requested at once
    /// using SIMD for N blocks at a time.
    fn stream_simd(comptime N: u8, r: *std.io.Reader, w: *std.io.Writer, limit: std.io.Limit) std.io.Reader.StreamError!usize {
        const self: *GptOssReader = @fieldParentPtr("interface", r);

        // Ensure we read at least one block up to as many as possible up to the limit.
        const block_count = try self.fill_readers_up_to_complete_block(limit);

        // We're reading/writing directly from/into the reader/writer buffers.
        const scales_buffer = self.scales_reader.buffered();
        const blocks_buffer = self.blocks_reader.buffered();
        var output_slice = try w.writableSlice(block_count * mxfp4.BYTES_PER_F32_BLOCK);

        // First process as many blocks as possible in SIMD with the highest possible width.
        const simd_blocks = block_count / N;
        for (0..simd_blocks) |i| {
            const scales: *const [N]u8 = @ptrCast(scales_buffer[i * N .. (i + 1) * N].ptr);
            const blocks: *const [N * 16]u8 = @ptrCast(blocks_buffer[(i * mxfp4.BLOCK_BYTES_SIZE * N) .. (i + 1) * mxfp4.BLOCK_BYTES_SIZE * N].ptr);
            const output: *[N * mxfp4.BYTES_PER_F32_BLOCK]u8 = @ptrCast(output_slice[i * N * mxfp4.BYTES_PER_F32_BLOCK .. (i + 1) * N * mxfp4.BYTES_PER_F32_BLOCK].ptr);
            mxfp4.dequantize.gpt_oss_blocks_simd(N, scales.*, @bitCast(blocks.*), output);
        }

        // Process remaining blocks one by one.
        for ((simd_blocks * N)..block_count) |i| {
            const scales: [1]u8 = .{scales_buffer[i]};
            const blocks: *const [16]u8 = @ptrCast(blocks_buffer[(i * mxfp4.BLOCK_BYTES_SIZE) .. (i + 1) * mxfp4.BLOCK_BYTES_SIZE].ptr);
            const output: *[mxfp4.BYTES_PER_F32_BLOCK]u8 = @ptrCast(output_slice[i * mxfp4.BYTES_PER_F32_BLOCK .. (i + 1) * mxfp4.BYTES_PER_F32_BLOCK].ptr);
            mxfp4.dequantize.gpt_oss_blocks_simd(1, scales, @bitCast(blocks.*), output);
        }

        try self.discard_read_blocks(block_count);
        return block_count * mxfp4.BYTES_PER_F32_BLOCK;
    }

    fn fill_readers_up_to_complete_block(self: *GptOssReader, limit: std.io.Limit) std.io.Reader.StreamError!usize {
        const blocks_limit = @intFromEnum(limit) / mxfp4.BYTES_PER_F32_BLOCK;

        // TODO: Here we assume that readers use a big enough buffer to at least retrieve one block of f32.
        //       I think it's a sensible expectation that at least 128 bytes buffer are used as MXFP4 is only ever used
        //       to store massive amounts of floats, but we could support that.
        if (blocks_limit == 0) return error.ReadFailed;

        // Try to fill as many blocks as possible up to the limit.
        var block_count = blocks_limit;
        self.blocks_reader.fill(blocks_limit * mxfp4.BLOCK_BYTES_SIZE) catch |err| switch (err) {
            error.EndOfStream => {
                switch (self.blocks_reader.bufferedLen()) {
                    // We have truly nothing left, return the EndOfStream.
                    0 => return if (self.scales_reader.peekByte()) |_| error.ReadFailed else |_| err,
                    // We have not enough data left for a full block.
                    1...mxfp4.BLOCK_BYTES_SIZE - 1 => return error.ReadFailed,
                    else => {
                        block_count = self.blocks_reader.bufferedLen() / mxfp4.BLOCK_BYTES_SIZE;
                    },
                }
            },
            else => return err,
        };

        // Ensure we have enough scales for the blocks we can read.
        if (self.scales_reader.bufferedLen() < block_count) {
            self.scales_reader.fill(block_count) catch |err| switch (err) {
                error.EndOfStream => {
                    // we don't have enough scales for the blocks we loaded,
                    // there is discrepancy between the blocks and the scales.
                    return error.ReadFailed;
                },
                else => return err,
            };
        }

        return block_count;
    }

    inline fn discard_read_blocks(self: *GptOssReader, count: usize) !void {
        try self.scales_reader.discardAll(count);
        try self.blocks_reader.discardAll(count * mxfp4.BLOCK_BYTES_SIZE);
    }
};
