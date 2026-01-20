const std = @import("std");
const builtin = @import("builtin");

const MXFP4_BLOCK_BYTES_SIZE = 16;
const MXFP4_VALUES_PER_BLOCK = 32;

// https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
const SimpleMxfp4Reader = struct {
    blocks_reader: *std.io.Reader,
    scales_reader: *std.io.Reader,
    fp4_block: [MXFP4_BLOCK_BYTES_SIZE]u8,
    f32_block: [MXFP4_VALUES_PER_BLOCK]f32,
    interface: std.io.Reader,

    pub fn init(blocks_reader: *std.io.Reader, scales_reader: *std.io.Reader, buffer: []u8, comptime endianness: std.builtin.Endian) SimpleMxfp4Reader {
        if (buffer.len < MXFP4_VALUES_PER_BLOCK * @sizeOf(f32)) @panic("Buffer must be at least 128 bits big to contain all values from a single block.");
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

test "Can read MXFP4 from test cases" {
    const t = @import("test_utils.zig");

    const gpa = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();

    const test_cases = try t.loadTestCases(&arena);

    const buffer = try gpa.alloc(u8, 128);
    defer gpa.free(buffer);

    for (test_cases.items) |test_case| {
        std.debug.print("Running test case: {s}\n", .{test_case.name});
        var scale_reader = std.io.Reader.fixed(test_case.scales_bytes);
        var block_reader = std.io.Reader.fixed(test_case.blocks_bytes);
        var reader = SimpleMxfp4Reader.init(&block_reader, &scale_reader, buffer, .little);
        const out = try gpa.alignedAlloc(u8, .@"4", test_case.f32.len * 4);
        defer gpa.free(out);
        try reader.interface.readSliceAll(out);
        const f32_data: []const f32 = std.mem.bytesAsSlice(f32, out);
        try std.testing.expectEqualSlices(f32, test_case.f32, f32_data);
    }
}

const TENSOR_NAME = "block.0.mlp.mlp1_weight";

const BinReader = struct {
    file: std.fs.File,
    filename: []u8,
    buffer: []u8,
    reader: std.fs.File.Reader,

    pub fn init(alloc: std.mem.Allocator, dir: std.fs.Dir, name: []const u8) !BinReader {
        const filename = try std.mem.concat(alloc, u8, &.{ TENSOR_NAME, ".", name, ".bin" });
        var file = try dir.openFile(filename, .{});
        const buffer = try alloc.alloc(u8, 128);
        const reader = file.reader(buffer);
        return BinReader{ .file = file, .filename = filename, .buffer = buffer, .reader = reader };
    }

    pub fn deinit(self: *BinReader, alloc: std.mem.Allocator) void {
        self.file.close();
        alloc.free(self.filename);
        alloc.free(self.buffer);
    }
};

test "Can read GPT-OSS weights" {
    const gpa = std.testing.allocator;
    var dir = try std.fs.cwd().openDir("data", .{});
    defer dir.close();

    var scales_bin = try BinReader.init(gpa, dir, "scales");
    defer scales_bin.deinit(gpa);

    var blocks_bin = try BinReader.init(gpa, dir, "blocks");
    defer blocks_bin.deinit(gpa);

    var expected_bin = try BinReader.init(gpa, dir, "f32");
    defer expected_bin.deinit(gpa);

    const buffer = try gpa.alloc(u8, 128);
    defer gpa.free(buffer);
    var reader = SimpleMxfp4Reader.init(&blocks_bin.reader, &scales_bin.reader, buffer, .little);

    const result = try gpa.alignedAlloc(u8, .@"4", 128);
    defer gpa.free(result);
    const expected = try gpa.alignedAlloc(u8, .@"4", 128);
    defer gpa.free(expected);

    var pos: usize = 0;
    while (true) {
        _ = reader.interface.readSliceAll(result) catch |err| switch (err) {
            error.EndOfStream => break,
            else => return err,
        };
        try expected_bin.reader.readSliceAll(expected);
        const f32_result: []const f32 = std.mem.bytesAsSlice(f32, result);
        const f32_expected: []const f32 = std.mem.bytesAsSlice(f32, expected);
        for (
            f32_result,
            f32_expected,
        ) |res, exp| {
            if (res != exp) {
                const scale_byte: u8 = try readAt(dir, scales_bin.filename, pos);
                const block_byte: u8 = try readAt(dir, blocks_bin.filename, pos / 2);
                std.debug.print(
                    "Mismatch at position {d} : expected {d}, got {d}; scale byte {b}, block byte {b}\n",
                    .{ pos, exp, res, scale_byte, block_byte },
                );
                @panic("");
            }
            pos += 1;
        }
    }
}

fn readAt(dir: std.fs.Dir, name: []u8, pos: usize) !u8 {
    const f = try dir.openFile(name, .{});
    defer f.close();
    var reader = f.reader(&.{});
    try reader.seekTo(@as(u64, pos));
    return reader.interface.takeByte();
}
