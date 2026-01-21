const std = @import("std");
const mxfp4_to_f32 = @import("mxfp4_to_f32");
const generated_test_cases = @import("generated_test_cases.zig");

test "Test case floats are consistent with expected values." {
    const gpa = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();

    const expected = try generated_test_cases.loadExpectedValues(&arena);
    const test_cases = try generated_test_cases.load(&arena);

    for (test_cases.items) |tc| {
        std.debug.print("Validating test case {s}\n", .{tc.name});
        const entry_opt = expected.case_name_to_values.get(tc.name);
        try std.testing.expect(entry_opt != null);
        const values = entry_opt.?;
        try std.testing.expectEqual(values.len, tc.f32.len);

        for (values, 0..) |value, idx| {
            try std.testing.expectApproxEqAbs(value, tc.f32[idx], 1e-6);
        }
    }
}

test "Can read MXFP4 from test cases" {
    const gpa = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();

    const test_cases = try generated_test_cases.load(&arena);

    const buffer = try gpa.alloc(u8, 128);
    defer gpa.free(buffer);

    for (test_cases.items) |test_case| {
        std.debug.print("Running test case: {s}\n", .{test_case.name});
        var scale_reader = std.io.Reader.fixed(test_case.scales_bytes);
        var block_reader = std.io.Reader.fixed(test_case.blocks_bytes);
        var reader = mxfp4_to_f32.Mxfp4Reader.init(&block_reader, &scale_reader, buffer, .little);
        const out = try gpa.alignedAlloc(u8, .@"4", test_case.f32.len * 4);
        defer gpa.free(out);
        try reader.interface.readSliceAll(out);
        const f32_data: []const f32 = std.mem.bytesAsSlice(f32, out);
        try std.testing.expectEqualSlices(f32, test_case.f32, f32_data);
    }
}

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
    var reader = mxfp4_to_f32.Mxfp4Reader.init(&blocks_bin.reader.interface, &scales_bin.reader.interface, buffer, .little);

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
        try expected_bin.reader.interface.readSliceAll(expected);
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

fn readAt(dir: std.fs.Dir, name: []const u8, pos: usize) !u8 {
    const f = try dir.openFile(name, .{});
    defer f.close();
    var reader = f.reader(&.{});
    try reader.seekTo(@as(u64, pos));
    return reader.interface.takeByte();
}
