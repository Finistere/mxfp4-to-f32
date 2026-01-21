const zbench = @import("zbench");
const std = @import("std");
const mxfp4 = @import("mxfp4");

const SCALES_BYTES = 64 * 1024;
const BLOCKS_BYTES = 1024 * 1024;

const L3CacheReaderBench = struct {
    scales: []const u8,
    blocks: []const u8,

    pub fn init(alloc: std.mem.Allocator) !L3CacheReaderBench {
        var dir = try std.fs.cwd().openDir("data", .{});
        defer dir.close();
        const scales_bytes = try loadPrefix(alloc, dir, "block.0.mlp.mlp1_weight.scales.bin", SCALES_BYTES);
        const blocks_bytes = try loadPrefix(alloc, dir, "block.0.mlp.mlp1_weight.blocks.bin", BLOCKS_BYTES);
        return L3CacheReaderBench{
            .scales = scales_bytes,
            .blocks = blocks_bytes,
        };
    }

    pub fn run(self: *L3CacheReaderBench, _: std.mem.Allocator) void {
        var scale_reader = std.io.Reader.fixed(self.scales);
        var block_reader = std.io.Reader.fixed(self.blocks);

        var reader_buffer: [1024]u8 = undefined;
        var read_buffer: [1024]u8 = undefined;

        var reader = mxfp4.io.Reader.init(
            &block_reader,
            &scale_reader,
            &reader_buffer,
            .little,
        );

        while (true) {
            reader.interface.readSliceAll(&read_buffer) catch |err| switch (err) {
                error.EndOfStream => break,
                else => @panic(@errorName(err)),
            };
        }
    }

    pub fn deinit(self: L3CacheReaderBench, alloc: std.mem.Allocator) void {
        alloc.free(self.scales);
        alloc.free(self.blocks);
    }
};

fn gpt_oss_tensor_reader_bench(alloc: std.mem.Allocator) void {
    faillible_gpt_oss_tensor_reader_bench(alloc) catch @panic("Bench failure");
}

fn faillible_gpt_oss_tensor_reader_bench(alloc: std.mem.Allocator) !void {
    var dir = try std.fs.cwd().openDir("data", .{});
    defer dir.close();

    var scales_bin = try BinReader.init(alloc, dir, "block.0.mlp.mlp1_weight.scales.bin");
    defer scales_bin.deinit(alloc);

    var blocks_bin = try BinReader.init(alloc, dir, "block.0.mlp.mlp1_weight.blocks.bin");
    defer blocks_bin.deinit(alloc);

    const buffer = try alloc.alloc(u8, 1024);
    defer alloc.free(buffer);
    var reader = mxfp4.io.Reader.init(&blocks_bin.reader.interface, &scales_bin.reader.interface, buffer, .little);

    const result = try alloc.alignedAlloc(u8, .@"4", 8192);
    defer alloc.free(result);

    while (true) {
        _ = reader.interface.readSliceAll(result) catch |err| switch (err) {
            error.EndOfStream => break,
            else => return err,
        };
    }
}

const DequantizeSSSE3Bench = struct {
    scale: u8,
    block: []const u8,

    pub fn init(alloc: std.mem.Allocator) !DequantizeSSSE3Bench {
        var dir = try std.fs.cwd().openDir("cases", .{});
        defer dir.close();
        const scales_bytes = try loadPrefix(alloc, dir, "mixed_wide.scales.bin", 1);
        const blocks_bytes = try loadPrefix(alloc, dir, "mixed_wide.blocks.bin", 16);
        return DequantizeSSSE3Bench{ .scale = scales_bytes[0], .block = blocks_bytes };
    }

    pub fn run(self: *DequantizeSSSE3Bench, _: std.mem.Allocator) void {
        for (0..1000_000) |_| {
            std.mem.doNotOptimizeAway(mxfp4.dequantize.one_block_ssse3(self.scale, self.block[0..16].*));
        }
    }

    pub fn deinit(self: DequantizeSSSE3Bench, alloc: std.mem.Allocator) void {
        alloc.free(self.block);
    }
};

const DequantizeBench = struct {
    scale: u8,
    block: []const u8,

    pub fn init(alloc: std.mem.Allocator) !DequantizeBench {
        var dir = try std.fs.cwd().openDir("cases", .{});
        defer dir.close();
        const scales_bytes = try loadPrefix(alloc, dir, "mixed_wide.scales.bin", 1);
        const blocks_bytes = try loadPrefix(alloc, dir, "mixed_wide.blocks.bin", 16);
        return DequantizeBench{ .scale = scales_bytes[0], .block = blocks_bytes };
    }

    pub fn run(self: *DequantizeBench, _: std.mem.Allocator) void {
        var f32_output: [32]f32 = undefined;
        for (0..1000_000) |_| {
            mxfp4.dequantize.one_block(self.scale, self.block[0..16].*, &f32_output);
            std.mem.doNotOptimizeAway(f32_output);
        }
    }

    pub fn deinit(self: DequantizeBench, alloc: std.mem.Allocator) void {
        alloc.free(self.block);
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var stdout_buffer: [1024]u8 = undefined;
    var stdout = std.fs.File.stdout().writer(&stdout_buffer);
    const writer = &stdout.interface;

    const l3_cache_reader_benchmark = try L3CacheReaderBench.init(allocator);
    defer l3_cache_reader_benchmark.deinit(allocator);

    const dequantize_benchmark = try DequantizeBench.init(allocator);
    defer dequantize_benchmark.deinit(allocator);

    const dequantize_ssse3_benchmark = try DequantizeSSSE3Bench.init(allocator);
    defer dequantize_ssse3_benchmark.deinit(allocator);

    var bench = zbench.Benchmark.init(allocator, .{
        .time_budget_ns = 4 * 1_000_000_000,
    });
    defer bench.deinit();
    try bench.addParam("2M floats (L3 cache)", &l3_cache_reader_benchmark, .{});
    try bench.add("530M floats", &gpt_oss_tensor_reader_bench, .{});
    try bench.addParam("Scalar", &dequantize_benchmark, .{});
    try bench.addParam("SSSE3", &dequantize_ssse3_benchmark, .{});
    try bench.run(writer);

    return writer.flush();
}

fn loadPrefix(alloc: std.mem.Allocator, dir: std.fs.Dir, filename: []const u8, len: usize) ![]u8 {
    var file = try dir.openFile(filename, .{});
    defer file.close();

    const out = try alloc.alloc(u8, len);
    errdefer alloc.free(out);
    const buffer = try alloc.alloc(u8, 1024);
    defer alloc.free(buffer);
    var reader = file.reader(buffer);
    try reader.interface.readSliceAll(out);
    return out;
}

const TENSOR_NAME = "block.0.mlp.mlp1_weight";

const BinReader = struct {
    file: std.fs.File,
    buffer: []u8,
    reader: std.fs.File.Reader,

    pub fn init(alloc: std.mem.Allocator, dir: std.fs.Dir, filename: []const u8) !BinReader {
        var file = try dir.openFile(filename, .{});
        const buffer = try alloc.alloc(u8, 16 * 1024);
        const reader = file.reader(buffer);
        return BinReader{ .file = file, .buffer = buffer, .reader = reader };
    }

    pub fn deinit(self: *BinReader, alloc: std.mem.Allocator) void {
        self.file.close();
        alloc.free(self.buffer);
    }
};
