const zbench = @import("zbench");
const std = @import("std");
const mxfp4 = @import("mxfp4");

const SCALES_BYTES = 64 * 1024;
const BLOCKS_BYTES = 1024 * 1024;

/// Benchmark reading 2M floats (64K scales and 1M blocks) which should fit in L3 cache.
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
        var read_buffer: [16 * 1024]u8 = undefined;

        var reader = mxfp4.io.GptOssReader.init(
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

/// Benchmark reading 530M floats from disk.
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
    var reader = mxfp4.io.GptOssReader.init(&blocks_bin.reader.interface, &scales_bin.reader.interface, buffer, .little);

    const result = try alloc.alignedAlloc(u8, .@"4", 16 * 1024);
    defer alloc.free(result);

    while (true) {
        _ = reader.interface.readSliceAll(result) catch |err| switch (err) {
            error.EndOfStream => break,
            else => return err,
        };
    }
}

/// Benchmark dequantization of a single block using SSSE3.
fn DequantizeSIMDBench(comptime N: usize) type {
    return struct {
        scales: *const [N]u8,
        blocks: *const [N * 16]u8,

        pub fn init(alloc: std.mem.Allocator) !@This() {
            var dir = try std.fs.cwd().openDir("data", .{});
            defer dir.close();
            const scales = try loadPrefix(alloc, dir, "block.0.mlp.mlp1_weight.scales.bin", N);
            const blocks = try loadPrefix(alloc, dir, "block.0.mlp.mlp1_weight.blocks.bin", N * 16);
            return .{ .scales = scales, .blocks = blocks };
        }

        pub fn run(self: *@This(), _: std.mem.Allocator) void {
            var output: [mxfp4.BYTES_PER_F32_BLOCK * N]u8 = undefined;
            for (0..1000_000) |_| {
                mxfp4.dequantize.gpt_oss_blocks_simd(N, self.scales.*, self.blocks.*, &output);
            }
            std.mem.doNotOptimizeAway(output);
        }

        pub fn deinit(self: @This(), alloc: std.mem.Allocator) void {
            alloc.free(self.scales);
            alloc.free(self.blocks);
        }
    };
}

/// Benchmark dequantization of a single block using scalar code.
const DequantizeBench = struct {
    scale: u8,
    block: []const u8,

    pub fn init(alloc: std.mem.Allocator) !DequantizeBench {
        var dir = try std.fs.cwd().openDir("cases", .{});
        defer dir.close();
        const scales_bytes = try loadPrefix(alloc, dir, "mixed_wide.scales.bin", 1);
        defer alloc.free(scales_bytes);
        const blocks_bytes = try loadPrefix(alloc, dir, "mixed_wide.blocks.bin", 16);
        return DequantizeBench{ .scale = scales_bytes[0], .block = blocks_bytes };
    }

    pub fn run(self: *DequantizeBench, _: std.mem.Allocator) void {
        var f32_output: [32]f32 = undefined;
        for (0..1000_000) |_| {
            mxfp4.dequantize.gpt_oss_one_block(self.scale, self.block[0..16].*, &f32_output);
        }
        std.mem.doNotOptimizeAway(f32_output);
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

    // const dequantize_benchmark = try DequantizeBench.init(allocator);
    // defer dequantize_benchmark.deinit(allocator);
    //
    // const dequantize_simd1_benchmark = try DequantizeSIMDBench(1).init(allocator);
    // defer dequantize_simd1_benchmark.deinit(allocator);
    //
    // const dequantize_simd2_benchmark = try DequantizeSIMDBench(2).init(allocator);
    // defer dequantize_simd2_benchmark.deinit(allocator);
    //
    // const dequantize_simd4_benchmark = try DequantizeSIMDBench(4).init(allocator);
    // defer dequantize_simd4_benchmark.deinit(allocator);

    var bench = zbench.Benchmark.init(allocator, .{
        .time_budget_ns = 4 * 1_000_000_000,
    });
    defer bench.deinit();
    try bench.addParam("2M floats (L3 cache)", &l3_cache_reader_benchmark, .{});
    try bench.add("530M floats", &gpt_oss_tensor_reader_bench, .{});
    // try bench.addParam("Scalar", &dequantize_benchmark, .{});
    // try bench.addParam("SIMD1", &dequantize_simd1_benchmark, .{});
    // try bench.addParam("SIMD2", &dequantize_simd2_benchmark, .{});
    // try bench.addParam("SIMD3", &dequantize_simd4_benchmark, .{});
    try bench.run(writer);

    return writer.flush();
}

fn loadPrefix(
    alloc: std.mem.Allocator,
    dir: std.fs.Dir,
    filename: []const u8,
    comptime N: usize,
) !*const [N]u8 {
    var file = try dir.openFile(filename, .{});
    defer file.close();

    const out = try alloc.alloc(u8, N);
    errdefer alloc.free(out);
    const buffer = try alloc.alloc(u8, 1024);
    defer alloc.free(buffer);
    var reader = file.reader(buffer);
    try reader.interface.readSliceAll(out);
    return @ptrCast(out.ptr);
}

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
