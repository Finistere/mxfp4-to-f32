const zbench = @import("zbench");
const std = @import("std");
const mxfp4_to_f32 = @import("mxfp4_to_f32");

const SCALES_BYTES = 64 * 1024;
const BLOCKS_BYTES = 1024 * 1024;

const SimpleReaderBench = struct {
    scales: []const u8,
    blocks: []const u8,

    pub fn run(self: SimpleReaderBench, _: std.mem.Allocator) void {
        var scale_reader = std.io.Reader.fixed(self.scales);
        var block_reader = std.io.Reader.fixed(self.blocks);

        var reader_buffer: [1024]u8 = undefined;
        var read_buffer: [1024]u8 = undefined;

        var reader = mxfp4_to_f32.Mxfp4Reader.init(
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
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var dir = try std.fs.cwd().openDir("data", .{});
    defer dir.close();

    const scales_bytes = try loadPrefix(allocator, dir, "block.0.mlp.mlp1_weight.scales.bin", SCALES_BYTES);
    defer allocator.free(scales_bytes);
    const blocks_bytes = try loadPrefix(allocator, dir, "block.0.mlp.mlp1_weight.blocks.bin", BLOCKS_BYTES);
    defer allocator.free(blocks_bytes);

    const bench_impl = SimpleReaderBench{
        .scales = scales_bytes,
        .blocks = blocks_bytes,
    };

    var stdout_buffer: [1024]u8 = undefined;
    var stdout = std.fs.File.stdout().writer(&stdout_buffer);
    const writer = &stdout.interface;

    try writer.print("Running mxfp4 reader benchmark...\n", .{});

    var bench = zbench.Benchmark.init(allocator, .{
        .time_budget_ns = 4 * 1_000_000_000,
    });
    defer bench.deinit();
    try bench.addParam("2M floats", &bench_impl, .{});
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
