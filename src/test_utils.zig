const std = @import("std");

pub const TestCase = struct {
    name: []const u8,
    mxfp4_bytes: []u8,
    f32: []f32,

    pub fn deinit(self: *TestCase, alloc: std.mem.Allocator) void {
        alloc.free(self.name);
        alloc.free(self.mxfp4_bytes);
        alloc.free(self.f32);
    }
};

pub fn loadTestCase(alloc: std.mem.Allocator) !std.ArrayList(TestCase) {
    var dir = try std.fs.cwd().openDir("cases", .{ .iterate = true });
    defer dir.close();

    var it = dir.iterate();
    var cases = try std.ArrayList(TestCase).initCapacity(alloc, 32);
    while (try it.next()) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, ".f32")) continue;
        const f32_filename = entry.name;

        // We must have a dot because this file ends with ".f32"
        const dot = std.mem.lastIndexOfScalar(u8, f32_filename, '.') orelse unreachable;
        const name = try alloc.dupe(u8, f32_filename[0..dot]);
        const mxfp4_filename = try std.mem.concat(alloc, u8, &.{ name, ".mxfp4" });
        defer alloc.free(mxfp4_filename);

        const max_bytes = 1 << 20; // 1 MiB
        const mxfp4_bytes = try dir.readFileAlloc(alloc, mxfp4_filename, max_bytes);
        const f32_bytes = try dir.readFileAllocOptions(alloc, f32_filename, max_bytes, null, .of(f32), null);
        if (f32_bytes.len % 4 != 0) {
            @panic("f32 file length is not a multiple of 4");
        }
        const f32_data: []f32 = std.mem.bytesAsSlice(f32, f32_bytes);

        try cases.append(alloc, TestCase{
            .name = name,
            .mxfp4_bytes = mxfp4_bytes,
            .f32 = f32_data,
        });
    }

    return cases;
}

test "Can load test cases" {
    const gpa = std.testing.allocator;
    var test_cases = try loadTestCase(gpa);
    defer {
        for (test_cases.items) |*tc| {
            tc.deinit(gpa);
        }
        test_cases.deinit(gpa);
    }

    for (test_cases.items) |tc| {
        std.debug.print("=== {s} ===\n{any}\n\n", .{ tc.name, tc.f32 });
    }
}
