//! By convention, root.zig is the root source file when making a library.
const std = @import("std");
const builtin = @import("builtin");

const MXFP4_BLOCK_SIZE = 17;
const MXFP4_VALUES_PER_BLOCK = 32;

// https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
const Mxfp4Adapter = struct {
    inner: std.io.Reader,
    mxfp4_block: [MXFP4_BLOCK_SIZE]u8,
    interface: std.io.Reader,

    pub fn init(inner: std.io.Reader, buffer: []u8, comptime endianness: std.builtin.Endian) Mxfp4Adapter {
        if (buffer.len < MXFP4_VALUES_PER_BLOCK * @sizeOf(f32)) @panic("Buffer must be at least 128 bits big to contain a single block.");
        if (endianness != builtin.target.cpu.arch.endian()) @panic("Only native endianness is supported by this adapter.");

        const reader = std.io.Reader{
            .vtable = &.{ .stream = Mxfp4Adapter.stream },
            .buffer = buffer,
            .seek = 0,
            .end = 0,
        };
        return Mxfp4Adapter{ .inner = inner, .mxfp4_block = undefined, .interface = reader };
    }

    fn stream(r: *std.io.Reader, w: *std.io.Writer, limit: std.io.Limit) std.io.Reader.StreamError!usize {
        const self: *Mxfp4Adapter = @fieldParentPtr("interface", r);

        try self.inner.readSliceAll(&self.mxfp4_block);

        const data: []const f32 = &([_]f32{1.0} ** MXFP4_VALUES_PER_BLOCK);

        // Here I do assume that we want the f32 in the "native" endianness of the current architecture.
        // Most modern architectures are little-endian, so
        const bytes: []const u8 = std.mem.sliceAsBytes(data);

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

test "Can read MXFP4" {
    const t = @import("test_utils.zig");

    const gpa = std.testing.allocator;
    var test_cases = try t.loadTestCase(gpa);
    defer {
        for (test_cases.items) |*tc| {
            tc.deinit(gpa);
        }
        test_cases.deinit(gpa);
    }

    const test_case = &test_cases.items[0];
    const buffer = try gpa.alloc(u8, 128);
    defer gpa.free(buffer);
    std.debug.print("Loaded {d} bytes\n", .{test_case.mxfp4_bytes.len});
    const mxfp4_reader = std.io.Reader.fixed(test_case.mxfp4_bytes);
    var reader = Mxfp4Adapter.init(mxfp4_reader, buffer, .little);

    const result = try gpa.alignedAlloc(u8, .@"4", test_case.f32.len * 4);
    defer gpa.free(result);

    try reader.interface.readSliceAll(result);
    const f32_data: []f32 = std.mem.bytesAsSlice(f32, result);

    std.debug.print("=== Reading {s} ===\n{any}\n\n", .{ test_case.name, f32_data });
}
