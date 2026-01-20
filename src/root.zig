const std = @import("std");
const builtin = @import("builtin");

const MXFP4_BLOCK_BYTES_SIZE = 17;
const MXFP4_VALUES_PER_BLOCK = 32;

// https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
const SimpleMxfp4Reader = struct {
    inner: std.io.Reader,
    mxfp4_block: [MXFP4_BLOCK_BYTES_SIZE]u8,
    f32_block: [MXFP4_VALUES_PER_BLOCK]f32,
    interface: std.io.Reader,

    pub fn init(inner: std.io.Reader, buffer: []u8, comptime endianness: std.builtin.Endian) SimpleMxfp4Reader {
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
        return SimpleMxfp4Reader{ .inner = inner, .mxfp4_block = undefined, .f32_block = undefined, .interface = reader };
    }

    fn stream(r: *std.io.Reader, w: *std.io.Writer, limit: std.io.Limit) std.io.Reader.StreamError!usize {
        const self: *SimpleMxfp4Reader = @fieldParentPtr("interface", r);

        try self.inner.readSliceAll(&self.mxfp4_block);
        dequantize(&self.mxfp4_block, &self.f32_block);

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

fn dequantize(input: []const u8, output: []f32) void {
    const kvalues = [_]i8{ 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };
    if (input.len != MXFP4_BLOCK_BYTES_SIZE) @panic("Input length must be exactly 17 bytes.");
    if (output.len != MXFP4_VALUES_PER_BLOCK) @panic("Output length must be exactly 32 values.");

    // First byte stores the shared scale exponent for the block.
    const d = e8m0_to_fp32_half(input[0]);

    var i: usize = 0;
    while (i < 16) : (i += 1) {
        // Each following byte contains two 4-bit indices into kvalues.
        const byte = input[1 + i];
        const low = byte & 0x0F;
        const high = byte >> 4;

        // Low nibbles fill the first 16 values; high nibbles fill the last 16.
        output[i] = d * @as(f32, @floatFromInt(kvalues[@as(usize, low)]));
        output[16 + i] = d * @as(f32, @floatFromInt(kvalues[@as(usize, high)]));
    }
}

test "Can read MXFP4" {
    const t = @import("test_utils.zig");

    const gpa = std.testing.allocator;
    var test_cases = try t.loadTestCases(gpa);
    defer {
        for (test_cases.items) |*tc| {
            tc.deinit(gpa);
        }
        test_cases.deinit(gpa);
    }

    const buffer = try gpa.alloc(u8, 128);
    defer gpa.free(buffer);

    for (test_cases.items) |test_case| {
        const mxfp4_reader = std.io.Reader.fixed(test_case.mxfp4_bytes);
        var reader = SimpleMxfp4Reader.init(mxfp4_reader, buffer, .little);
        const out = try gpa.alignedAlloc(u8, .@"4", test_case.f32.len * 4);
        defer gpa.free(out);
        try reader.interface.readSliceAll(out);
        const f32_data: []const f32 = std.mem.bytesAsSlice(f32, out);
        try std.testing.expectEqualSlices(f32, test_case.f32, f32_data);
    }
}
