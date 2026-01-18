//! By convention, root.zig is the root source file when making a library.
const std = @import("std");

// const Mxfp4Reader = struct {
//     inner: std.io.Reader,
//
//     pub fn init(inner: std.io.Reader) Mxfp4Reader {
//         return Mxfp4Reader{
//             .inner = inner,
//         };
//     }
//
//     pub fn reader() *std.io.Reader {
//         return .{ .vtable = &.{ .stream = stream }, .buffer = &.{}, .seek = 0, .end = 0 };
//     }
//
//     fn stream(r: *std.io.Reader, w: *std.io.Writer, limit: std.io.Limit) std.io.StreamError!usize {}
// };
//
test "Can load test cases" {
    const t = @import("test_utils.zig");

    const gpa = std.testing.allocator;
    var test_cases = try t.loadTestCase(gpa);
    defer {
        for (test_cases.items) |*tc| {
            tc.deinit(gpa);
        }
        test_cases.deinit(gpa);
    }
}
