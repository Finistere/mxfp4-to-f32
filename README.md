# mxfp4-to-f32

Zig `std.io.Reader` for MXFP4 encoded F32 gpt-oss tensors.

The whole environment is configured with devenv (Nix).

## Test cases generation

I'm relying on the Python implementation of the quantizer from llama.cpp in [gguf-py](https://github.com/ggml-org/llama.cpp/tree/master/gguf-py) which are tested against the C version (apparently from the repo, haven't double checked). The MXFP4 quantizer is only available from the sources GitHub. Their latest `0.17.1` isn't recent enough.
