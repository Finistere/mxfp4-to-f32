{
  pkgs,
  lib,
  ...
}: let
  rev = "2fbde785bc106ae1c4102b0e82b9b41d9c466579";
  py = pkgs.python313.override {
    packageOverrides = final: prev: {
      gguf = prev.gguf.overridePythonAttrs (old: {
        version = "git-${rev}";
        src = pkgs.fetchFromGitHub {
          inherit rev;
          owner = "ggml-org";
          repo = "llama.cpp";
          hash = "sha256-xfoRaizx48ldMG7GZZokQvuHKCH9OpuqN/hRUXb9GY8=";
          sparseCheckout = ["gguf-py"];
        };
        dependencies = (old.dependencies or []) ++ [final.requests];
      });
    };
  };
  buildInputs = with pkgs; [
    stdenv.cc.cc
    libuv
    zlib.dev
    libxml2
    cmake
  ];
in {
  env = {
    LD_LIBRARY_PATH = "${lib.makeLibraryPath buildInputs}";
    CMAKE_PREFIX_PATH = "${pkgs.zlib}:${pkgs.zlib.dev}:${pkgs.libxml2}:${pkgs.libxml2.dev}";
    CPATH = "${pkgs.zlib.dev}/include:${pkgs.libxml2.dev}/include";
    LIBRARY_PATH = "${pkgs.zlib}/lib:${pkgs.libxml2}/lib:${pkgs.zlib.dev}/lib:${pkgs.libxml2.dev}/lib";
  };
  packages = with pkgs; [
    zlib
  ];
  languages.python = {
    enable = true;
    uv = {
      enable = true;
      # sync.enable = true;
    };
    # package = py.withPackages (
    #   ps:
    #     with ps; [
    #       gguf
    #       torch
    #       huggingface-hub
    #       safetensors
    #       triton
    #     ]
    # );
  };
  languages.zig = {
    enable = true;
  };
}
