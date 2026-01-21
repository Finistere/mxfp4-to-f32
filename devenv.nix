{
  pkgs,
  lib,
  ...
}:
let
  buildInputs = with pkgs; [
    stdenv.cc.cc
    libuv
    zlib.dev
    libxml2
    cmake
  ];
in
{
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
    uv.enable = true;
  };
  languages.zig = {
    enable = true;
  };
}
