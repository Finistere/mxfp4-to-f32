{pkgs, ...}: let
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
in {
  packages = with pkgs; [];
  languages.python = {
    enable = true;
    package = py.withPackages (ps: with ps; [gguf]);
  };
  languages.zig = {
    enable = true;
  };
}
