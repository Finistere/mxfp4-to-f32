{pkgs, ...}: {
  packages = with pkgs; [];
  languages.python = {
    enable = true;
    package = pkgs.python313.withPackages (ps: with ps; [gguf]);
  };
}
