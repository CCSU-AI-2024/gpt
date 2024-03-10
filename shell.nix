{ pkgs ? import <nixpkgs> { } }:

pkgs.mkShell {
  packages = with pkgs; [
    python3
    python311Packages.torch
    mypy
  ];
}