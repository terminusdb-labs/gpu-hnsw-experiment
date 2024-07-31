{ pkgs ? import <nixpkgs> {
  config = {
    allowUnfree = true;
    cudaEnabled = true;
  };
}
}:
pkgs.mkShell {
  buildInputs = with pkgs; [
    clang
    cudatoolkit
    cargo
    rustc
  ];
  shellHook = ''
      LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.cudatoolkit}/lib
      #export CUDA_PATH=${pkgs.cudatoolkit}
   '';

}
