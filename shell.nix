{ pkgs ? import <nixpkgs> {
  config = {
    allowUnfree = true;
    cudaEnabled = true;
  };
}
}:
pkgs.mkShell {
  buildInputs = with pkgs; [
    clang_15
    cudatoolkit
    cargo
    rustc
  ];
  shellHook = ''
LD_LIBRARY_PATH=${pkgs.cudatoolkit}/lib:$LD_LIBRARY_PATH
if [ -e "/etc/NIXOS" ];then
  export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.cudatoolkit}/lib:$LD_LIBRARY_PATH
fi
export CUDA_PATH=${pkgs.cudatoolkit}
export NVCC_CCBIN=${pkgs.clang_15}/bin/clang++
   '';

}
