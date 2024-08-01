fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/*.cu");

    let out_dir = std::env::var("OUT_DIR").unwrap();

    let builder = bindgen_cuda::Builder::default();
    //println!("cargo:info={builder:?}");
    let bindings = builder.build_ptx().unwrap();
    bindings.write(format!("{out_dir}/kernels.rs")).unwrap();
}
