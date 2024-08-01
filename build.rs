extern crate cc;

fn main() {
    //panic!("{:?}", std::env::current_dir().unwrap());
    cc::Build::new()
        .cuda(true)
	.flag("--cubin")
	.flag("-allow-unsupported-compiler")
        .file("src/kernels/sin.cu")
        .compile("libsin.cubin");

    /* Link CUDA Runtime (libcudart.so) */
    println!("cargo:rustc-link-lib=cudart");
}
