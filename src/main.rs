use cudarc;
use cudarc::driver::LaunchAsync;
use cudarc::driver::LaunchConfig;
use std::include_str;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = cudarc::driver::CudaDevice::new(0)?;

    // allocate buffers
    let inp = dev.htod_copy(vec![1.0f32; 100])?;
    let mut out = dev.alloc_zeros::<f32>(100)?;

    let ptx = cudarc::nvrtc::compile_ptx(include_str!("kernels/sin.cu"))?;

    // and dynamically load it into the device
    dev.load_ptx(ptx, "my_module", &["sin_kernel"])?;

    let sin_kernel = dev.get_func("my_module", "sin_kernel").unwrap();
    let cfg = LaunchConfig::for_num_elems(100);
    unsafe { sin_kernel.launch(cfg, (&mut out, &inp, 100usize)) }?;

    let out_host: Vec<f32> = dev.dtoh_sync_copy(&out)?;
    assert_eq!(out_host, [1.0; 100].map(f32::sin));
    Ok(())
}
