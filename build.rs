fn main() {
    println!("cargo:rerun-if-changed=src/shaders/render.comp");
    println!("cargo:rerun-if-changed=src/shaders/update.comp");
    println!("cargo:rerun-if-changed=src/shaders/denoise.comp");
    println!("cargo:rerun-if-changed=src/shaders/reproject.comp");
    println!("cargo:rerun-if-changed=src/shaders/accumulate.comp");
    println!("cargo:rerun-if-changed=src/shaders/intersect.comp");
    println!("cargo:rerun-if-changed=src/shaders/voxel.comp");
}