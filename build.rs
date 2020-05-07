fn main() {
    println!("cargo:rerun-if-changed=src/shaders/render.comp");
    println!("cargo:rerun-if-changed=src/shaders/update.comp");
    println!("cargo:rerun-if-changed=src/shaders/denoise.comp");
}