pub mod cs {
    vulkano_shaders::shader!{
        ty: "compute",
        path: "src/shaders/march.comp",
    }
}

pub use cs::ty::PushConstantData;
pub use cs::ty::VoxelChunk;