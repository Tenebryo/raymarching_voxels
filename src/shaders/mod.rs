pub mod render_cs {
    vulkano_shaders::shader!{
        ty: "compute",
        path: "src/shaders/render.comp",
    }
}

pub mod update_cs {
  vulkano_shaders::shader!{
      ty: "compute",
      path: "src/shaders/update.comp",
  }
}

pub mod denoise_cs {
  vulkano_shaders::shader!{
      ty: "compute",
      path: "src/shaders/denoise.comp",
  }
}

// Push Constant Types
pub use render_cs::ty::RenderPushConstantData;
pub use update_cs::ty::UpdatePushConstantData;
pub use denoise_cs::ty::DenoisePushConstantData;

// Graphics Primitive Types
pub use render_cs::ty::VoxelChunk;
pub use render_cs::ty::Material;
pub use render_cs::ty::PointLight;