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

pub use render_cs::ty::RenderPushConstantData;
pub use update_cs::ty::UpdatePushConstantData;

pub use render_cs::ty::VoxelChunk;
pub use render_cs::ty::Material;
pub use render_cs::ty::PointLight;