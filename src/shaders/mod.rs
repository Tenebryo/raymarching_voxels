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

pub mod reproject_cs {
  vulkano_shaders::shader!{
      ty: "compute",
      path: "src/shaders/reproject.comp",
  }
}

pub mod accumulate_cs {
  vulkano_shaders::shader!{
      ty: "compute",
      path: "src/shaders/accumulate.comp",
  }
}

// Push Constant Types
pub use render_cs::ty::RenderPushConstantData;
pub use update_cs::ty::UpdatePushConstantData;
pub use denoise_cs::ty::DenoisePushConstantData;
pub use reproject_cs::ty::ReprojectPushConstantData;

// Graphics Primitive Types
pub use render_cs::ty::Material;
pub use render_cs::ty::PointLight;