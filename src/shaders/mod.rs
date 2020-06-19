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

pub mod intersect_cs {
    vulkano_shaders::shader!{
        ty: "compute",
        path: "src/shaders/intersect.comp",
        include: ["src/shaders/"],
    }
}

pub mod pre_trace_cs {
    vulkano_shaders::shader!{
        ty: "compute",
        path: "src/shaders/pre_trace.comp",
        include: ["src/shaders/"],
    }
}

pub mod lighting_cs {
    vulkano_shaders::shader!{
        ty: "compute",
        path: "src/shaders/lighting.comp",
        include: ["src/shaders/"],
    }
}

// Push Constant Types
pub use render_cs::ty::RenderPushConstantData;
pub use update_cs::ty::UpdatePushConstantData;
pub use denoise_cs::ty::DenoisePushConstantData;
pub use reproject_cs::ty::ReprojectPushConstantData;
pub use intersect_cs::ty::IntersectPushConstants;
pub use pre_trace_cs::ty::PreTracePushConstants;
pub use lighting_cs::ty::LightingPushConstantData;

// Graphics Primitive Types
pub use lighting_cs::ty::BRDF;
pub use lighting_cs::ty::PointLight;
pub use lighting_cs::ty::DirectionalLight;
pub use lighting_cs::ty::SpotLight;
pub use lighting_cs::ty::Material;

// these redefinition shenanigans are necessary because serde can't quite derive
// serialize/deserialize for types in another module that are used in Vec fields
use serde::{Serialize, Deserialize};
#[derive(Copy, Clone, Serialize, Deserialize, Debug)]
#[repr(C)]
pub struct VChildDescriptor {
    pub sub_voxels : [i32; 8],
}