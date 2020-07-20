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

pub mod light_bounce_cs {
    vulkano_shaders::shader!{
        ty: "compute",
        path: "src/shaders/light_bounce.comp",
        include: ["src/shaders/"],
    }
}

pub mod light_occlude_cs {
    vulkano_shaders::shader!{
        ty: "compute",
        path: "src/shaders/light_occlude.comp",
        include: ["src/shaders/"],
    }
}

pub mod light_combine_cs {
    vulkano_shaders::shader!{
        ty: "compute",
        path: "src/shaders/light_combine.comp",
        include: ["src/shaders/"],
    }
}

pub mod normal_blend_cs {
    vulkano_shaders::shader!{
        ty: "compute",
        path: "src/shaders/normal_blend.comp",
        include: ["src/shaders/"],
    }
}

// Push Constant Types
pub use update_cs::ty::UpdatePushConstantData;
pub use denoise_cs::ty::DenoisePushConstantData;
pub use reproject_cs::ty::ReprojectPushConstantData;
pub use intersect_cs::ty::IntersectPushConstants;
pub use pre_trace_cs::ty::PreTracePushConstants;
pub use light_bounce_cs::ty::LightBouncePushConstantData;
pub use light_occlude_cs::ty::LightOccludePushConstantData;
pub use light_combine_cs::ty::LightCombinePushConstantData;
pub use normal_blend_cs::ty::NormalBlendPushConstantData;

// Graphics Primitive Types
// pub use light_bounce_cs::ty::BRDF;
pub use light_bounce_cs::ty::PointLight;
pub use light_bounce_cs::ty::DirectionalLight;
pub use light_bounce_cs::ty::SpotLight;
pub use light_bounce_cs::ty::Material;

// these redefinition shenanigans are necessary because serde can't quite derive
// serialize/deserialize for types in another module that are used in Vec fields
use serde::{Serialize, Deserialize};
#[derive(Copy, Clone, Serialize, Deserialize, Debug)]
#[repr(C)]
pub struct VChildDescriptor {
    pub sub_voxels : [i32; 8],
}