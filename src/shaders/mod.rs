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

// Push Constant Types
pub use render_cs::ty::RenderPushConstantData;
pub use update_cs::ty::UpdatePushConstantData;
pub use denoise_cs::ty::DenoisePushConstantData;
pub use reproject_cs::ty::ReprojectPushConstantData;
pub use intersect_cs::ty::IntersectPushConstants;

// Graphics Primitive Types
pub use render_cs::ty::Material;
pub use render_cs::ty::PointLight;

// Voxel Types
pub use intersect_cs::ty::VMaterial;

// these redefinition shenanigans are necessary because serde can't quite derive
// serialize/deserialize for types in another module that are used in Vec fields
use serde::{Serialize, Deserialize};
#[derive(Copy, Clone, Serialize, Deserialize, Debug)]
#[repr(C)]
pub struct VChildDescriptor {
    pub sub_voxels : [i32; 8],
}