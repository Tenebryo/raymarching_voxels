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

pub mod postprocess_cs {
    vulkano_shaders::shader!{
        ty: "compute",
        path: "src/shaders/postprocess.comp",
        include: [],
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
pub use postprocess_cs::ty::PostprocessPushConstantData;

// Graphics Primitive Types
// pub use light_bounce_cs::ty::BRDF;
pub use light_bounce_cs::ty::PointLight;
pub use light_bounce_cs::ty::DirectionalLight;
pub use light_bounce_cs::ty::SpotLight;
pub use light_bounce_cs::ty::Material;


use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::buffer::CpuAccessibleBuffer;
use vulkano::device::Device;
use vulkano::pipeline::ComputePipeline;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::descriptor::pipeline_layout::PipelineLayoutAbstract;
use std::sync::Arc;