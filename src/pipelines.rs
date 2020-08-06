
use vulkano::device::Device;
use vulkano::device::Queue;

use vulkano::pipeline::ComputePipeline;
use vulkano::pipeline::ComputePipelineAbstract;
use vulkano::descriptor::pipeline_layout::PipelineLayout;

use std::sync::Arc;

use crate::shaders;

pub struct Pipelines {
    pub _update           : Arc<ComputePipeline<PipelineLayout<shaders::update_cs::Layout>>>,
    pub _denoise          : Arc<ComputePipeline<PipelineLayout<shaders::denoise_cs::Layout>>>,
    pub reproject         : Arc<ComputePipeline<PipelineLayout<shaders::reproject_cs::Layout>>>,
    pub intersect         : Arc<ComputePipeline<PipelineLayout<shaders::intersect_cs::Layout>>>,
    pub pre_trace         : Arc<ComputePipeline<PipelineLayout<shaders::pre_trace_cs::Layout>>>,
    pub light_bounce      : Arc<ComputePipeline<PipelineLayout<shaders::light_bounce_cs::Layout>>>,
    pub light_occlude     : Arc<ComputePipeline<PipelineLayout<shaders::light_occlude_cs::Layout>>>,
    pub light_combine     : Arc<ComputePipeline<PipelineLayout<shaders::light_combine_cs::Layout>>>,
    pub normal_blend      : Arc<ComputePipeline<PipelineLayout<shaders::normal_blend_cs::Layout>>>,
    pub atrous            : Arc<ComputePipeline<PipelineLayout<shaders::atrous_cs::Layout>>>,
    pub apply_texture     : Arc<ComputePipeline<PipelineLayout<shaders::apply_texture_cs::Layout>>>,
    pub postprocess       : Arc<ComputePipeline<PipelineLayout<shaders::postprocess_cs::Layout>>>,
    pub stratified_sample : Arc<ComputePipeline<PipelineLayout<shaders::stratified_sample_cs::Layout>>>,
    pub sample_decay      : Arc<ComputePipeline<PipelineLayout<shaders::sample_decay_cs::Layout>>>,
    pub ray_test          : Arc<ComputePipeline<PipelineLayout<shaders::ray_test_cs::Layout>>>,
}

impl Pipelines {
    pub fn new(device : Arc<Device>, local_size_x : u32, local_size_y : u32) -> Pipelines {
        // build update compute pipeline
        let _update = Arc::new({
            // raytracing shader
            use shaders::update_cs;

            let shader = update_cs::Shader::load(device.clone()).unwrap();
            ComputePipeline::new(device.clone(), &shader.main_entry_point(), &()).unwrap()
        });

        // build denoise compute pipeline
        let _denoise = Arc::new({
            // raytracing shader
            use shaders::denoise_cs;

            let shader = denoise_cs::Shader::load(device.clone()).unwrap();
            ComputePipeline::new(device.clone(), &shader.main_entry_point(), &()).unwrap()
        });

        // build denoise compute pipeline
        let reproject = Arc::new({
            // raytracing shader
            use shaders::reproject_cs;

            let spec_consts = reproject_cs::SpecializationConstants{
                constant_1 : local_size_x,
                constant_2 : local_size_y,
            };

            let shader = reproject_cs::Shader::load(device.clone()).unwrap();
            ComputePipeline::new(device.clone(), &shader.main_entry_point(), &spec_consts).unwrap()
        });

        // build denoise compute pipeline
        let intersect = Arc::new({
            // raytracing shader
            use shaders::intersect_cs;

            let spec_consts = intersect_cs::SpecializationConstants{
                constant_1 : local_size_x,
                constant_2 : local_size_y,
            };

            let shader = intersect_cs::Shader::load(device.clone()).unwrap();
            ComputePipeline::new(device.clone(), &shader.main_entry_point(), &spec_consts).unwrap()
        });

        // build denoise compute pipeline
        let pre_trace = Arc::new({
            // raytracing shader
            use shaders::pre_trace_cs;

            let spec_consts = pre_trace_cs::SpecializationConstants{
                constant_1 : local_size_x,
                constant_2 : local_size_y,
            };

            let shader = pre_trace_cs::Shader::load(device.clone()).unwrap();
            ComputePipeline::new(device.clone(), &shader.main_entry_point(), &spec_consts).unwrap()
        });

        // build denoise compute pipeline
        let light_bounce = Arc::new({
            // raytracing shader
            use shaders::light_bounce_cs;

            let spec_consts = light_bounce_cs::SpecializationConstants{
                constant_1 : local_size_x,
                constant_2 : local_size_y,
            };

            let shader = light_bounce_cs::Shader::load(device.clone()).unwrap();
            ComputePipeline::new(device.clone(), &shader.main_entry_point(), &spec_consts).unwrap()
        });

        // build denoise compute pipeline
        let light_occlude = Arc::new({
            // raytracing shader
            use shaders::light_occlude_cs;

            let spec_consts = light_occlude_cs::SpecializationConstants{
                constant_1 : local_size_x,
                constant_2 : local_size_y,
            };

            let shader = light_occlude_cs::Shader::load(device.clone()).unwrap();
            ComputePipeline::new(device.clone(), &shader.main_entry_point(), &spec_consts).unwrap()
        });

        // build denoise compute pipeline
        let light_combine = Arc::new({
            // raytracing shader
            use shaders::light_combine_cs;

            let spec_consts = light_combine_cs::SpecializationConstants{
                constant_1 : local_size_x,
                constant_2 : local_size_y,
            };

            let shader = light_combine_cs::Shader::load(device.clone()).unwrap();
            ComputePipeline::new(device.clone(), &shader.main_entry_point(), &spec_consts).unwrap()
        });

        // build blending compute pipeline
        let normal_blend = Arc::new({
            // raytracing shader
            use shaders::normal_blend_cs;

            let spec_consts = normal_blend_cs::SpecializationConstants{
                PATCH_DIST : 1,
                constant_1 : local_size_x,
                constant_2 : local_size_y,
            };

            let shader = normal_blend_cs::Shader::load(device.clone()).unwrap();
            ComputePipeline::new(device.clone(), &shader.main_entry_point(), &spec_consts).unwrap()
        });

        let atrous = Arc::new({
            use shaders::atrous_cs;

            let spec_consts = atrous_cs::SpecializationConstants{
                constant_1 : local_size_x,
                constant_2 : local_size_y,
            };

            let shader = atrous_cs::Shader::load(device.clone()).unwrap();
            ComputePipeline::new(device.clone(), &shader.main_entry_point(), &spec_consts).unwrap()
        });

        let apply_texture = Arc::new({
            use shaders::apply_texture_cs;

            let spec_consts = apply_texture_cs::SpecializationConstants{
                constant_1 : local_size_x,
                constant_2 : local_size_y,
            };

            let shader = apply_texture_cs::Shader::load(device.clone()).unwrap();
            ComputePipeline::new(device.clone(), &shader.main_entry_point(), &spec_consts).unwrap()
        });

        let postprocess = Arc::new({
            use shaders::postprocess_cs;

            let spec_consts = postprocess_cs::SpecializationConstants{
                constant_1 : local_size_x,
                constant_2 : local_size_y,
            };

            let shader = postprocess_cs::Shader::load(device.clone()).unwrap();
            ComputePipeline::new(device.clone(), &shader.main_entry_point(), &spec_consts).unwrap()
        });


        let stratified_sample = Arc::new({
            use shaders::stratified_sample_cs;

            let spec_consts = stratified_sample_cs::SpecializationConstants{
                constant_1 : local_size_x,
                constant_2 : local_size_y,
            };

            let shader = stratified_sample_cs::Shader::load(device.clone()).unwrap();
            ComputePipeline::new(device.clone(), &shader.main_entry_point(), &spec_consts).unwrap()
        });


        let sample_decay = Arc::new({
            use shaders::sample_decay_cs;

            let spec_consts = sample_decay_cs::SpecializationConstants{
                constant_1 : local_size_x,
                constant_2 : local_size_y,
            };

            let shader = sample_decay_cs::Shader::load(device.clone()).unwrap();
            ComputePipeline::new(device.clone(), &shader.main_entry_point(), &spec_consts).unwrap()
        });
        
        let ray_test = Arc::new({
            use shaders::ray_test_cs;

            let spec_consts = ray_test_cs::SpecializationConstants{
                constant_1 : local_size_x * local_size_y,
            };

            let shader = ray_test_cs::Shader::load(device.clone()).unwrap();
            ComputePipeline::new(device.clone(), &shader.main_entry_point(), &spec_consts).unwrap()
        });

        Pipelines {
            _update,
            _denoise,
            reproject,
            intersect,
            pre_trace,
            light_bounce,
            light_occlude,
            light_combine,
            normal_blend,
            atrous,
            apply_texture,
            postprocess,
            stratified_sample,
            sample_decay,
            ray_test,
        }
    }
}