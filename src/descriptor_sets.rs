use crate::pipelines::Pipelines;
use crate::gbuffer::GBuffer;
use crate::dbuffer::DBuffer;

use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::descriptor::descriptor_set::DescriptorSet;
use vulkano::descriptor::pipeline_layout::PipelineLayoutAbstract;

use std::sync::Arc;

pub struct DescriptorSets {
    pub reproject_set         : Arc<dyn DescriptorSet + Send + Sync>,
    pub stratified_sample_set : Arc<dyn DescriptorSet + Send + Sync>,
    pub intersect_set         : Arc<dyn DescriptorSet + Send + Sync>,
    pub pre_trace_set         : Arc<dyn DescriptorSet + Send + Sync>,
    pub light_bounce_set      : Arc<dyn DescriptorSet + Send + Sync>,
    pub light_combine_set     : Arc<dyn DescriptorSet + Send + Sync>,
    pub apply_texture_set     : Arc<dyn DescriptorSet + Send + Sync>,
    pub postprocess_set       : Arc<dyn DescriptorSet + Send + Sync>,
    pub atrous_set_a          : Arc<dyn DescriptorSet + Send + Sync>,
    pub atrous_set_b          : Arc<dyn DescriptorSet + Send + Sync>,
    pub light_occlude_set_0   : Arc<dyn DescriptorSet + Send + Sync>,
    pub light_occlude_set_1   : Arc<dyn DescriptorSet + Send + Sync>,
    pub sample_decay_set      : Arc<dyn DescriptorSet + Send + Sync>,
}

impl DescriptorSets {
    pub fn new(pipeline : &Pipelines, gbuffer : &GBuffer, dbuffer : &DBuffer) -> DescriptorSets {

        let reproject_layout = pipeline.reproject.layout().descriptor_set_layout(0).unwrap();
        let reproject_set = Arc::new(PersistentDescriptorSet::start(reproject_layout.clone())
            .add_image(gbuffer.position0_buffer.clone()).unwrap()
            .add_image(gbuffer.normal0_buffer.clone()).unwrap()
            // .add_image(gbuffer.postprocess_input_buffer.clone()).unwrap()
            .add_image(gbuffer.hdr_light_buffer.clone()).unwrap()
            // .add_image(gbuffer.prev_col_buffer.clone()).unwrap()
            .add_image(gbuffer.prev_pos_buffer.clone()).unwrap()
            .add_image(gbuffer.prev_cnt_buffer.clone()).unwrap()
            .add_image(gbuffer.reprojected_col_buffer.clone()).unwrap()
            .add_image(gbuffer.reprojected_pos_buffer.clone()).unwrap()
            .add_image(gbuffer.reprojected_cnt_buffer.clone()).unwrap()
            .add_image(gbuffer.reprojection_dist_buffer.clone()).unwrap()
            .build().unwrap()
        );

        let stratified_sample_layout = pipeline.stratified_sample.layout().descriptor_set_layout(0).unwrap();
        let stratified_sample_set = Arc::new(PersistentDescriptorSet::start(stratified_sample_layout.clone())
            .add_image(gbuffer.stratum_index_buffer.clone()).unwrap()
            .add_image(gbuffer.position0_buffer.clone()).unwrap()
            .add_image(gbuffer.stratum_pos_buffer.clone()).unwrap()
            .build().unwrap()
        );

        let sample_decay_layout = pipeline.sample_decay.layout().descriptor_set_layout(0).unwrap();
        let sample_decay_set = Arc::new(PersistentDescriptorSet::start(sample_decay_layout.clone())
            .add_image(gbuffer.reprojected_cnt_buffer.clone()).unwrap()
            .build().unwrap()
        );

        let intersect_layout = pipeline.intersect.layout().descriptor_set_layout(0).unwrap();
        let intersect_set = Arc::new(PersistentDescriptorSet::start(intersect_layout.clone())
            // initial depth buffer
            .add_image(gbuffer.pre_depth_buffer.clone()).unwrap()
            // normal buffer
            .add_image(gbuffer.normal0_buffer.clone()).unwrap()
            // position buffer
            .add_image(gbuffer.position0_buffer.clone()).unwrap()
            // depth buffer
            .add_image(gbuffer.depth_buffer.clone()).unwrap()
            // material buffer
            .add_image(gbuffer.material0_buffer.clone()).unwrap()
            // random seed buffer
            .add_image(gbuffer.noise_0_buffer.clone()).unwrap()
            .add_image(gbuffer.noise_1_buffer.clone()).unwrap()
            .add_image(gbuffer.noise_2_buffer.clone()).unwrap()
            .add_sampled_image(dbuffer.blue_noise_tex.clone(), dbuffer.nst_sampler.clone()).unwrap()
            .add_image(gbuffer.iteration_count_buffer.clone()).unwrap()
            .add_image(gbuffer.stratum_index_buffer.clone()).unwrap()
            .add_buffer(dbuffer.svdag_geometry_buffer.clone()).unwrap()
            .add_buffer(dbuffer.svdag_material_buffer.clone()).unwrap()
            .build().unwrap()
        );

        let pre_trace_layout = pipeline.pre_trace.layout().descriptor_set_layout(0).unwrap();
        let pre_trace_set = Arc::new(PersistentDescriptorSet::start(pre_trace_layout.clone())
            // depth buffer
            .add_image(gbuffer.pre_depth_buffer.clone()).unwrap()
            .add_buffer(dbuffer.svdag_geometry_buffer.clone()).unwrap()
            .add_buffer(dbuffer.svdag_material_buffer.clone()).unwrap()
            .build().unwrap()
        );

        let light_bounce_layout = pipeline.light_bounce.layout().descriptor_set_layout(0).unwrap();
        let light_bounce_set = Arc::new(PersistentDescriptorSet::start(light_bounce_layout.clone())
            .add_buffer(dbuffer.material_buffer.clone()).unwrap()
            .add_buffer(dbuffer.point_light_buffer.clone()).unwrap()
            .add_buffer(dbuffer.directional_light_buffer.clone()).unwrap()
            .add_buffer(dbuffer.spot_light_buffer.clone()).unwrap()
            // position buffers
            .add_image(gbuffer.position0_buffer.clone()).unwrap()
            .add_image(gbuffer.position1_buffer.clone()).unwrap()
            // normal buffers
            .add_image(gbuffer.normal0_buffer.clone()).unwrap()
            .add_image(gbuffer.normal1_buffer.clone()).unwrap()
            // depth buffer
            .add_image(gbuffer.depth_buffer.clone()).unwrap()
            // matrial buffers
            .add_image(gbuffer.material0_buffer.clone()).unwrap()
            .add_image(gbuffer.material1_buffer.clone()).unwrap()
            // random seed buffer
            .add_image(gbuffer.noise_0_buffer.clone()).unwrap()
            .add_image(gbuffer.noise_1_buffer.clone()).unwrap()
            .add_image(gbuffer.noise_2_buffer.clone()).unwrap()
            // light index buffer
            .add_image(gbuffer.light_index_buffer.clone()).unwrap()
            // light direction buffer
            .add_image(gbuffer.ldir0_buffer.clone()).unwrap()
            .add_image(gbuffer.ldir1_buffer.clone()).unwrap()
            // light value buffer
            .add_image(gbuffer.light0_buffer.clone()).unwrap()
            .add_image(gbuffer.light1_buffer.clone()).unwrap()
            .add_image(gbuffer.iteration_count_buffer.clone()).unwrap()
            // count and pixel masking buffers
            .add_image(gbuffer.reprojected_cnt_buffer.clone()).unwrap()
            .add_image(gbuffer.sample_mask_buffer.clone()).unwrap()
            // voxel data
            .add_buffer(dbuffer.svdag_geometry_buffer.clone()).unwrap()
            .add_buffer(dbuffer.svdag_material_buffer.clone()).unwrap()
            .build().unwrap()
        );

        let light_occlude_layout = pipeline.light_occlude.layout().descriptor_set_layout(0).unwrap();
        let light_occlude_set_0 = Arc::new(PersistentDescriptorSet::start(light_occlude_layout.clone())
            // material buffer
            .add_buffer(dbuffer.material_buffer.clone()).unwrap()
            // position buffers
            .add_image(gbuffer.position0_buffer.clone()).unwrap()
            // light direction buffer
            .add_image(gbuffer.ldir0_buffer.clone()).unwrap()
            // light value buffer
            .add_image(gbuffer.light0_buffer.clone()).unwrap()
            .add_image(gbuffer.iteration_count_buffer.clone()).unwrap()
            // count and pixel masking buffers
            .add_image(gbuffer.sample_mask_buffer.clone()).unwrap()
            // voxel data
            .add_buffer(dbuffer.svdag_geometry_buffer.clone()).unwrap()
            .add_buffer(dbuffer.svdag_material_buffer.clone()).unwrap()
            .build().unwrap()
        );
        let light_occlude_set_1 = Arc::new(PersistentDescriptorSet::start(light_occlude_layout.clone())
            // material buffer
            .add_buffer(dbuffer.material_buffer.clone()).unwrap()
            // position buffers
            .add_image(gbuffer.position1_buffer.clone()).unwrap()
            // light direction buffer
            .add_image(gbuffer.ldir1_buffer.clone()).unwrap()
            // light value buffer
            .add_image(gbuffer.light1_buffer.clone()).unwrap()
            .add_image(gbuffer.iteration_count_buffer.clone()).unwrap()
            // count and pixel masking buffers
            .add_image(gbuffer.sample_mask_buffer.clone()).unwrap()
            // voxel data
            .add_buffer(dbuffer.svdag_geometry_buffer.clone()).unwrap()
            .add_buffer(dbuffer.svdag_material_buffer.clone()).unwrap()
            .build().unwrap()
        );

        
        let light_combine_layout = pipeline.light_combine.layout().descriptor_set_layout(0).unwrap();
        let light_combine_set = Arc::new(PersistentDescriptorSet::start(light_combine_layout.clone())
            .add_buffer(dbuffer.material_buffer.clone()).unwrap()
            // position buffers
            .add_image(gbuffer.position0_buffer.clone()).unwrap()
            .add_image(gbuffer.position1_buffer.clone()).unwrap()
            // normal buffers
            .add_image(gbuffer.normal0_buffer.clone()).unwrap()
            .add_image(gbuffer.normal1_buffer.clone()).unwrap()
            // depth buffer
            .add_image(gbuffer.depth_buffer.clone()).unwrap()
            // material buffers
            .add_image(gbuffer.material0_buffer.clone()).unwrap()
            .add_image(gbuffer.material1_buffer.clone()).unwrap()
            // light index buffer
            .add_image(gbuffer.light_index_buffer.clone()).unwrap()
            // light direction buffer
            .add_image(gbuffer.ldir0_buffer.clone()).unwrap()
            .add_image(gbuffer.ldir1_buffer.clone()).unwrap()
            // light value buffer
            .add_image(gbuffer.light0_buffer.clone()).unwrap()
            .add_image(gbuffer.light1_buffer.clone()).unwrap()
            // output image
            .add_image(gbuffer.hdr_light_buffer.clone()).unwrap()
            // .add_image(swapchain_images[image_num].clone()).unwrap()
            // temporal integration buffers
            .add_image(gbuffer.reprojected_col_buffer.clone()).unwrap()
            .add_image(gbuffer.reprojected_pos_buffer.clone()).unwrap()
            .add_image(gbuffer.reprojected_cnt_buffer.clone()).unwrap()
            .add_image(gbuffer.prev_cnt_buffer.clone()).unwrap()
            .add_sampled_image(dbuffer.skysphere_tex.clone(), dbuffer.lin_sampler.clone()).unwrap()
            // count and pixel masking buffers
            .add_image(gbuffer.sample_mask_buffer.clone()).unwrap()
            // voxel data
            .add_buffer(dbuffer.svdag_geometry_buffer.clone()).unwrap()
            .add_buffer(dbuffer.svdag_material_buffer.clone()).unwrap()
            .build().unwrap()
        );

        let atrous_layout = pipeline.atrous.layout().descriptor_set_layout(0).unwrap();
        
        let atrous_set_a = Arc::new(PersistentDescriptorSet::start(atrous_layout.clone())
            .add_image(gbuffer.postprocess_input_buffer.clone()).unwrap()
            .add_image(gbuffer.position0_buffer.clone()).unwrap()
            .add_image(gbuffer.normal0_buffer.clone()).unwrap()
            .add_image(gbuffer.material0_buffer.clone()).unwrap()
            .add_image(gbuffer.prev_cnt_buffer.clone()).unwrap()
            .add_image(gbuffer.temp_buffers[0].clone()).unwrap()
            .build().unwrap()
        );
        let atrous_set_b = Arc::new(PersistentDescriptorSet::start(atrous_layout.clone())
            .add_image(gbuffer.temp_buffers[0].clone()).unwrap()
            .add_image(gbuffer.position0_buffer.clone()).unwrap()
            .add_image(gbuffer.normal0_buffer.clone()).unwrap()
            .add_image(gbuffer.material0_buffer.clone()).unwrap()
            .add_image(gbuffer.prev_cnt_buffer.clone()).unwrap()
            .add_image(gbuffer.postprocess_input_buffer.clone()).unwrap()
            .build().unwrap()
        );

        let apply_texture_layout = pipeline.apply_texture.layout().descriptor_set_layout(0).unwrap();
        let apply_texture_set = Arc::new(PersistentDescriptorSet::start(apply_texture_layout.clone())
            .add_image(gbuffer.postprocess_input_buffer.clone()).unwrap()
            .add_image(gbuffer.material0_buffer.clone()).unwrap()
            .add_buffer(dbuffer.material_buffer.clone()).unwrap()
            .build().unwrap()
        );


        let postprocess_layout = pipeline.postprocess.layout().descriptor_set_layout(0).unwrap();
        let postprocess_set = Arc::new(PersistentDescriptorSet::start(postprocess_layout.clone())
            .add_image(gbuffer.postprocess_input_buffer.clone()).unwrap()
            .add_image(gbuffer.output_buffer.clone()).unwrap()
            .add_image(gbuffer.luminance_buffer_t.clone()).unwrap()
            .build().unwrap()
        );

        DescriptorSets {
            reproject_set,
            stratified_sample_set,
            intersect_set,
            pre_trace_set,
            light_bounce_set,
            light_combine_set,
            apply_texture_set,
            postprocess_set,
            atrous_set_a,
            atrous_set_b,
            light_occlude_set_0,
            light_occlude_set_1,
            sample_decay_set,
        }
    }
}