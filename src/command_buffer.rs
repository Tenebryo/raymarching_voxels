use vulkano::command_buffer::AutoCommandBufferBuilder;

use super::gbuffer::GBuffer;
use super::pipelines::Pipelines;
use super::shaders::PushConstantData;
use super::descriptor_sets::DescriptorSets;


pub fn build_rendering_command_buffer(
    cmd_buf : &mut AutoCommandBufferBuilder,
    pipeline : &Pipelines,
    desc_sets : &DescriptorSets,
    gbuffer : &GBuffer,
    pc : &PushConstantData,
    dim : (u32, u32),
    local_size : (u32, u32)
) {
    let (width, height) = dim;

    let (local_size_x, local_size_y) = local_size;

    let block_dim_x = (width - 1) / local_size_x + 1;
    let block_dim_y = (height - 1) / local_size_y + 1;

    cmd_buf
        .clear_color_image(gbuffer.reprojected_col_buffer.clone(), [0.0; 4].into()).unwrap()
        .clear_color_image(gbuffer.reprojected_pos_buffer.clone(), [0.0; 4].into()).unwrap()
        .clear_color_image(gbuffer.reprojected_cnt_buffer.clone(), [1;   4].into()).unwrap()
        .copy_image(
            gbuffer.position0_buffer.clone(), [0,0,0], 0, 0,
            gbuffer.prev_pos_buffer.clone(), [0,0,0], 0, 0,
            [width, height, 1], 1
        ).unwrap()
        .dispatch([(gbuffer.pre_trace_width - 1) / pc.sample_decay.patch_size + 1, (gbuffer.pre_trace_height - 1) / local_size_y + 1, 1], pipeline.pre_trace.clone(), desc_sets.pre_trace_set.clone(), pc.pre_trace).unwrap()
        .dispatch([block_dim_x, block_dim_y, 1], pipeline.intersect.clone(), desc_sets.intersect_set.clone(), pc.intersect).unwrap()
        .dispatch([block_dim_x, block_dim_y, 1], pipeline.stratified_sample.clone(), desc_sets.stratified_sample_set.clone(), pc.stratifiedsample).unwrap()
        // reproject previous frame
        .dispatch([block_dim_x, block_dim_y, 1], pipeline.reproject.clone(), desc_sets.reproject_set.clone(), pc.reproject).unwrap()
        // decay some reprojected samples
        .dispatch([block_dim_x / pc.sample_decay.patch_size, block_dim_y / pc.sample_decay.patch_size, 1], pipeline.sample_decay.clone(), desc_sets.sample_decay_set.clone(), pc.sample_decay).unwrap()
        // lighting calculations
        .dispatch([block_dim_x, block_dim_y, 1], pipeline.path_bounce.clone(), desc_sets.path_bounce.clone(), pc.path_bounce).unwrap()
        .dispatch([block_dim_x, block_dim_y, 1], pipeline.raycast.clone(), desc_sets.raycast_bounce.clone(), pc.raycast).unwrap()
        .dispatch([block_dim_x, block_dim_y, 1], pipeline.path_occlude.clone(), desc_sets.path_occlude_0.clone(), pc.path_occlude).unwrap()
        .dispatch([block_dim_x, block_dim_y, 1], pipeline.path_occlude.clone(), desc_sets.path_occlude_1.clone(), pc.path_occlude).unwrap()
        .dispatch([block_dim_x, block_dim_y, 1], pipeline.light_combine.clone(), desc_sets.light_combine_set.clone(), pc.light_combine).unwrap()
        .copy_image(
            gbuffer.hdr_light_buffer.clone(), [0,0,0], 0, 0,
            gbuffer.postprocess_input_buffer.clone(), [0,0,0], 0, 0,
            [width, height, 1], 1
        ).unwrap();
}