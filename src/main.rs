#![allow(dead_code)]

mod timing;
mod shaders;
mod noise;
// mod vox;

use timing::Timing;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::descriptor::PipelineLayoutAbstract;
use vulkano::device::{Device, DeviceExtensions};
use vulkano::instance::{Instance, PhysicalDevice, QueueFamily};
use vulkano::pipeline::ComputePipeline;
use vulkano::sync::{GpuFuture, FlushError};
use vulkano::sync;

use vulkano::image::{SwapchainImage, StorageImage, ImmutableImage, Dimensions};
use vulkano::sampler::{Sampler, SamplerAddressMode, Filter, MipmapMode};
use vulkano::pipeline::viewport::Viewport;
use vulkano::swapchain::{AcquireError, PresentMode, SurfaceTransform, Swapchain, SwapchainCreationError, ColorSpace, FullscreenExclusive};
use vulkano::swapchain;
use vulkano::format::{Format, ClearValue};

use vulkano_win::VkSurfaceBuild;
use winit::window::{WindowBuilder, Window};
use winit::event_loop::{EventLoop, ControlFlow};
use winit::event::{Event, WindowEvent, VirtualKeyCode};

use winit_input_helper::WinitInputHelper;

use cgmath::{Vector3, Quaternion, InnerSpace, Rotation3, Rad, Rotation};

use png;

use std::io::Cursor;
use std::sync::Arc;
use std::time::Instant;
use std::io::stdout;

use crossterm::{
    ExecutableCommand,
    cursor::MoveUp
};

use opensimplex::OsnContext;

const BITS_PER_VOXEL : i32 = 16;
const VOXELS_PER_U32 : i32 = 32 / BITS_PER_VOXEL;

const BUFFER_FORMAT : Format = Format::R32G32B32A32Sfloat;

const NUM_BUFFERS : usize = 8;
const NUM_TEMP_IMAGES : usize = 2;

fn main() {

    //*************************************************************************************************************************************
    // Device Initialization
    //*************************************************************************************************************************************

    // As with other examples, the first step is to create an instance.
    let required_extensions = vulkano_win::required_extensions();
    let instance = Instance::new(None, &required_extensions, None).unwrap();

    // Choose which physical device to use.
    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();
    
    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .with_title("Voxel Renderer")
        .build_vk_surface(&event_loop, instance.clone()).unwrap();

    let compute_queue_family = physical.queue_families().find(|&q| q.supports_compute()).unwrap();
    // We take the first queue that supports drawing to our window.
    let graphics_queue_family = physical.queue_families().find(|&q| q.supports_graphics() && surface.is_supported(q).unwrap_or(false)).unwrap();
    // get the a transfer queue family. This will be used to transfer voxel data.
    let transfer_queue_family = physical.queue_families().find(|&q| q.explicitly_supports_transfers()).unwrap();

    // Now initializing the device.
    let (device, mut queues) = Device::new(physical, physical.supported_features(),
        &DeviceExtensions{khr_swapchain: true, khr_storage_buffer_storage_class:true, ..DeviceExtensions::none()},
        [(compute_queue_family, 0.5), (graphics_queue_family, 0.5), (transfer_queue_family, 0.25)].iter().cloned()).unwrap();

    // Since we can request multiple queues, the `queues` variable is in fact an iterator. In this
    // example we use only one queue, so we just retrieve the first and only element of the
    // iterator and throw it away.

    let compute_queue = queues.next().unwrap();
    let graphics_queue = queues.next().unwrap();

    println!("Device initialized");
    
    //*************************************************************************************************************************************
    // Compute Pipeline Creation
    //*************************************************************************************************************************************

    // build rendering compute pipeline
    let render_compute_pipeline = Arc::new({
        // raytracing shader
        use shaders::render_cs;

        use shaders::render_cs::SpecializationConstants;

        let spec_const = SpecializationConstants {
            BITS_PER_VOXEL
        };

        let shader = render_cs::Shader::load(device.clone()).unwrap();
        ComputePipeline::new(device.clone(), &shader.main_entry_point(), &spec_const).unwrap()
    });

    // build update compute pipeline
    let _update_compute_pipeline = Arc::new({
        // raytracing shader
        use shaders::update_cs;

        let shader = update_cs::Shader::load(device.clone()).unwrap();
        ComputePipeline::new(device.clone(), &shader.main_entry_point(), &()).unwrap()
    });

    // build denoise compute pipeline
    let denoise_compute_pipeline = Arc::new({
        // raytracing shader
        use shaders::denoise_cs;

        let shader = denoise_cs::Shader::load(device.clone()).unwrap();
        ComputePipeline::new(device.clone(), &shader.main_entry_point(), &()).unwrap()
    });

    // build denoise compute pipeline
    let reproject_compute_pipeline = Arc::new({
        // raytracing shader
        use shaders::reproject_cs;

        let shader = reproject_cs::Shader::load(device.clone()).unwrap();
        ComputePipeline::new(device.clone(), &shader.main_entry_point(), &()).unwrap()
    });

    // build denoise compute pipeline
    let accumulate_compute_pipeline = Arc::new({
        // raytracing shader
        use shaders::accumulate_cs;

        let shader = accumulate_cs::Shader::load(device.clone()).unwrap();
        ComputePipeline::new(device.clone(), &shader.main_entry_point(), &()).unwrap()
    });

    println!("Render Pipeline initialized");

    
    //*************************************************************************************************************************************
    // Screen Buffer Allocation
    //*************************************************************************************************************************************

    // build a swapchain compatible with the window surface we built earlier
    let (mut swapchain, mut swapchain_images) = {
        let caps = surface.capabilities(physical).unwrap();
        let usage = caps.supported_usage_flags;

        let alpha = caps.supported_composite_alpha.iter().next().unwrap();

        let format = caps.supported_formats[0].0;

        let dimensions: [u32; 2] = surface.window().inner_size().into();

        Swapchain::new(device.clone(), surface.clone(), caps.min_image_count, format,
            dimensions, 1, usage, &graphics_queue, SurfaceTransform::Identity, alpha,
            PresentMode::Fifo, FullscreenExclusive::Default, true, ColorSpace::SrgbNonLinear).unwrap()

    };

    println!("Swapchain initialized");


    let [width, height]: [u32; 2] = surface.window().inner_size().into();
    let (mut image_buffers, mut depth_buffers, mut tmp_images, mut screen_normals, mut screen_positions) = rebuild_intermediate_images(device.clone(), compute_queue_family.clone(), width, height);

    println!("Storage Images initialized");
    

    //*************************************************************************************************************************************
    // Constant Data
    //*************************************************************************************************************************************

    let lin_sampler = Sampler::new(device.clone(), Filter::Linear, Filter::Linear,
        MipmapMode::Linear, SamplerAddressMode::Repeat, SamplerAddressMode::Repeat,
        SamplerAddressMode::Repeat, 0.0, 1.0, 0.0, 0.0).unwrap();

    let nst_sampler = Sampler::new(device.clone(), Filter::Nearest, Filter::Nearest,
        MipmapMode::Nearest, SamplerAddressMode::Repeat, SamplerAddressMode::Repeat,
        SamplerAddressMode::Repeat, 0.0, 1.0, 0.0, 0.0).unwrap();

    let (blue_noise_tex, load_future) = {
        let png_bytes = include_bytes!("../data/LDR_RGBA_0.png").to_vec();
        let cursor = Cursor::new(png_bytes);
        let decoder = png::Decoder::new(cursor);
        let (info, mut reader) = decoder.read_info().unwrap();
        let dimensions = Dimensions::Dim2d { width: info.width, height: info.height };
        let mut image_data = Vec::new();
        image_data.resize((info.width * info.height * 4) as usize, 0);
        reader.next_frame(&mut image_data).unwrap();

        ImmutableImage::from_iter(
            image_data.iter().cloned(),
            dimensions,
            Format::R8G8B8A8Srgb,
            compute_queue.clone()
        ).unwrap()
    };
    
    //*************************************************************************************************************************************
    // Miscellaneous constants
    //*************************************************************************************************************************************

    let mut dynamic_state = DynamicState { line_width: None, viewports: None, scissors: None, compare_mask: None, write_mask: None, reference: None };

    update_dynamic_state(&swapchain_images, &mut dynamic_state);

    let mut recreate_swapchain = false;

    let mut previous_frame_end = Some(Box::new(load_future) as Box<dyn GpuFuture>);

    let mut fps = Timing::new(256);
    const FRAME_COUNTS : i32 = 144;
    let mut frame_counts = FRAME_COUNTS;
    let dimensions: [u32; 2] = surface.window().inner_size().into();
    let mut surface_width = dimensions[0];
    let mut surface_height = dimensions[1];
    let p_start = Instant::now();

    let mut input = WinitInputHelper::new();

    let mut forward = Vector3::new(0.0, 0.0, 1.0);
    let up = Vector3::new(0.0, 1.0, 0.0);
    let mut position = Vector3::new(0.0, 512.0, 0.0);
    let mut old_position = position;
    let mut input_time = Instant::now();
    let mut pitch = 0.0;
    let mut yaw = 0.0;

    let mut render_pc = shaders::RenderPushConstantData {
        cam_o : [0.0, 0.0, 0.0],
        cam_f : [0.0, 0.0, 1.0],
        cam_u : [0.0, 1.0, 0.0],
        vdim : [256, 256, 128],
        render_dist : 1064.0,
        time : p_start.elapsed().as_secs_f32(),
        gamma : 1.0,

        // dummy variables for alignment
        _dummy0 : [0;4],
        _dummy1 : [0;4],
    };

    let denoise_pc = shaders::DenoisePushConstantData {
        c_phi : 0.05,
        n_phi : 64.0,
        p_phi : 1.0,
        step_width : 1,
    };

    let mut reproject_pc = shaders::ReprojectPushConstantData {
        movement : [0.0; 3],
        old_forward : [forward.x, forward.y, forward.z],
        new_forward : [forward.x, forward.y, forward.z],
        up : [up.x, up.y, up.z],
        reproject_type : 0,

        _dummy0 : [0; 4],
        _dummy1 : [0; 4],
    };
    
    //*************************************************************************************************************************************
    // Voxel, Material, and Light Data Allocation and Generation
    //*************************************************************************************************************************************

    // build the voxel data buffer
    let voxel_data_buffer = unsafe {

        let num_voxels = render_pc.vdim[0] * render_pc.vdim[1] * render_pc.vdim[2];

        // CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, data_iter).unwrap()
        CpuAccessibleBuffer::<[u32]>::uninitialized_array(device.clone(), num_voxels as usize, BufferUsage::all(), false).unwrap()
    };

    println!("Voxel Buffer initialized");

    {
        let ctx = OsnContext::new(123).unwrap();

        let _ns = noise::WorleyNoise3D::new(8);
        

        let mut lock = voxel_data_buffer.write().unwrap();

        let mut index = [0,0,0];

        for uint in lock.iter_mut() {

            let mut data = 0;

            // for each uint in the buffer, there are VOXELS_PER_U32, so we have to
            // generate multiple voxels per uint.
            for i in 0..VOXELS_PER_U32 {

                let xn = index[0] as f64;
                let yn = index[1] as f64;
                let zn = index[2] as f64;

                let noise_scale = 64.0;

                let nx = ctx.noise3(xn / noise_scale, yn / noise_scale, zn / noise_scale);

                // let nx = ns.sample(xn / noise_scale, yn / noise_scale, zn / noise_scale);

                if nx > 0.0 {
                    data |= 1 << (BITS_PER_VOXEL * i);
                }

                index[0] += 1;
                if index[0] == render_pc.vdim[0] {
                    index[0] = 0;
                    index[1] += 1;
                    if index[1] == render_pc.vdim[1] {
                        index[1] = 0;
                        index[2] += 1;
                    }
                }
            }

            *uint = data;
        }
    }


    println!("Voxel Data initialized");

    // create a list of materials to render
    let material_data_buffer = {
        use shaders::Material;

        let materials = [
            // air material
            Material {albedo : [0.0; 3], transparency: 1.0, emission: [0.0; 3], flags: 0b00000000, roughness: 0.0, shininess: 0.3, _dummy0: [0;8]},
            // solid material
            Material {albedo : [1.0; 3], transparency: 0.0, emission: [0.0; 3], flags: 0b00000001, roughness: 0.0, shininess: 0.3, _dummy0: [0;8]}
        ];

        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, materials.iter().cloned()).unwrap()
    };

    
    println!("Material Data initialized");

    // create a list of materials to render
    let point_light_data_buffer = {
        use shaders::PointLight;

        let lights = [
            //sun
            PointLight {
                position : [0.0, 1000.0, 0.0],
                intensity : 1.0e5,
                color : [0.5, 1.0, 1.0],
                size : 5.0,
            }
        ];

        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, lights.iter().cloned()).unwrap()
    };

    //*************************************************************************************************************************************
    // Main Event Loop
    //*************************************************************************************************************************************
    
    println!("Light Data initialized");

    event_loop.run(move |event, _, control_flow| {

        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                *control_flow = ControlFlow::Exit;
            },
            Event::WindowEvent { event: WindowEvent::Resized(_), .. } => {
                recreate_swapchain = true;
            },
            Event::RedrawEventsCleared => {

                // start FPS sample
                fps.start_sample();
                
                previous_frame_end.as_mut().unwrap().cleanup_finished();

                // recreate the swapchain when the window is resized
                if recreate_swapchain {
                    // Get the new dimensions of the window.
                    let dimensions: [u32; 2] = surface.window().inner_size().into();
                    surface_width = dimensions[0];
                    surface_height = dimensions[1];
                    let (new_swapchain, new_images) = match swapchain.recreate_with_dimensions(dimensions) {
                        Ok(r) => r,
                        // This error tends to happen when the user is manually resizing the window.
                        // Simply restarting the loop is the easiest way to fix this issue.
                        Err(SwapchainCreationError::UnsupportedDimensions) => return,
                        Err(e) => panic!("Failed to recreate swapchain: {:?}", e)
                    };

                    swapchain = new_swapchain;
                    swapchain_images = new_images;

                    update_dynamic_state(&swapchain_images, &mut dynamic_state);

                    // re-allocate buffer images
                    let (new_image_buffers, new_depth_buffers, new_tmp_images, new_screen_normals, new_screen_positions) = 
                        rebuild_intermediate_images(device.clone(), compute_queue.family(), surface_width, surface_height);

                    image_buffers = new_image_buffers;
                    depth_buffers = new_depth_buffers;
                    tmp_images = new_tmp_images;
                    screen_normals = new_screen_normals;
                    screen_positions = new_screen_positions;

                    recreate_swapchain = false;
                }

                // get next swapchain image
                let (image_num, suboptimal, acquire_future) = match swapchain::acquire_next_image(swapchain.clone(), None) {
                    Ok(r) => r,
                    Err(AcquireError::OutOfDate) => {
                        recreate_swapchain = true;
                        return;
                    },
                    Err(e) => panic!("Failed to acquire next image: {:?}", e)
                };

                // detect when the swapchain needs to be recreated
                if suboptimal {
                    recreate_swapchain = true;
                }

                render_pc.time = p_start.elapsed().as_secs_f32();
                reproject_pc.new_forward = [forward.x, forward.y, forward.z];
                reproject_pc.movement = [position.x - old_position.x, position.y - old_position.y, position.z - old_position.z];
                old_position = position;

                // TODO: maybe cache DescriptorSets and modify them rather than construct them over and over
                // would need to rebuild them whenever the window is resized, since the referenced buffers are remade
                let render_layout = render_compute_pipeline.layout().descriptor_set_layout(0).unwrap();
                let render_set = Arc::new(PersistentDescriptorSet::start(render_layout.clone())
                    // .add_image(images[image_num].clone()).unwrap()
                    .add_image(image_buffers[0].clone()).unwrap()
                    .add_image(screen_normals.clone()).unwrap()
                    .add_image(screen_positions.clone()).unwrap()
                    .add_image(depth_buffers[0].clone()).unwrap()
                    // .add_image(blue_noise_tex.clone()).unwrap()
                    .add_sampled_image(blue_noise_tex.clone(), nst_sampler.clone()).unwrap()
                    .add_buffer(voxel_data_buffer.clone()).unwrap()
                    .add_buffer(material_data_buffer.clone()).unwrap()
                    .add_buffer(point_light_data_buffer.clone()).unwrap()
                    .build().unwrap()
                );

                let accumulate_layout = accumulate_compute_pipeline.layout().descriptor_set_layout(0).unwrap();
                let accumulate_set = Arc::new(PersistentDescriptorSet::start(accumulate_layout.clone())
                        .add_image(tmp_images[0].clone()).unwrap()
                        // TODO: make the buffer size dynamic
                        .enter_array().unwrap()
                            .add_image(image_buffers[0].clone()).unwrap()
                            .add_image(image_buffers[1].clone()).unwrap()
                            .add_image(image_buffers[2].clone()).unwrap()
                            .add_image(image_buffers[3].clone()).unwrap()
                            .add_image(image_buffers[4].clone()).unwrap()
                            .add_image(image_buffers[5].clone()).unwrap()
                            .add_image(image_buffers[6].clone()).unwrap()
                            .add_image(image_buffers[7].clone()).unwrap()
                        .leave_array().unwrap()
                        .build().unwrap()
                );
                
                let denoise_layout = denoise_compute_pipeline.layout().descriptor_set_layout(0).unwrap();
                // descriptor set for denoising iteration reading tmp_images[0] and writing tmp_images[1]
                let denoise_set_01 = Arc::new(PersistentDescriptorSet::start(denoise_layout.clone())
                    .add_image(tmp_images[0].clone()).unwrap()
                    .add_image(screen_normals.clone()).unwrap()
                    .add_image(screen_positions.clone()).unwrap()
                    .add_image(tmp_images[1].clone()).unwrap()
                    .build().unwrap()
                );
                // descriptor set for denoising iteration reading tmp_images[1] and writing tmp_images[0]
                let denoise_set_10 = Arc::new(PersistentDescriptorSet::start(denoise_layout.clone())
                    .add_image(tmp_images[1].clone()).unwrap()
                    .add_image(screen_normals.clone()).unwrap()
                    .add_image(screen_positions.clone()).unwrap()
                    .add_image(tmp_images[0].clone()).unwrap()
                    .build().unwrap()
                );
                // descriptor set for denoising iteration reading tmp_images[0] and writing to the next swapchain image
                let denoise_set_end = Arc::new(PersistentDescriptorSet::start(denoise_layout.clone())
                    .add_image(tmp_images[0].clone()).unwrap()
                    .add_image(screen_normals.clone()).unwrap()
                    .add_image(screen_positions.clone()).unwrap()
                    .add_image(swapchain_images[image_num].clone()).unwrap()
                    .build().unwrap()
                );

                //
                let reproject_layout = reproject_compute_pipeline.layout().descriptor_set_layout(0).unwrap();
                let reproject_sets = (0..(NUM_BUFFERS - 1)).rev().map(|i| (
                        i,
                        // in the reprojection stage, images first get
                        Arc::new(PersistentDescriptorSet::start(reproject_layout.clone())
                            .add_sampled_image(image_buffers[i].clone(), nst_sampler.clone()).unwrap()
                            .add_sampled_image(depth_buffers[i].clone(), nst_sampler.clone()).unwrap()
                            .add_image(tmp_images[0].clone()).unwrap()
                            .add_image(tmp_images[1].clone()).unwrap()
                            .build().unwrap()
                        ),
                        Arc::new(PersistentDescriptorSet::start(reproject_layout.clone())
                            .add_sampled_image(tmp_images[0].clone(), nst_sampler.clone()).unwrap()
                            .add_sampled_image(tmp_images[1].clone(), nst_sampler.clone()).unwrap()
                            .add_image(image_buffers[i + 1].clone()).unwrap()
                            .add_image(depth_buffers[i + 1].clone()).unwrap()
                            .build().unwrap()
                        )
                    ))
                    .collect::<Vec<_>>();

                use shaders::DenoisePushConstantData;
                use shaders::ReprojectPushConstantData;

                // we build a command buffer for this frame
                // needs to be built each frame because we don't know which swapchain image we will be told to render to
                // its possible a command buffer could be built for each swapchain ahead of time, but that would add complexity
                let mut raymarch_command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), compute_queue.family()).unwrap();

                // reproject old frames
                for (i, rproj_pos_set, rproj_rot_set) in reproject_sets {
                    raymarch_command_buffer = raymarch_command_buffer
                        .clear_color_image(image_buffers[i+1].clone(), ClearValue::Float([0.0; 4])).unwrap()
                        .clear_color_image(depth_buffers[i+1].clone(), ClearValue::Float([1.0e6; 4])).unwrap()
                        .dispatch([(surface_width - 1) / 32 + 1, (surface_height - 1) / 32 + 1, 1], reproject_compute_pipeline.clone(), rproj_pos_set.clone(), ReprojectPushConstantData{reproject_type: 1, ..reproject_pc}).unwrap()
                        .dispatch([(surface_width - 1) / 32 + 1, (surface_height - 1) / 32 + 1, 1], reproject_compute_pipeline.clone(), rproj_rot_set.clone(), ReprojectPushConstantData{reproject_type: 0, ..reproject_pc}).unwrap()

                }

                // render
                let raymarch_command_buffer = raymarch_command_buffer
                    .dispatch([(surface_width - 1) / 32 + 1, (surface_height - 1) / 32 + 1, 1], render_compute_pipeline.clone(), render_set.clone(), render_pc).unwrap()
                    .dispatch([(surface_width - 1) / 32 + 1, (surface_height - 1) / 32 + 1, 1], accumulate_compute_pipeline.clone(), accumulate_set.clone(), ()).unwrap()
                    .dispatch([(surface_width - 1) / 32 + 1, (surface_height - 1) / 32 + 1, 1], denoise_compute_pipeline.clone(), denoise_set_01.clone(),  DenoisePushConstantData{step_width : 2, ..denoise_pc}).unwrap()
                    .dispatch([(surface_width - 1) / 32 + 1, (surface_height - 1) / 32 + 1, 1], denoise_compute_pipeline.clone(), denoise_set_10.clone(),  DenoisePushConstantData{step_width : 1, ..denoise_pc}).unwrap()
                    .dispatch([(surface_width - 1) / 32 + 1, (surface_height - 1) / 32 + 1, 1], denoise_compute_pipeline.clone(), denoise_set_end.clone(), DenoisePushConstantData{step_width : 1, ..denoise_pc}).unwrap()
                    .build().unwrap();

                let future = previous_frame_end.take().unwrap()
                    .join(acquire_future)
                    // rendering is done in a compute shader
                    .then_execute(compute_queue.clone(), raymarch_command_buffer).unwrap()
                    // present the frame when rendering is complete
                    .then_swapchain_present(compute_queue.clone(), swapchain.clone(), image_num)
                    .then_signal_fence_and_flush();

                match future {
                    Ok(future) => {
                        previous_frame_end = Some(Box::new(future) as Box<_>);
                    },
                    Err(FlushError::OutOfDate) => {
                        recreate_swapchain = true;
                        previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<_>);
                    }
                    Err(e) => {
                        println!("Failed to flush future: {:?}", e);
                        previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<_>);
                    }
                }

                reproject_pc.old_forward = reproject_pc.new_forward;

                // FPS information
                fps.end_sample();

                frame_counts -= 1;
                if frame_counts <= 0 {
                    use std::f32::consts::PI;
                    frame_counts = FRAME_COUNTS;
                    let (t, t_var, fps, _) = fps.stats();

                    // print some debug information
                    stdout().execute(MoveUp(4)).unwrap();
                    println!("FPS: {:0<4.2} ({:0<5.3}ms +/- {:0<5.3}ms)  ", fps, t * 1000.0, t_var * 1000.0);
                    println!("Position: {:0<5.3?}                        ", position);
                    println!("Forward: {:0<5.3?}                         ", forward);
                    println!("P/Y: {:0<5.3}/{:0<5.3}                     ", 180.0 * pitch / PI, 180.0 * yaw / PI);
                }
            },
            _ => ()
        }
        
        // input handling
        if input.update(event) {
            let mut movement = Vector3::new(0.0, 0.0, 0.0);
            let left = up.cross(forward);
            let dt = input_time.elapsed().as_secs_f32();
            input_time = Instant::now();

            let mut speed = 10.0;

            if input.key_held(VirtualKeyCode::W) {movement += forward;}
            if input.key_held(VirtualKeyCode::A) {movement += left;}
            if input.key_held(VirtualKeyCode::S) {movement -= forward;}
            if input.key_held(VirtualKeyCode::D) {movement -= left;}
            if input.key_held(VirtualKeyCode::Space) {movement += up;}
            if input.key_held(VirtualKeyCode::LShift) {movement -= up;}
            if input.key_held(VirtualKeyCode::LControl) {speed = 100.0;}

            // ensure that movement on the diagonals isn't faster

            if movement.magnitude() > 1e-4 {movement = movement.normalize()};

            position += speed * dt *  movement;


            if let Some((_mx, _my)) = input.mouse() {

                // let mx = mx - surface_width as f32 / 2.0;
                // let my = my - surface_height as f32 / 2.0;

                if input.mouse_held(0) {
                    let (dx, dy) = input.mouse_diff();

                    pitch += dy / 270.0;
                    yaw   += dx / 270.0;

                    use std::f32::consts::PI;

                    if pitch < - PI / 2.0 { pitch = - PI / 2.0; }
                    if pitch > PI / 2.0 {pitch = PI / 2.0; }

                    // forward = Quaternion::from_sv(yaw, Vector3::unit_y()) * (Quaternion::from_sv(pitch, Vector3::unit_x()) * Vector3::unit_z());
                    let rot_p = Quaternion::from_angle_x(Rad(pitch));
                    let rot_y = Quaternion::from_angle_y(Rad(-yaw));
                    forward = rot_y.rotate_vector(rot_p.rotate_vector(Vector3::unit_z()));
                    forward = forward.normalize();
                }
            }

            render_pc.cam_o = [position.x, position.y, position.z];
            render_pc.cam_f = [forward.x, forward.y, forward.z];

            
            if input.key_held(VirtualKeyCode::Escape) {*control_flow = ControlFlow::Exit;}
        }
    });
}


/// This method is called once during initialization, then again whenever the window is resized
fn update_dynamic_state(
    images: &[Arc<SwapchainImage<Window>>],
    dynamic_state: &mut DynamicState
) {
    let dimensions = images[0].dimensions();

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [dimensions[0] as f32, dimensions[1] as f32],
        depth_range: 0.0 .. 1.0,
    };
    dynamic_state.viewports = Some(vec!(viewport));
}

// When the window is resized, the various screen-shaped buffers need to be resized
// this is done often and is a bit repetitive, so it is its own function
fn rebuild_intermediate_images(
    device : Arc<Device>, queue_family : QueueFamily, width : u32, height : u32
) -> (
    Vec<Arc<StorageImage<Format>>>,
    Vec<Arc<StorageImage<Format>>>,
    Vec<Arc<StorageImage<Format>>>,
    Arc<StorageImage<Format>>,
    Arc<StorageImage<Format>>
) {
    let image_buffers = (0..NUM_BUFFERS)
            .map(|_| { StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, BUFFER_FORMAT, [queue_family].iter().cloned()).unwrap() })
            .collect::<Vec<_>>();
    let depth_buffers = (0..NUM_BUFFERS)
            .map(|_| { StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, Format::R32Sfloat, [queue_family].iter().cloned()).unwrap() })
            .collect::<Vec<_>>();

    let tmp_images = (0..NUM_TEMP_IMAGES)
            .map(|_| { StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, BUFFER_FORMAT, [queue_family].iter().cloned()).unwrap() })
            .collect::<Vec<_>>();

    let screen_normals = StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, BUFFER_FORMAT, [queue_family].iter().cloned()).unwrap();
    let screen_positions = StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, BUFFER_FORMAT, [queue_family].iter().cloned()).unwrap();

    (image_buffers, depth_buffers, tmp_images, screen_normals, screen_positions)
}