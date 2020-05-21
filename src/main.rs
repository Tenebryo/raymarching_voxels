#![allow(dead_code)]

mod timing;
mod shaders;
mod noise;
mod gbuffer;
mod vox;

use timing::Timing;
use gbuffer::GBuffer;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState, sys::{UnsafeCommandBufferBuilder, Kind, Flags}, pool::StandardCommandPool};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::descriptor::PipelineLayoutAbstract;
use vulkano::device::{Device, DeviceExtensions};
use vulkano::instance::{Instance, PhysicalDevice, QueueFamily};
use vulkano::pipeline::ComputePipeline;
use vulkano::sync::{GpuFuture, FlushError, PipelineStages};
use vulkano::sync;
use vulkano::query::{QueryType, UnsafeQueryPool, UnsafeQuery};

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
        .with_maximized(true)
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
    let _render_compute_pipeline = Arc::new({
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
    let _denoise_compute_pipeline = Arc::new({
        // raytracing shader
        use shaders::denoise_cs;

        let shader = denoise_cs::Shader::load(device.clone()).unwrap();
        ComputePipeline::new(device.clone(), &shader.main_entry_point(), &()).unwrap()
    });

    // build denoise compute pipeline
    let _reproject_compute_pipeline = Arc::new({
        // raytracing shader
        use shaders::reproject_cs;

        let shader = reproject_cs::Shader::load(device.clone()).unwrap();
        ComputePipeline::new(device.clone(), &shader.main_entry_point(), &()).unwrap()
    });

    // build denoise compute pipeline
    let _accumulate_compute_pipeline = Arc::new({
        // raytracing shader
        use shaders::accumulate_cs;

        let shader = accumulate_cs::Shader::load(device.clone()).unwrap();
        ComputePipeline::new(device.clone(), &shader.main_entry_point(), &()).unwrap()
    });

    // build denoise compute pipeline
    let intersect_compute_pipeline = Arc::new({
        // raytracing shader
        use shaders::intersect_cs;

        let shader = intersect_cs::Shader::load(device.clone()).unwrap();
        ComputePipeline::new(device.clone(), &shader.main_entry_point(), &()).unwrap()
    });

    println!("Render Pipeline initialized");

    let timestamp_query_pool = Arc::new(UnsafeQueryPool::new(device.clone(), QueryType::Timestamp, 16).unwrap());

    let timestamp_command_pool = Arc::new(StandardCommandPool::new(device.clone(), compute_queue.family()));

    
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
    let mut gbuffer = GBuffer::new_buffers(device.clone(), compute_queue_family.clone(), width, height, 2);
    let mut prev_gbuffer = GBuffer::new_buffers(device.clone(), compute_queue_family.clone(), width, height, 0);

    println!("Storage Images initialized");
    

    //*************************************************************************************************************************************
    // Constant Data
    //*************************************************************************************************************************************

    let _lin_sampler = Sampler::new(device.clone(), Filter::Linear, Filter::Linear,
        MipmapMode::Linear, SamplerAddressMode::Repeat, SamplerAddressMode::Repeat,
        SamplerAddressMode::Repeat, 0.0, 1.0, 0.0, 0.0).unwrap();

    let _nst_sampler = Sampler::new(device.clone(), Filter::Nearest, Filter::Nearest,
        MipmapMode::Nearest, SamplerAddressMode::Repeat, SamplerAddressMode::Repeat,
        SamplerAddressMode::Repeat, 0.0, 1.0, 0.0, 0.0).unwrap();

    let (_blue_noise_tex, load_future) = {
        let png_bytes = [
            include_bytes!("../data/blue_noise/HDR_RGBA_0.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_1.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_2.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_3.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_4.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_5.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_6.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_7.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_8.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_9.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_10.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_11.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_12.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_13.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_14.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_15.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_16.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_17.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_18.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_19.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_20.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_21.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_22.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_23.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_24.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_25.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_26.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_27.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_28.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_29.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_30.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_31.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_32.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_33.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_34.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_35.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_36.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_37.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_38.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_39.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_40.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_41.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_42.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_43.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_44.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_45.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_46.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_47.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_48.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_49.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_50.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_51.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_52.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_53.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_54.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_55.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_56.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_57.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_58.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_59.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_60.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_61.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_62.png").to_vec(),
            include_bytes!("../data/blue_noise/HDR_RGBA_63.png").to_vec()
        ];

        let mut image_data = Vec::new();
        image_data.resize((64 * 64 * 64 * 4) as usize, 0);
        let dimensions = Dimensions::Dim3d { width: 64, height: 64, depth: png_bytes.len() as u32 };
        for i in 0..64 {
            let cursor = Cursor::new(png_bytes[i].clone());
            let decoder = png::Decoder::new(cursor);
            let (_info, mut reader) = decoder.read_info().unwrap();
            // if let Dimensions::Dim3d{ref mut width, ref mut height, ..} = dimensions {
            //     *width = info.width;
            //     *height = info.height;
            // }
            reader.next_frame(&mut image_data[(64 * 64 * 4 * i)..(64 * 64 * 4 * (i+1))]).unwrap();
        }

        println!("{:?}", dimensions);

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

    let mut forward = Vector3::new(1.0, 1.0, 1.0).normalize();
    let up = Vector3::new(0.0, 1.0, 0.0);
    let mut position = 0.5f32 * Vector3::new(-1.0, -1.0, -1.0);
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
        n_point_lights: 1,

        // dummy variables for alignment
        // _dummy0 : [0;4],
        // _dummy1 : [0;4],
    };

    let _denoise_pc = shaders::DenoisePushConstantData {
        c_phi : 0.05,
        n_phi : 64.0,
        p_phi : 1.0,
        step_width : 1,
    };

    let mut intersect_pc = shaders::IntersectPushConstants {
        camera_origin : [position.x, position.y, position.z],
        camera_forward : [forward.x, forward.y, forward.z],
        camera_up : [up.x, up.y, up.z],

        _dummy0 : [0;4],
        _dummy1 : [0;4],
    };

    // let mut reproject_pc = shaders::ReprojectPushConstantData {
    //     movement : [0.0; 3],
    //     old_forward : [forward.x, forward.y, forward.z],
    //     new_forward : [forward.x, forward.y, forward.z],
    //     up : [up.x, up.y, up.z],
    //     reproject_type : 0,

    //     _dummy0 : [0; 4],
    //     _dummy1 : [0; 4],
    // };
    
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

    let svdag_geometry_data = {

        let chunk_bytes = include_bytes!("../data/bunny.svdag");

        bincode::deserialize::<vox::VoxelChunk>(chunk_bytes).expect("Deserialization Failed")
    };

    let svdag_geometry_buffer = {
        CpuAccessibleBuffer::<[shaders::VChildDescriptor]>::from_iter(device.clone(), BufferUsage::all(), false, svdag_geometry_data.voxels.iter().cloned()).unwrap()
    };

    let svdag_material_buffer = {

        let materials = [
            // solid material
            shaders::VMaterial{
                color : [1.0; 3],
                shininess : 1.0,
                emission : [0.0; 3],
                _dummy0 : [0;4],
            },
        ];

        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, materials.iter().cloned()).unwrap()
    };


    println!("Voxel Data initialized");

    // create a list of materials to render
    let _material_data_buffer = {
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
    let _point_light_data_buffer = {
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

    println!("");
    println!("");
    println!("");
    println!("");

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

                    gbuffer = GBuffer::new_buffers(device.clone(), compute_queue.family(), surface_width, surface_height, 2);
                    prev_gbuffer = GBuffer::new_buffers(device.clone(), compute_queue.family(), surface_width, surface_height, 2);

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
                // reproject_pc.new_forward = [forward.x, forward.y, forward.z];
                
                intersect_pc.camera_forward = [forward.x, forward.y, forward.z];
                intersect_pc.camera_origin = [position.x, position.y, position.z];
                intersect_pc.camera_up = [up.x, up.y, up.z];
                // reproject_pc.movement = [position.x - old_position.x, position.y - old_position.y, position.z - old_position.z];
                old_position = position;

                let intersect_layout = intersect_compute_pipeline.layout().descriptor_set_layout(0).unwrap();
                let intersect_set = Arc::new(PersistentDescriptorSet::start(intersect_layout.clone())
                    // normal buffer
                    .add_image(gbuffer.normal_buffer.clone()).unwrap()
                    // position buffer
                    .add_image(swapchain_images[image_num].clone()).unwrap()
                    // depth buffer
                    .add_image(gbuffer.depth_buffer.clone()).unwrap()
                    // voxel index buffer
                    .add_image(gbuffer.index_buffer.clone()).unwrap()
                    .add_buffer(svdag_geometry_buffer.clone()).unwrap()
                    .add_buffer(svdag_material_buffer.clone()).unwrap()
                    .build().unwrap()
                );

                // we build a command buffer for this frame
                // needs to be built each frame because we don't know which swapchain image we will be told to render to
                // its possible a command buffer could be built for each swapchain ahead of time, but that would add complexity
                let render_command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), compute_queue.family()).unwrap();

                // render
                let render_command_buffer = render_command_buffer
                    .dispatch([(surface_width - 1) / 32 + 1, (surface_height - 1) / 32 + 1, 1], intersect_compute_pipeline.clone(), intersect_set.clone(), intersect_pc).unwrap()
                    .dispatch([(surface_width - 1) / 32 + 1, (surface_height - 1) / 32 + 1, 1], intersect_compute_pipeline.clone(), intersect_set.clone(), intersect_pc).unwrap()
                    .dispatch([(surface_width - 1) / 32 + 1, (surface_height - 1) / 32 + 1, 1], intersect_compute_pipeline.clone(), intersect_set.clone(), intersect_pc).unwrap()
                    .dispatch([(surface_width - 1) / 32 + 1, (surface_height - 1) / 32 + 1, 1], intersect_compute_pipeline.clone(), intersect_set.clone(), intersect_pc).unwrap()
                    .dispatch([(surface_width - 1) / 32 + 1, (surface_height - 1) / 32 + 1, 1], intersect_compute_pipeline.clone(), intersect_set.clone(), intersect_pc).unwrap()
                    .dispatch([(surface_width - 1) / 32 + 1, (surface_height - 1) / 32 + 1, 1], intersect_compute_pipeline.clone(), intersect_set.clone(), intersect_pc).unwrap()
                    .dispatch([(surface_width - 1) / 32 + 1, (surface_height - 1) / 32 + 1, 1], intersect_compute_pipeline.clone(), intersect_set.clone(), intersect_pc).unwrap()
                    .dispatch([(surface_width - 1) / 32 + 1, (surface_height - 1) / 32 + 1, 1], intersect_compute_pipeline.clone(), intersect_set.clone(), intersect_pc).unwrap()
                    .build().unwrap();

                let future = previous_frame_end.take().unwrap()
                    .join(acquire_future)
                    // rendering is done in a compute shader
                    .then_execute(compute_queue.clone(), render_command_buffer).unwrap()
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

                // reproject_pc.old_forward = reproject_pc.new_forward;

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

            let mut speed = 0.5;

            if input.key_held(VirtualKeyCode::W) {movement += forward;}
            if input.key_held(VirtualKeyCode::A) {movement += left;}
            if input.key_held(VirtualKeyCode::S) {movement -= forward;}
            if input.key_held(VirtualKeyCode::D) {movement -= left;}
            if input.key_held(VirtualKeyCode::Space) {movement += up;}
            if input.key_held(VirtualKeyCode::LShift) {movement -= up;}
            if input.key_held(VirtualKeyCode::LControl) {speed = 2.0;}
            if input.key_held(VirtualKeyCode::LAlt) {speed = 0.1;}

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

