#![allow(dead_code, unused_imports)]

extern crate nalgebra as na;

mod timing;
mod shaders;
mod noise;
mod gbuffer;
mod vox;
mod brdf;

use timing::Timing;
use gbuffer::GBuffer;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, DeviceLocalBuffer};
use vulkano::command_buffer::{StateCacher, AutoCommandBufferBuilder, DynamicState, sys::{UnsafeCommandBufferBuilderPipelineBarrier, UnsafeCommandBufferBuilder, Kind, Flags}, pool::StandardCommandPool};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::descriptor::PipelineLayoutAbstract;
use vulkano::device::{Device, DeviceExtensions};
use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice, QueueFamily};
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
use std::path::*;

use rand::prelude::*;

use crossterm::{
    ExecutableCommand,
    cursor::MoveUp
};

fn main() {

    //*************************************************************************************************************************************
    // Device Initialization
    //*************************************************************************************************************************************

    // As with other examples, the first step is to create an instance.
    let required_extensions = InstanceExtensions{
        // needed to get physical device metadata
        khr_get_physical_device_properties2: true,
        ..vulkano_win::required_extensions()
    };
    let instance = Instance::new(None, &required_extensions, None).unwrap();

    // Choose which physical device to use.
    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();
    
    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .with_title("Voxel Renderer")
        .with_maximized(true)
        .build_vk_surface(&event_loop, instance.clone()).unwrap();

    surface.window().set_maximized(true);

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

    let (local_size_x, local_size_y) = match physical.extended_properties().subgroup_size() {
        Some(subgroup_size) => {
            println!(
                "Subgroup size for '{}' device is {}",
                physical.name(),
                subgroup_size
            );

            // Most of the subgroup values are divisors of 8
            (8, subgroup_size / 8)
        }
        None => {
            println!("This Vulkan driver doesn't provide physical device Subgroup information");

            // Using fallback constant
            (8, 8)
        }
    };

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

        let spec_consts = intersect_cs::SpecializationConstants{
            constant_1 : local_size_x,
            constant_2 : local_size_y,
        };

        let shader = intersect_cs::Shader::load(device.clone()).unwrap();
        ComputePipeline::new(device.clone(), &shader.main_entry_point(), &spec_consts).unwrap()
    });

    // build denoise compute pipeline
    let pre_trace_compute_pipeline = Arc::new({
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
    let light_bounce_compute_pipeline = Arc::new({
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
    let light_occlude_compute_pipeline = Arc::new({
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
    let light_combine_compute_pipeline = Arc::new({
        // raytracing shader
        use shaders::light_combine_cs;

        let spec_consts = light_combine_cs::SpecializationConstants{
            constant_1 : local_size_x,
            constant_2 : local_size_y,
        };

        let shader = light_combine_cs::Shader::load(device.clone()).unwrap();
        ComputePipeline::new(device.clone(), &shader.main_entry_point(), &spec_consts).unwrap()
    });

    println!("Render Pipeline initialized");

    let _timestamp_query_pool = Arc::new(UnsafeQueryPool::new(device.clone(), QueryType::Timestamp, 16).unwrap());

    let _timestamp_command_pool = Arc::new(StandardCommandPool::new(device.clone(), compute_queue.family()));


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

    let nst_sampler = Sampler::new(device.clone(), Filter::Nearest, Filter::Nearest,
        MipmapMode::Nearest, SamplerAddressMode::Repeat, SamplerAddressMode::Repeat,
        SamplerAddressMode::Repeat, 0.0, 1.0, 0.0, 0.0).unwrap();

    let (blue_noise_tex, load_future) = {
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

        let mut image_data : Vec<u8> = Vec::new();
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
    let mut position = 0.1f32 * Vector3::new(-1.0, -1.0, -1.0);
    let mut old_position = position;
    let mut input_time = Instant::now();
    let mut pitch = 0.0;
    let mut yaw = 0.0;

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
        max_depth : 16,
        render_dist: 100.0,
        frame_idx : 0,
        noise_idx : 0,
        noise_frames : 64,
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

    // parse voxel goemetry data
    let mut svdag_geometry_data = {

        let chunk_bytes = include_bytes!("../data/bunny.svdag");

        bincode::deserialize::<vox::VoxelChunk>(chunk_bytes).expect("Deserialization Failed")
    };

    svdag_geometry_data.multiply_root_by_8();
    svdag_geometry_data.multiply_root_by_8();
    svdag_geometry_data.multiply_root_by_8();
    svdag_geometry_data.multiply_root_by_8();

    // calculate the lod materials
    svdag_geometry_data.calculate_lod_materials();

    // load the voxel data onto the GPU
    let svdag_geometry_buffer = {
        CpuAccessibleBuffer::<[shaders::VChildDescriptor]>::from_iter(device.clone(), BufferUsage::all(), false, svdag_geometry_data.voxels.iter().cloned()).unwrap()
    };

    let svdag_material_buffer = {
        CpuAccessibleBuffer::<[u32]>::from_iter(device.clone(), BufferUsage::all(), false, svdag_geometry_data.lod_materials.iter().cloned()).unwrap()
    };


    println!("Voxel Data iniitialized");

    // create a list of point lights to render
    let point_light_buffer = {
        use shaders::PointLight;

        let lights = [
            //sun
            PointLight {
                position : [0.0, 1000.0, 0.0],
                power : 1.0e5,
                color : [0.5, 1.0, 1.0],
                radius : 0.0,
            }
        ];

        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, lights.iter().cloned()).unwrap()
    };
    // create a list of directional lights to render
    let directional_light_buffer = {
        use shaders::DirectionalLight;

        let lights = [
            //sun
            DirectionalLight {
                direction : [0.0, -1.0, 0.0],
                color : [0.8; 3],
                _dummy0 : [0;4],
                _dummy1 : [0;4],
            }
        ];

        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, lights.iter().cloned()).unwrap()
    };
    // create a list of spot lights to render
    let spot_light_buffer = {
        use shaders::SpotLight;

        let lights = [
            //sun
            SpotLight {
                direction : [1.0, 0.0, 0.0],
                position : [0.0; 3],
                half_angle : 0.0,
                power : 0.0,
                color : [1.0, 1.0, 1.0],
                _dummy0 : [0;4],
            }
        ];

        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, lights.iter().cloned()).unwrap()
    };

    println!("Created Light Buffers");
    
    // let basic_brdf = brdf::BRDF::read_matusik_brdf_file(Path::new("data/teflon.bin"), [90, 90, 90, 90], 2, 2).expect("could not load brdf file");
    let basic_brdf = bincode::deserialize::<brdf::BRDF>(include_bytes!("../data/teflon-16-16-128-32.brdf")).unwrap();
    // create the brdf buffer
    let brdf_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, (0..(basic_brdf.len())).map(|_| 0.0f32)).unwrap();

    basic_brdf.write_to_buffer(&mut brdf_buffer.write().unwrap(), 0);


    // create a list of materials to render
    let material_buffer = {
        use shaders::Material;

        let materials = [
            // air material
            // Material {brdf : basic_brdf.create_shader_type(0), albedo : [0.0; 3], transparency: 1.0, emission: [0.0; 3], flags: 0b00000000, roughness: 0.0, shininess: 0.3, _dummy0: [0;8]},
            Material {brdf : 0, albedo : [0.0; 3], transparency: 1.0, emission: [0.0; 3], flags: 0b00000000, roughness: 0.0, shininess: 0.3, _dummy0: [0;12], _dummy1: [0;8]},
            // solid material
            // Material {brdf : basic_brdf.create_shader_type(0), albedo : [1.0; 3], transparency: 0.0, emission: [0.0; 3], flags: 0b00000001, roughness: 0.0, shininess: 0.3, _dummy0: [0;8]}
            Material {brdf : 0, albedo : [1.0; 3], transparency: 0.0, emission: [0.0; 3], flags: 0b00000001, roughness: 0.0, shininess: 0.3, _dummy0: [0;12], _dummy1: [0;8]}
        ];

        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, materials.iter().cloned()).unwrap()
    };

    
    println!("Created BRDF Buffer");
    println!("Material Data initialized");


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

                let time = p_start.elapsed().as_secs_f32();

                if let Ok(mut wlock) = directional_light_buffer.write() {
                    wlock[0].direction = [time.sin(), time.cos(), 0.0];
                }

                // reproject_pc.new_forward = [forward.x, forward.y, forward.z];
                
                intersect_pc.camera_forward = [forward.x, forward.y, forward.z];
                intersect_pc.camera_origin = [position.x, position.y, position.z];
                intersect_pc.camera_up = [up.x, up.y, up.z];
                intersect_pc.frame_idx += 1;
                intersect_pc.noise_idx = thread_rng().gen_range(0,intersect_pc.noise_frames);
                // reproject_pc.movement = [position.x - old_position.x, position.y - old_position.y, position.z - old_position.z];
                old_position = position;

                let intersect_layout = intersect_compute_pipeline.layout().descriptor_set_layout(0).unwrap();
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
                    .add_image(gbuffer.rng_seed_buffer.clone()).unwrap()
                    .add_sampled_image(blue_noise_tex.clone(), nst_sampler.clone()).unwrap()
                    .add_buffer(svdag_geometry_buffer.clone()).unwrap()
                    .add_buffer(svdag_material_buffer.clone()).unwrap()
                    .build().unwrap()
                );

                let pre_trace_pc = shaders::PreTracePushConstants {
                    camera_forward : intersect_pc.camera_forward,
                    camera_origin : intersect_pc.camera_origin,
                    camera_up : intersect_pc.camera_up,
                    max_depth : 5,

                    _dummy0 : [0;4],
                    _dummy1 : [0;4],
                };

                let pre_trace_layout = pre_trace_compute_pipeline.layout().descriptor_set_layout(0).unwrap();
                let pre_trace_set = Arc::new(PersistentDescriptorSet::start(pre_trace_layout.clone())
                    // depth buffer
                    .add_image(gbuffer.pre_depth_buffer.clone()).unwrap()
                    .add_buffer(svdag_geometry_buffer.clone()).unwrap()
                    .add_buffer(svdag_material_buffer.clone()).unwrap()
                    .build().unwrap()
                );


                let light_bounce_layout = light_bounce_compute_pipeline.layout().descriptor_set_layout(0).unwrap();
                let light_bounce_set = Arc::new(PersistentDescriptorSet::start(light_bounce_layout.clone())
                    .add_buffer(material_buffer.clone()).unwrap()
                    .add_buffer(point_light_buffer.clone()).unwrap()
                    .add_buffer(directional_light_buffer.clone()).unwrap()
                    .add_buffer(spot_light_buffer.clone()).unwrap()
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
                    .add_image(gbuffer.rng_seed_buffer.clone()).unwrap()
                    // light index buffer
                    .add_image(gbuffer.light_index_buffer.clone()).unwrap()
                    // light direction buffer
                    .add_image(gbuffer.ldir0_buffer.clone()).unwrap()
                    .add_image(gbuffer.ldir1_buffer.clone()).unwrap()
                    // light value buffer
                    .add_image(gbuffer.light0_buffer.clone()).unwrap()
                    .add_image(gbuffer.light1_buffer.clone()).unwrap()
                    // voxel data
                    .add_buffer(svdag_geometry_buffer.clone()).unwrap()
                    .add_buffer(svdag_material_buffer.clone()).unwrap()
                    .build().unwrap()
                );

                let light_bounce_pc = shaders::LightBouncePushConstantData {
                    camera_forward : pre_trace_pc.camera_forward,
                    camera_up : pre_trace_pc.camera_up,
                    camera_origin : pre_trace_pc.camera_origin,
                    n_directional_lights : 1,
                    n_point_lights : 0,
                    n_spot_lights : 0,
                    render_dist : intersect_pc.render_dist,
                    max_depth : 8,
                };
                
                let light_occlude_layout = light_occlude_compute_pipeline.layout().descriptor_set_layout(0).unwrap();
                let light_occlude_set_0 = Arc::new(PersistentDescriptorSet::start(light_occlude_layout.clone())
                    // material buffer
                    .add_buffer(material_buffer.clone()).unwrap()
                    // position buffers
                    .add_image(gbuffer.position0_buffer.clone()).unwrap()
                    // light direction buffer
                    .add_image(gbuffer.ldir0_buffer.clone()).unwrap()
                    // light value buffer
                    .add_image(gbuffer.light0_buffer.clone()).unwrap()
                    // voxel data
                    .add_buffer(svdag_geometry_buffer.clone()).unwrap()
                    .add_buffer(svdag_material_buffer.clone()).unwrap()
                    .build().unwrap()
                );
                let light_occlude_set_1 = Arc::new(PersistentDescriptorSet::start(light_occlude_layout.clone())
                    // material buffer
                    .add_buffer(material_buffer.clone()).unwrap()
                    // position buffers
                    .add_image(gbuffer.position1_buffer.clone()).unwrap()
                    // light direction buffer
                    .add_image(gbuffer.ldir1_buffer.clone()).unwrap()
                    // light value buffer
                    .add_image(gbuffer.light1_buffer.clone()).unwrap()
                    // voxel data
                    .add_buffer(svdag_geometry_buffer.clone()).unwrap()
                    .add_buffer(svdag_material_buffer.clone()).unwrap()
                    .build().unwrap()
                );

                let light_occlude_pc = shaders::LightOccludePushConstantData {
                    render_dist : intersect_pc.render_dist,
                    num_materials : 2,
                    max_depth : 8,
                };

                
                let light_combine_layout = light_combine_compute_pipeline.layout().descriptor_set_layout(0).unwrap();
                let light_combine_set = Arc::new(PersistentDescriptorSet::start(light_combine_layout.clone())
                    .add_buffer(material_buffer.clone()).unwrap()
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
                    // random seed buffer
                    .add_image(gbuffer.rng_seed_buffer.clone()).unwrap()
                    // light index buffer
                    .add_image(gbuffer.light_index_buffer.clone()).unwrap()
                    // light direction buffer
                    .add_image(gbuffer.ldir0_buffer.clone()).unwrap()
                    .add_image(gbuffer.ldir1_buffer.clone()).unwrap()
                    // light value buffer
                    .add_image(gbuffer.light0_buffer.clone()).unwrap()
                    .add_image(gbuffer.light1_buffer.clone()).unwrap()
                    // output image
                    .add_image(swapchain_images[image_num].clone()).unwrap()
                    // temporal buffer
                    .add_image(gbuffer.temp_buffers[0].clone()).unwrap()
                    // voxel data
                    .add_buffer(svdag_geometry_buffer.clone()).unwrap()
                    .add_buffer(svdag_material_buffer.clone()).unwrap()
                    .build().unwrap()
                );

                let light_combine_pc = shaders::LightCombinePushConstantData {
                    ambient_light : [0.02; 3],
                    camera_forward : pre_trace_pc.camera_forward,
                    camera_up : pre_trace_pc.camera_up,
                    camera_origin : pre_trace_pc.camera_origin,
                    frame_idx : intersect_pc.frame_idx,
                    _dummy0:[0;4],
                    _dummy1:[0;4],
                    _dummy2:[0;4],
                };

                // number of blocks depends on the runtime-computed local_size parameters of the physical device
                let block_dim_x = (surface_width - 1) / local_size_x + 1;
                let block_dim_y = (surface_width - 1) / local_size_y + 1;

                // we build a command buffer for this frame
                // needs to be built each frame because we don't know which swapchain image we will be told to render to
                // its possible a command buffer could be built for each swapchain ahead of time, but that would add complexity
                // let mut render_command_buffer_builder = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), compute_queue.family()).unwrap();

                // render_command_buffer_builder
                //     .dispatch([(gbuffer.pre_trace_width - 1) / local_size_x + 1, (gbuffer.pre_trace_height - 1) / local_size_y + 1, 1], pre_trace_compute_pipeline.clone(), pre_trace_set.clone(), pre_trace_pc).unwrap()
                //     .dispatch([block_dim_x, block_dim_y, 1], intersect_compute_pipeline.clone(), intersect_set.clone(), intersect_pc).unwrap()
                //     .dispatch([block_dim_x, block_dim_y, 1], light_bounce_compute_pipeline.clone(), light_bounce_set.clone(), light_bounce_pc).unwrap()
                //     .dispatch([block_dim_x, block_dim_y, 1], light_occlude_compute_pipeline.clone(), light_occlude_set_0.clone(), light_occlude_pc).unwrap()
                //     .dispatch([block_dim_x, block_dim_y, 1], light_occlude_compute_pipeline.clone(), light_occlude_set_1.clone(), light_occlude_pc).unwrap()
                //     .dispatch([block_dim_x, block_dim_y, 1], light_combine_compute_pipeline.clone(), light_combine_set.clone(), light_combine_pc).unwrap();

                // let render_command_buffer = render_command_buffer_builder.build().unwrap();

                let mut render_command_buffer_builder = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), compute_queue.family()).unwrap();

                render_command_buffer_builder
                    .dispatch([(gbuffer.pre_trace_width - 1) / local_size_x + 1, (gbuffer.pre_trace_height - 1) / local_size_y + 1, 1], pre_trace_compute_pipeline.clone(), pre_trace_set.clone(), pre_trace_pc).unwrap()
                    .dispatch([block_dim_x, block_dim_y, 1], intersect_compute_pipeline.clone(), intersect_set.clone(), intersect_pc).unwrap()
                    .dispatch([block_dim_x, block_dim_y, 1], light_bounce_compute_pipeline.clone(), light_bounce_set.clone(), light_bounce_pc).unwrap()
                    .dispatch([block_dim_x, block_dim_y, 1], light_occlude_compute_pipeline.clone(), light_occlude_set_0.clone(), light_occlude_pc).unwrap()
                    .dispatch([block_dim_x, block_dim_y, 1], light_occlude_compute_pipeline.clone(), light_occlude_set_1.clone(), light_occlude_pc).unwrap()
                    .dispatch([block_dim_x, block_dim_y, 1], light_combine_compute_pipeline.clone(), light_combine_set.clone(), light_combine_pc).unwrap();

                let render_command_buffer = render_command_buffer_builder.build().unwrap();


                let future = previous_frame_end.take().unwrap()
                    .join(acquire_future)
                    // rendering is done in a series of compute shader commands
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
                    frame_counts = FRAME_COUNTS / 10;
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

            if input.scroll_diff() < 0.0 && intersect_pc.max_depth > 0{
                intersect_pc.max_depth -= 1;
            } else if input.scroll_diff() > 0.0 && intersect_pc.max_depth < 15 {
                intersect_pc.max_depth += 1;
            }

            
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

