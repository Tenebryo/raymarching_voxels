#![allow(dead_code, unused_imports)]

extern crate nalgebra as na;

mod timing;
mod shaders;
mod noise;
mod gbuffer;
mod dbuffer;
mod vox;
mod brdf;
mod pipelines;
mod descriptor_sets;
mod command_buffer;

use timing::Timing;
use gbuffer::GBuffer;


use vulkano::SafeDeref;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, DeviceLocalBuffer};
use vulkano::command_buffer::pool::{CommandPool};
use vulkano::command_buffer::pool::standard::StandardCommandPoolAlloc;
use vulkano::command_buffer::{StateCacher, AutoCommandBufferBuilder, DynamicState, sys::{UnsafeCommandBufferBuilderPipelineBarrier, UnsafeCommandBufferBuilder, Kind, Flags, UnsafeCommandBuffer}, pool::StandardCommandPool};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::descriptor::PipelineLayoutAbstract;
use vulkano::device::{Device, DeviceExtensions, Queue};
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

use imgui::{Context, FontConfig, FontGlyphRanges, FontSource, Ui};
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use imgui_vulkano_renderer::{Renderer, RendererError};

use cgmath::{Vector3, Quaternion, InnerSpace, Rotation3, Rad, Rotation};

use png;

use std::io::Cursor;
use std::sync::Arc;
use std::time::Instant;
use std::io::stdout;
use std::path::*;
use std::collections::VecDeque;

use std::ffi::CStr;

use rand::prelude::*;

use crossterm::{
    ExecutableCommand,
    cursor::MoveUp
};


fn create_dag_sphere(depth : usize, c : Vector3<f32>, r : f32) -> vox::VoxelChunk {
    vox::VoxelChunk::from_intersection_test(depth, |v, s| {

        let mut sq_dist = 0.0;

        if c.x < v.x - s { sq_dist += (v.x - s - c.x).powi(2); }
        if c.x > v.x + s { sq_dist += (v.x + s - c.x).powi(2); }
        
        if c.y < v.y - s { sq_dist += (v.y - s - c.y).powi(2); }
        if c.y > v.y + s { sq_dist += (v.y + s - c.y).powi(2); }
        
        if c.z < v.z - s { sq_dist += (v.z - s - c.z).powi(2); }
        if c.z > v.z + s { sq_dist += (v.z + s - c.z).powi(2); }

        sq_dist < r * r
    })
}

fn main() {

    let init_start = Instant::now();

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

    let compute_graphics_queue_family = physical.queue_families()
        .find(|&q| 
            q.supports_compute() && 
            q.supports_graphics() && 
            surface.is_supported(q).unwrap_or(false)
        ).unwrap();
    let compute_queue_family = physical.queue_families()
        .find(|&q| 
            q.supports_compute()
        ).unwrap();
    // We take the first queue that supports drawing to our window.
    // let graphics_queue_family = physical.queue_families().find(|&q| q.supports_graphics() && surface.is_supported(q).unwrap_or(false)).unwrap();
    // get the a transfer queue family. This will be used to transfer voxel data.
    // let transfer_queue_family = physical.queue_families().find(|&q| q.explicitly_supports_transfers()).unwrap();

    // Now initializing the device.
    let (device, mut queues) = Device::new(physical, physical.supported_features(),
        &DeviceExtensions{khr_swapchain: true, khr_storage_buffer_storage_class:true, ..DeviceExtensions::none()},
        [(compute_graphics_queue_family, 0.5), (compute_queue_family, 0.5)].iter().cloned()).unwrap();

    // Since we can request multiple queues, the `queues` variable is in fact an iterator. In this
    // example we use only one queue, so we just retrieve the first and only element of the
    // iterator and throw it away.

    let render_queue = queues.next().unwrap();
    let compute_queue = queues.next().unwrap();

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

    let pipeline = pipelines::Pipelines::new(device.clone(), local_size_x, local_size_y);

    println!("Compute Pipelines initialized");

    //*************************************************************************************************************************************
    // Screen Buffer Allocation
    //*************************************************************************************************************************************


    let swapchain_format;

    // build a swapchain compatible with the window surface we built earlier
    let (mut swapchain, mut swapchain_images) = {
        let caps = surface.capabilities(physical).unwrap();
        let usage = caps.supported_usage_flags;

        let alpha = caps.supported_composite_alpha.iter().next().unwrap();

        let format = caps.supported_formats[0].0;
        swapchain_format = format;

        println!("Using Swapchain Format: {:?}", format);

        let dimensions: [u32; 2] = surface.window().inner_size().into();

        Swapchain::new(device.clone(), surface.clone(), caps.min_image_count, format,
            dimensions, 1, usage, &compute_queue, SurfaceTransform::Identity, alpha,
            PresentMode::Fifo, FullscreenExclusive::Default, true, ColorSpace::SrgbNonLinear).unwrap()

    };

    println!("Swapchain initialized");
    
    let stratum_size = 4;
    let [mut width, mut height]: [u32; 2] = surface.window().inner_size().into();
    let num_temp_buffers = 4;
    let mut gbuffer = GBuffer::new_buffers(device.clone(), compute_queue_family.clone(), width, height, num_temp_buffers, stratum_size);

    println!("Storage Images initialized");
    
    let mut imgui = Context::create();
    imgui.set_ini_filename(None);

    let mut platform = WinitPlatform::init(&mut imgui);

    platform.attach_window(imgui.io_mut(), surface.window(), HiDpiMode::Rounded);
    
    let hidpi_factor = platform.hidpi_factor();
    let font_size = (13.0 * hidpi_factor) as f32;
    imgui.fonts().add_font(&[
        FontSource::DefaultFontData {
            config: Some(FontConfig {
                size_pixels: font_size,
                ..FontConfig::default()
            }),
        },
        FontSource::TtfData {
            data: include_bytes!("../data/font/mplus-1p-regular.ttf"),
            size_pixels: font_size,
            config: Some(FontConfig {
                rasterizer_multiply: 1.75,
                glyph_ranges: FontGlyphRanges::japanese(),
                ..FontConfig::default()
            }),
        },
    ]);

    let mut imgui_renderer = Renderer::init(&mut imgui, device.clone(), render_queue.clone(), swapchain_format).unwrap();

    println!("ImGui Context and Renderer Initialized");

    //*************************************************************************************************************************************
    // Constant Data
    //*************************************************************************************************************************************

    // let lin_sampler = Sampler::new(device.clone(), Filter::Linear, Filter::Linear,
    //     MipmapMode::Linear, SamplerAddressMode::Repeat, SamplerAddressMode::Repeat,
    //     SamplerAddressMode::Repeat, 0.0, 1.0, 0.0, 0.0).unwrap();

    let lin_sampler = Sampler::simple_repeat_linear_no_mipmap(device.clone());

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

        // println!("{:?}", dimensions);

        ImmutableImage::from_iter(
            image_data.iter().cloned(),
            dimensions,
            Format::R8G8B8A8Srgb,
            render_queue.clone()
        ).unwrap()
    };
    
    println!("Loaded Noise Texture.");

    let (skysphere_tex, load_future2) = {
        let data_cursor = std::io::Cursor::new(
            std::fs::read("./data/tex/skysphere_01_4k.hdr").unwrap()
        );

        let img = image::hdr::HdrDecoder::new(data_cursor).unwrap();

        let dim = img.metadata();

        let img_data = img.read_image_hdr().unwrap();

        let mut rgba32f_data = vec![0f32; img_data.len() * 4];

        for i in 0..(img_data.len()) {
            let j = 4 * i;

            rgba32f_data[j + 0] = img_data[i].0[0];
            rgba32f_data[j + 1] = img_data[i].0[1];
            rgba32f_data[j + 2] = img_data[i].0[2];
            rgba32f_data[j + 3] = 1.0;
        }

        let dim = Dimensions::Dim2d {
            width : dim.width,
            height : dim.height,
        };

        let (tex, fut) = ImmutableImage::from_iter(rgba32f_data.iter().cloned(), dim, Format::R32G32B32A32Sfloat, render_queue.clone()).unwrap();

        (tex, fut.join(load_future))
    };

    println!("Loaded Skysphere Texture.");

    //*************************************************************************************************************************************
    // Miscellaneous variables @VARS
    //*************************************************************************************************************************************

    let mut recreate_swapchain = false;

    // let mut previous_frame_end = Some(Box::new(load_future) as Box<dyn GpuFuture>);
    let mut previous_frame_end = Some(Box::new(load_future2) as Box<dyn GpuFuture>);
    // let mut previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>);

    let mut fps = Timing::new(16);
    const FRAME_COUNTS : i32 = 10;
    let mut frame_counts = FRAME_COUNTS;
    let p_start = Instant::now();
    let mut adaptation = 2.0;
    let mut exposure = 1.0;
    let mut reprojection_miss_ratio = 0.02;
    let mut sample_cutoff = 32;
    let mut atrous_col_weight = 0.8; // 0.1;
    let mut atrous_nrm_weight = 0.0001; // 32.0;
    let mut atrous_pos_weight = 0.8; // 16.0;
    let mut atrous_iterations = 6; // 16.0;
    let mut atrous_enable = true; // 16.0;
    let mut sample_decay_patch_size = 3u32;

    let mut ui_enabled = true;
    let mut ui_wants_mouse_capture = false;
    let mut ui_wants_keyboard_capture = false;
    let mut render_new_frame = true;
    let mut render_single_frame = false;
    let mut last_mouse_pos = (0.0, 0.0);

    let mut input = WinitInputHelper::new();

    let mut forward = Vector3::new(1.0, 0.0, 0.0).normalize();
    let up = Vector3::new(0.0, 1.0, 0.0);
    let mut position = Vector3::new(0.2, 0.075, 0.3); // sponza
    // let mut position = Vector3::new(0.2, 0.075, 0.2); // sibenik
    let mut old_position = position;
    let mut old_forward = forward;
    let mut old_up = up;
    let mut input_time = Instant::now();
    let mut pitch = 0.0;
    let mut yaw = 0.0;

    let mut intersect_pc = shaders::IntersectPushConstants {
        camera_origin : [position.x, position.y, position.z],
        camera_forward : [forward.x, forward.y, forward.z],
        camera_up : [up.x, up.y, up.z],
        max_depth : 16,
        render_dist: 100.0,
        frame_idx : 0,
        noise_0 : [0,0,0],
        noise_1 : [0,0,0],
        noise_2 : [0,0,0],
        noise_frames : 64,
        _dummy0 : [0; 4],
        _dummy1 : [0; 4],
    };

    //*************************************************************************************************************************************
    // Voxel, Material, and Light Data Allocation and Generation
    //*************************************************************************************************************************************

    // parse voxel goemetry data
    let mut svdag_geometry_data = {

        // let chunk_bytes = std::fs::read("./data/dag/sponza.svdag").unwrap();
        // let chunk_bytes = std::fs::read("./data/dag/sponza_tex.svdag").unwrap();
        // let chunk_bytes = std::fs::read("./data/dag/sphere_1k.svdag").unwrap();
        let chunk_bytes = std::fs::read("./data/dag/sponza_tex_1k.svdag").unwrap();
        // let chunk_bytes = std::fs::read("./data/dag/sponza_tex_16k.svdag").unwrap();
        // let chunk_bytes = std::fs::read("./data/dag/sponza_mats.svdag").unwrap();
        // let chunk_bytes = std::fs::read("./data/dag/sibenik_mats.svdag").unwrap();
        
        bincode::deserialize::<vox::VoxelChunk>(&chunk_bytes).expect("Deserialization Failed")
    };
    
    svdag_geometry_data.calculate_lod_materials();

    let dag_buffer_start = Instant::now();

    // load the voxel data onto the GPU
    let svdag_geometry_buffer = {
        CpuAccessibleBuffer::<[vox::VChildDescriptor]>::from_iter(device.clone(), BufferUsage::all(), false, svdag_geometry_data.voxels.iter().cloned()).unwrap()
    };

    let svdag_material_buffer = {
        CpuAccessibleBuffer::<[u32]>::from_iter(device.clone(), BufferUsage::all(), false, svdag_geometry_data.lod_materials.iter().cloned()).unwrap()
    };


    println!("Voxel Data initialized ({:?})", dag_buffer_start.elapsed());

    // create a list of point lights to render
    let point_light_buffer = {
        use shaders::PointLight;

        let lights = [
            // PointLight {
            //     position : [position.x, position.y, position.z],
            //     power : 0.01,
            //     // power : 0.002,
            //     color : [1.0, 1.0, 1.0],
            //     radius : 0.005,
            // },
            // PointLight {
            //     position : [0.2, 0.12, 0.3],
            //     power : 0.006,
            //     color : [1.0, 1.0, 1.0],
            //     radius : 0.02,
            //     max_radiance : 5.0,
            //     _dummy0 : [0;12],
            // },
            // PointLight {
            //     position : [0.5, 0.3, 0.3],
            //     power : 0.01,
            //     color : [1.0, 1.0, 1.0],
            //     radius : 0.02,
            //     max_radiance : 10.0,
            //     _dummy0 : [0;12],
            // },
            PointLight {
                position : [0.19, 0.125, 0.2],
                power : 0.002,
                color : [1.0, 1.0, 1.0],
                radius : 0.005,
                max_radiance : 1.0,
                _dummy0 : [0;12],
            },
            PointLight {
                position : [0.19, 0.125, 0.40],
                power : 0.002,
                color : [1.0, 1.0, 1.0],
                radius : 0.005,
                max_radiance : 1.0,
                _dummy0 : [0;12],
            },
            PointLight {
                position : [0.81, 0.125, 0.2],
                power : 0.002,
                color : [1.0, 1.0, 1.0],
                radius : 0.005,
                max_radiance : 1.0,
                _dummy0 : [0;12],
            },
            PointLight {
                position : [0.81, 0.125, 0.40],
                power : 0.002,
                color : [1.0, 1.0, 1.0],
                radius : 0.005,
                max_radiance : 1.0,
                _dummy0 : [0;12],
            }
        ];

        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, lights.iter().cloned()).unwrap()
    };
    // create a list of directional lights to render
    let directional_light_buffer = {
        use shaders::DirectionalLight;

        let lights = [
            DirectionalLight {
                direction : [0.2, -1.0, 0.2],
                color : [1.0; 3],
                noise_scale : 0.2,
                _dummy0 : [0;4],
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
                max_radiance : 1.0,
            }
        ];

        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, lights.iter().cloned()).unwrap()
    };

    println!("Created Light Buffers");
    
    // let basic_brdf = brdf::BRDF::read_matusik_brdf_file(Path::new("data/teflon.bin"), [90, 90, 90, 90], 2, 2).expect("could not load brdf file");
    let basic_brdf = bincode::deserialize::<brdf::BRDF>(include_bytes!("../data/brdf/teflon-16-16-128-32.brdf")).unwrap();
    // create the brdf buffer
    let brdf_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, (0..(basic_brdf.len())).map(|_| 0.0f32)).unwrap();

    basic_brdf.write_to_buffer(&mut brdf_buffer.write().unwrap(), 0);
    
    println!("Created BRDF Buffer");

    let num_materials;

    // create a list of materials to render
    let material_buffer = {
        use shaders::Material;


        // let materials = [
        //     // air material
        //     // Material {brdf : basic_brdf.create_shader_type(0), albedo : [0.0; 3], transparency: 1.0, emission: [0.0; 3], flags: 0b00000000, roughness: 0.0, metalness: 0.3, _dummy0: [0;8]},
        //     Material {brdf : 0, albedo : [0.0; 3], transparency: 1.0, emission: [0.0; 3], flags: 0b00000000, roughness: 0.0, metalness: 0.3, _dummy0: [0;12], _dummy1: [0;8]},
        //     // solid material
        //     // Material {brdf : basic_brdf.create_shader_type(0), albedo : [1.0; 3], transparency: 0.0, emission: [0.0; 3], flags: 0b00000001, roughness: 0.0, metalness: 0.3, _dummy0: [0;8]}
        //     Material {brdf : 0, albedo : [1.0, 1.0, 1.0], transparency: 0.0, emission: [0.0; 3], flags: 0b00000001, roughness: 0.0, metalness: 0.3, _dummy0: [0;12], _dummy1: [0;8]},
        //     Material {brdf : 0, albedo : [1.0, 1.0, 1.0], transparency: 0.0, emission: [0.1; 3], flags: 0b00000001, roughness: 0.0, metalness: 0.3, _dummy0: [0;12], _dummy1: [0;8]}
        // ];

        // num_materials = 2;

        // CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, materials.iter().cloned()).unwrap()


        let material_bytes = std::fs::read("./data/dag/sponza_tex_1k.mats")
        // let material_bytes = std::fs::read("./data/dag/sponza_tex_16k.mats")
        // let material_bytes = std::fs::read("./data/dag/sponza_tex.mats")
        // let material_bytes = std::fs::read("./data/dag/sponza_mats.mats")
        // let material_bytes = std::fs::read("./data/dag/sibenik_mats.mats")
            .expect("failed to read material file");

        let mut materials : Vec<shaders::Material> = vec![
            Material {brdf : 0, albedo : [0.0; 3], transparency: 1.0, emission: [0.0; 3], flags: 0b00000000, roughness: 0.0, metalness: 0.3, _dummy0: [0;12], _dummy1: [0;8]},
        ];
        materials.extend(bincode::deserialize::<Vec<vox::Material>>(&material_bytes).unwrap().iter()
            .map(|m| shaders::Material {
                albedo : m.albedo,
                emission : m.emission,
                roughness : m.roughness,
                metalness : m.metalness,
                brdf : 0,
                flags : 0,
                transparency : 0.0,
                _dummy0: [0;12], _dummy1: [0;8],
            }));

        num_materials = materials.len() as u32;
            
        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, materials.iter().cloned()).unwrap()

    };

    
    println!("Material Data initialized");

    let mut dbuffer = dbuffer::DBuffer {
        svdag_geometry_buffer    : svdag_geometry_buffer.clone(),
        svdag_material_buffer    : svdag_material_buffer.clone(),
        directional_light_buffer : directional_light_buffer.clone(),
        point_light_buffer       : point_light_buffer.clone(),
        spot_light_buffer        : spot_light_buffer.clone(),
        brdf_buffer              : brdf_buffer.clone(),
        material_buffer          : material_buffer.clone(),
        blue_noise_tex           : blue_noise_tex.clone(),
        nst_sampler              : nst_sampler.clone(),
        skysphere_tex            : skysphere_tex.clone(),
        lin_sampler              : lin_sampler.clone(),
    };

    let mut desc_sets = descriptor_sets::DescriptorSets::new(&pipeline, &gbuffer, &dbuffer);

    let luminance_buffer = CpuAccessibleBuffer::<[f32]>::from_iter(device.clone(), BufferUsage::all(), false, [0.0; 4].iter().cloned()).unwrap();
    let mut avg_luminance = 0.0;
    let luminance_frame_period = 4;

    //*************************************************************************************************************************************
    // Main Event Loop @MAINLOOP
    //*************************************************************************************************************************************

    println!("Initialization Finished! (Took {:?})", init_start.elapsed());

    println!("Starting Event Loop");

    let depth_scale = (i32::MAX as f64 / intersect_pc.render_dist as f64) as u32;

    let mut imgui_frame_time = Instant::now();

    let mut cpu_rendering_time = 0.0;

    let mut postprocess_frame_dt = Instant::now();

    event_loop.run(move |event, _, control_flow| {

        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                *control_flow = ControlFlow::Exit;
            },
            ref e @ Event::WindowEvent { event: WindowEvent::Resized(_), .. } => {
                platform.handle_event(imgui.io_mut(), surface.window(), e);
                recreate_swapchain = true;
            },
            Event::NewEvents(_) => {
                imgui_frame_time = imgui.io_mut().update_delta_time(imgui_frame_time);
            }
            Event::MainEventsCleared => {
            }
            Event::RedrawEventsCleared => {
                // start FPS sample
                fps.start_sample();
                
                let cpu_rendering_start = Instant::now();

                previous_frame_end.as_mut().unwrap().cleanup_finished();

                // recreate the swapchain when the window is resized
                if recreate_swapchain {
                    // Get the new dimensions of the window.
                    let dimensions: [u32; 2] = surface.window().inner_size().into();
                    width = dimensions[0];
                    height = dimensions[1];
                    let (new_swapchain, new_images) = match swapchain.recreate_with_dimensions(dimensions) {
                        Ok(r) => r,
                        // This error tends to happen when the user is manually resizing the window.
                        // Simply restarting the loop is the easiest way to fix this issue.
                        Err(SwapchainCreationError::UnsupportedDimensions) => return,
                        Err(e) => panic!("Failed to recreate swapchain: {:?}", e)
                    };

                    swapchain = new_swapchain;
                    swapchain_images = new_images;

                    // re-allocate buffer images

                    gbuffer = GBuffer::new_buffers(device.clone(), render_queue.family(), width, height, num_temp_buffers, stratum_size);
                    desc_sets = descriptor_sets::DescriptorSets::new(&pipeline, &gbuffer, &dbuffer);

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
                
                platform
                    .prepare_frame(imgui.io_mut(), &surface.window())
                    .expect("Failed to prepare frame");
                

                if intersect_pc.frame_idx % luminance_frame_period > 0 {
                    if let Ok(lock) = luminance_buffer.read() {
                        avg_luminance = lock[0] + lock[1] + lock[2];
                    }
                }

                let ui = imgui.frame();

                if ui_enabled {
                    use imgui::*;
                    
                    // @UI

                    let (t, _t_var, fps, _) = fps.stats();

                    imgui::Window::new(im_str!("Info"))
                        .flags(WindowFlags::NO_MOVE | WindowFlags::NO_COLLAPSE)
                        .size([400.0, height as f32], Condition::FirstUseEver)
                        .position([0.0, 0.0], Condition::FirstUseEver)
                        .build(&ui, || {
                            ui.text(format!("Framerate: {:3.1} fps ({:1.3} ms)", fps, 1000.0 * t));
                            ui.separator();
                            ui.text(format!("Position:  {:0<5.3?}", position));
                            ui.text(format!("Forward:   {:0<5.3?}", forward));
                            ui.text(format!("Pitch/Yaw: {:0<5.3?}/{:0<5.3?}", 180.0 * pitch / std::f32::consts::PI, 180.0 * yaw / std::f32::consts::PI));
                            ui.text(format!("CPU Time   {:5.3} us", 1e6 * cpu_rendering_time));
                            ui.text(format!("Luminance  {:5.3}", avg_luminance));

                            // let mouse_pos = ui.io().mouse_pos;
                            // ui.text(format!(
                            //     "Mouse Position: ({:.1},{:.1})",
                            //     mouse_pos[0], mouse_pos[1]
                            // ));

                            ui.separator();

                            imgui::Slider::new(im_str!("Exposure"), 0.1..=5.0)
                                .build(&ui, &mut exposure);
                                
                            imgui::Slider::new(im_str!("Adaptation"), 0.1..=5.0)
                                .build(&ui, &mut adaptation);

                            imgui::Slider::new(im_str!("LoD"), 1..=15)
                                .build(&ui, &mut intersect_pc.max_depth);

                            imgui::Slider::new(im_str!("Miss Ratio"), 0.0..=1.0)
                                .power(2.0)
                                .build(&ui, &mut reprojection_miss_ratio);
                                
                            imgui::InputInt::new(&ui, im_str!("Sample Cutoff"), &mut sample_cutoff)
                                .step(1).step_fast(10)
                                .auto_select_all(true)
                                .build();
                            
                            let mut sdps = sample_decay_patch_size as i32;

                            imgui::InputInt::new(&ui, im_str!("Sample Decay Patch Size"), &mut sdps)
                                .step(1)
                                .auto_select_all(true)
                                .build();

                            sample_decay_patch_size = u32::max(2, sdps as u32);

                            sample_cutoff = sample_cutoff.abs();

                            ui.separator();

                            if CollapsingHeader::new(im_str!("A-Trous Filter Parameters")).build(&ui) {

                                ui.checkbox(im_str!("Enabled"), &mut atrous_enable);
                                
                                imgui::Slider::new(im_str!("c_phi"), 0.00001..=4.0)
                                    .power(2.0)
                                    .build(&ui, &mut atrous_col_weight);
                                
                                imgui::Slider::new(im_str!("n_phi"), 0.00001..=4.0)
                                    .power(2.0)
                                    .build(&ui, &mut atrous_nrm_weight);
                                
                                imgui::Slider::new(im_str!("p_phi"), 0.00001..=4.0)
                                    .power(2.0)
                                    .build(&ui, &mut atrous_pos_weight);

                                imgui::Slider::new(im_str!("iterations"), 2..=8)
                                    .build(&ui, &mut atrous_iterations);

                            }
                            ui.separator();

                            ui.checkbox(im_str!("Render Single Frame"), &mut render_single_frame);


                            ui_wants_mouse_capture = ui.io().want_capture_mouse;
                            ui_wants_keyboard_capture = ui.io().want_capture_keyboard;
                        });
                }

                let _time = p_start.elapsed().as_secs_f32();
                
                // Push Constant Construction @PC

                intersect_pc.camera_forward = [forward.x, forward.y, forward.z];
                intersect_pc.camera_origin = [position.x, position.y, position.z];
                intersect_pc.camera_up = [up.x, up.y, up.z];
                intersect_pc.frame_idx += 1;
                intersect_pc.noise_0 = [
                    thread_rng().gen_range(0,intersect_pc.noise_frames),
                    thread_rng().gen_range(0,intersect_pc.noise_frames),
                    thread_rng().gen_range(0,intersect_pc.noise_frames)
                ];
                intersect_pc.noise_1 = [
                    thread_rng().gen_range(0,intersect_pc.noise_frames),
                    thread_rng().gen_range(0,intersect_pc.noise_frames),
                    thread_rng().gen_range(0,intersect_pc.noise_frames)
                ];
                intersect_pc.noise_2 = [
                    thread_rng().gen_range(0,intersect_pc.noise_frames),
                    thread_rng().gen_range(0,intersect_pc.noise_frames),
                    thread_rng().gen_range(0,intersect_pc.noise_frames)
                ];
                
                let reproject_pc = shaders::ReprojectPushConstantData {
                    p_origin  : [old_position.x, old_position.y, old_position.z],
                    p_forward : [old_forward.x,  old_forward.y,  old_forward.z ],
                    p_up      : [old_up.x,       old_up.y,       old_up.z      ],
                    depth_scale,
                    reprojection_miss_ratio,
                    _dummy0 : [0; 4],
                    _dummy1 : [0; 4],
                };

                let stratified_sample_pc = shaders::StratifiedSamplePushConstantData {
                    patch_size : stratum_size,
                };

                let pre_trace_pc = shaders::PreTracePushConstants {
                    camera_forward : intersect_pc.camera_forward,
                    camera_origin : intersect_pc.camera_origin,
                    camera_up : intersect_pc.camera_up,
                    max_depth : u32::min(6, intersect_pc.max_depth),

                    _dummy0 : [0;4],
                    _dummy1 : [0;4],
                };

                let light_bounce_pc = shaders::LightBouncePushConstantData {
                    camera_forward : pre_trace_pc.camera_forward,
                    camera_up : pre_trace_pc.camera_up,
                    camera_origin : pre_trace_pc.camera_origin,
                    n_directional_lights : 1,
                    n_point_lights : 4,
                    n_spot_lights : 0,
                    render_dist : intersect_pc.render_dist,
                    max_depth : intersect_pc.max_depth,
                    sample_cutoff : sample_cutoff as u32,
                };
                
                let light_occlude_pc_a = shaders::LightOccludePushConstantData {
                    render_dist : intersect_pc.render_dist,
                    num_materials,
                    max_depth : intersect_pc.max_depth,
                    bounce_idx : 0,
                };
                
                let light_occlude_pc_b = shaders::LightOccludePushConstantData {
                    render_dist : intersect_pc.render_dist,
                    num_materials,
                    max_depth : intersect_pc.max_depth,
                    bounce_idx : 1,
                };

                let light_combine_pc = shaders::LightCombinePushConstantData {
                    ambient_light : [0.00; 3],
                    camera_forward : pre_trace_pc.camera_forward,
                    camera_up : pre_trace_pc.camera_up,
                    camera_origin : pre_trace_pc.camera_origin,
                    frame_idx : intersect_pc.frame_idx,
                    _dummy0:[0;4],
                    _dummy1:[0;4],
                    _dummy2:[0;4],
                };

                let atrous_pc = shaders::AtrousPushConstantData {
                    camera_origin : intersect_pc.camera_origin,
                    stride : 1,
                    col_weight : atrous_col_weight as f32,
                    nrm_weight : atrous_nrm_weight as f32,
                    pos_weight : atrous_pos_weight as f32,
                };
                
                let sample_decay_pc = shaders::SampleDecayPushConstantData {
                    offset : [
                        thread_rng().gen_range(0i32, sample_decay_patch_size as i32),
                        thread_rng().gen_range(0i32, sample_decay_patch_size as i32)
                    ],
                    patch_size : sample_decay_patch_size,
                    min_samples : sample_cutoff as u32 / 4,
                };

                let dt = {
                    let now = Instant::now();
                    let t = now.duration_since(postprocess_frame_dt).as_secs_f32();
                    postprocess_frame_dt = now;
                    t
                };

                let postprocess_pc = shaders::PostprocessPushConstantData {
                    exposure,
                    dt,
                    frame_idx : intersect_pc.frame_idx,
                    adaptation,
                    center : [width / 2, height / 2],
                };

                // @CMDBUF
                // we build a command buffer for this frame
                // needs to be built each frame because we don't know which swapchain image we will be told to render to
                // its possible a command buffer could be built for each swapchain ahead of time, but that would add complexity
                // another option is building a command buffer for the render pipeline with fixed inputs and outputs, then blit the output to the swapchain
                
                // number of blocks depends on the runtime-computed local_size parameters of the physical device
                let block_dim_x = (width - 1) / local_size_x + 1;
                let block_dim_y = (height - 1) / local_size_y + 1;

                let mut render_command_buffer_builder = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), render_queue.family()).unwrap();

                if render_new_frame {

                    render_command_buffer_builder
                        // set up reprojection buffers
                        .clear_color_image(gbuffer.reprojected_col_buffer.clone(), [0.0; 4].into()).unwrap()
                        .clear_color_image(gbuffer.reprojected_pos_buffer.clone(), [0.0; 4].into()).unwrap()
                        .clear_color_image(gbuffer.reprojected_cnt_buffer.clone(), [1;   4].into()).unwrap()
                        .copy_image(
                            gbuffer.position0_buffer.clone(), [0,0,0], 0, 0,
                            gbuffer.prev_pos_buffer.clone(), [0,0,0], 0, 0,
                            [width, height, 1], 1
                        ).unwrap()
                        .dispatch([(gbuffer.pre_trace_width - 1) / local_size_x + 1, (gbuffer.pre_trace_height - 1) / local_size_y + 1, 1], pipeline.pre_trace.clone(), desc_sets.pre_trace_set.clone(), pre_trace_pc).unwrap()
                        .dispatch([block_dim_x, block_dim_y, 1], pipeline.intersect.clone(), desc_sets.intersect_set.clone(), intersect_pc).unwrap()
                        .dispatch([block_dim_x, block_dim_y, 1], pipeline.stratified_sample.clone(), desc_sets.stratified_sample_set.clone(), stratified_sample_pc).unwrap()
                        // reproject previous frame
                        .dispatch([block_dim_x, block_dim_y, 1], pipeline.reproject.clone(), desc_sets.reproject_set.clone(), reproject_pc).unwrap()
                        // decay some reprojected samples
                        .dispatch([block_dim_x / sample_decay_patch_size, block_dim_y / sample_decay_patch_size, 1], pipeline.sample_decay.clone(), desc_sets.sample_decay_set.clone(), sample_decay_pc).unwrap()
                        // lighting calculations
                        .dispatch([block_dim_x, block_dim_y, 1], pipeline.light_bounce.clone(), desc_sets.light_bounce_set.clone(), light_bounce_pc).unwrap()
                        .dispatch([block_dim_x, block_dim_y, 1], pipeline.light_occlude.clone(), desc_sets.light_occlude_set_0.clone(), light_occlude_pc_a).unwrap()
                        .dispatch([block_dim_x, block_dim_y, 1], pipeline.light_occlude.clone(), desc_sets.light_occlude_set_1.clone(), light_occlude_pc_b).unwrap()
                        .dispatch([block_dim_x, block_dim_y, 1], pipeline.light_combine.clone(), desc_sets.light_combine_set.clone(), light_combine_pc).unwrap()
                        .copy_image(
                            gbuffer.hdr_light_buffer.clone(), [0,0,0], 0, 0,
                            gbuffer.postprocess_input_buffer.clone(), [0,0,0], 0, 0,
                            [width, height, 1], 1
                        ).unwrap();
                    
                    if atrous_enable {
                        // apply atrous filtering to the hdr image
                        let mut s = 1;
                        for _ in 0..(atrous_iterations/2) {
                            render_command_buffer_builder
                                // filtering
                                .dispatch([block_dim_x, block_dim_y, 1], pipeline.atrous.clone(), desc_sets.atrous_set_a.clone(), shaders::AtrousPushConstantData{stride :   s, ..atrous_pc}).unwrap()
                                .dispatch([block_dim_x, block_dim_y, 1], pipeline.atrous.clone(), desc_sets.atrous_set_b.clone(), shaders::AtrousPushConstantData{stride : 2*s, ..atrous_pc}).unwrap();
                            
                            s *= 4;
                        }
                    }
                    
                    render_command_buffer_builder
                        .dispatch([block_dim_x, block_dim_y, 1], pipeline.apply_texture.clone(), desc_sets.apply_texture_set.clone(), ()).unwrap();

                    // compute average luminance by successively blit-ing down to a single pixel
                    render_command_buffer_builder
                        .blit_image(
                            gbuffer.postprocess_input_buffer.clone(), [0,0,0], [width as i32, height as i32, 1], 0, 0, 
                            gbuffer.luminance_buffer.clone(), [0,0,0], [32,32,1], 0, 0, 1, Filter::Linear
                        ).unwrap()
                        .blit_image(
                            gbuffer.luminance_buffer.clone(), [0,0,0], [32,32,1], 0, 0, 
                            gbuffer.luminance_buffer_t.clone(), [0,0,0], [16,16,1], 0, 0, 1, Filter::Linear
                        ).unwrap()
                        .blit_image(
                            gbuffer.luminance_buffer_t.clone(), [0,0,0], [16,16,1], 0, 0, 
                            gbuffer.luminance_buffer.clone(), [0,0,0], [8,8,1], 0, 0, 1, Filter::Linear
                        ).unwrap()
                        .blit_image(
                            gbuffer.luminance_buffer.clone(), [0,0,0], [8,8,1], 0, 0, 
                            gbuffer.luminance_buffer_t.clone(), [0,0,0], [4,4,1], 0, 0, 1, Filter::Linear
                        ).unwrap()
                        .blit_image(
                            gbuffer.luminance_buffer_t.clone(), [0,0,0], [4,4,1], 0, 0, 
                            gbuffer.luminance_buffer.clone(), [0,0,0], [2,2,1], 0, 0, 1, Filter::Linear
                        ).unwrap()
                        .blit_image(
                            gbuffer.luminance_buffer.clone(), [0,0,0], [2,2,1], 0, 0, 
                            gbuffer.luminance_buffer_t.clone(), [0,0,0], [1,1,1], 0, 0, 1, Filter::Linear
                        ).unwrap();

                    render_command_buffer_builder
                        // post processing
                        .dispatch([block_dim_x, block_dim_y, 1], pipeline.postprocess.clone(), desc_sets.postprocess_set.clone(), postprocess_pc).unwrap()
                        // copy output to a buffer to enable single frame mode
                        .copy_image(
                            gbuffer.output_buffer.clone(), [0,0,0], 0, 0,
                            gbuffer.prev_swapchain.clone(), [0,0,0], 0, 0,
                            [width, height, 1], 1
                        ).unwrap()
                        // blit the output to the swapchain
                        .blit_image(
                            gbuffer.output_buffer.clone(),       [0,0,0], [width as i32, height as i32, 1], 0, 0, 
                            swapchain_images[image_num].clone(), [0,0,0], [width as i32, height as i32, 1], 0, 0, 1, Filter::Linear
                        ).unwrap();

                    if intersect_pc.frame_idx % luminance_frame_period == 0 {
                        // copy the average frame luminance to a host-accessible buffer every `luminance_frame_period` frames
                        render_command_buffer_builder
                            .copy_image_to_buffer_dimensions(
                                gbuffer.luminance_buffer_t.clone(),
                                luminance_buffer.clone(),
                                [0; 3], [1; 3], 0, 1, 0
                            ).unwrap();
                    }
                    
                    old_position = position;
                    old_forward = forward;
                    old_up = up;

                    render_new_frame = !render_single_frame;
                } else {
                    // if we are rendering in single frame mode, copy the previous rendered frame
                    render_command_buffer_builder
                        .blit_image(
                            gbuffer.prev_swapchain.clone(),      [0,0,0], [width as i32, height as i32, 1], 0, 0, 
                            swapchain_images[image_num].clone(), [0,0,0], [width as i32, height as i32, 1], 0, 0, 1, Filter::Linear
                        ).unwrap();
                }

                let render_command_buffer = render_command_buffer_builder.build().unwrap();

                // build ui command buffer
                let mut ui_cmd_buf_builder = AutoCommandBufferBuilder::new(device.clone(), render_queue.family()).unwrap();
                
                platform.prepare_render(&ui, &surface.window());
                let draw_data = ui.render();
                imgui_renderer.draw_commands(&mut ui_cmd_buf_builder, render_queue.clone(), swapchain_images[image_num].clone(), draw_data).unwrap();
                
                let ui_cmd_buf = ui_cmd_buf_builder.build().unwrap();
                
                cpu_rendering_time = cpu_rendering_start.elapsed().as_secs_f32();

                // execute the command buffers for voxel and ui rendering
                let future = previous_frame_end.take().unwrap()
                    .join(acquire_future)
                    // rendering is done in a series of compute shader commands
                    .then_execute(render_queue.clone(), render_command_buffer).unwrap()
                    .then_execute(render_queue.clone(), ui_cmd_buf).unwrap()
                    // present the frame when rendering is complete
                    .then_swapchain_present(render_queue.clone(), swapchain.clone(), image_num)
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

                // FPS information
                fps.end_sample();

                frame_counts -= 1;
                if frame_counts <= 0 {
                    use std::f32::consts::PI;
                    frame_counts = FRAME_COUNTS;
                }
            },
            ref event => {

                platform.handle_event(imgui.io_mut(), surface.window(), &event);
                
            }
        }

        
        // @INPUT
        if input.update(event) {
            let mut movement = Vector3::new(0.0, 0.0, 0.0);
            let left = up.cross(forward);
            let dt = input_time.elapsed().as_secs_f32();
            input_time = Instant::now();

            let mut speed = 0.02;

            if !ui_wants_keyboard_capture && !render_single_frame {

                if input.key_held(VirtualKeyCode::W) {movement += forward;}
                if input.key_held(VirtualKeyCode::A) {movement += left;}
                if input.key_held(VirtualKeyCode::S) {movement -= forward;}
                if input.key_held(VirtualKeyCode::D) {movement -= left;}
                if input.key_held(VirtualKeyCode::Space) {movement += up;}
                if input.key_held(VirtualKeyCode::LShift) {movement -= up;}
                if input.key_held(VirtualKeyCode::LControl) {speed = 1.0;}
                if input.key_held(VirtualKeyCode::LAlt) {speed = 0.005;}
                
                if input.key_pressed(VirtualKeyCode::Return) {render_new_frame = true;}
                if input.key_pressed(VirtualKeyCode::Tab) {ui_enabled = !ui_enabled;}

            }

            // ensure that movement on the diagonals isn't faster

            if movement.magnitude() > 1e-4 {movement = movement.normalize()};

            position += speed * dt *  movement;

            if let Some((mut mx, mut my)) = input.mouse() {

                let (dx, dy) = (mx - last_mouse_pos.0, my - last_mouse_pos.1);
                // let (dx, dy) = input.mouse_diff();
                // let mx = mx - surface_width as f32 / 2.0;
                // let my = my - surface_height as f32 / 2.0;

                if !ui_wants_mouse_capture && !render_single_frame {

                    if input.mouse_held(0) {

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
                        surface.window().set_cursor_position(winit::dpi::PhysicalPosition::new(width / 2, height / 2)).unwrap();
                        
                        mx = (width / 2) as f32;
                        my = (height / 2) as f32;

                    }
                }

                
                if input.mouse_pressed(1) {
                    
                    let start = Instant::now();
                    
                    let r = svdag_geometry_data.raycast(position, forward, 16, 100.0);
                    
                    if r.hit {

                        let center = position + forward * r.dist;

                        let sphere = create_dag_sphere(10, center, 0.02);
                        if input.key_held(VirtualKeyCode::LControl) {
                            svdag_geometry_data.combine(&sphere, false, false);
                        } else {
                            svdag_geometry_data.subtract(&sphere, false);
                        }
                        let svdag_geometry_buffer = {
                            CpuAccessibleBuffer::<[vox::VChildDescriptor]>::from_iter(device.clone(), BufferUsage::all(), false, svdag_geometry_data.voxels.iter().cloned()).unwrap()
                        };

                        dbuffer.svdag_geometry_buffer = svdag_geometry_buffer;
                        desc_sets = descriptor_sets::DescriptorSets::new(&pipeline, &gbuffer, &dbuffer);

                        println!("Modification Took: {:?}", start.elapsed());
                    }
                }

                last_mouse_pos = (mx, my);
            }

            if input.scroll_diff() < 0.0 && intersect_pc.max_depth > 0 {
                intersect_pc.max_depth -= 1;
            } else if input.scroll_diff() > 0.0 && intersect_pc.max_depth < 15 {
                intersect_pc.max_depth += 1;
            }
            
            if input.key_held(VirtualKeyCode::Escape) {*control_flow = ControlFlow::Exit;}
        }
        
    });
}