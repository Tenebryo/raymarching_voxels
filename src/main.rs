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
use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice};
use vulkano::pipeline::ComputePipeline;
use vulkano::sync::{GpuFuture, FlushError};
use vulkano::sync;

use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, Subpass, RenderPassAbstract};
use vulkano::image::{SwapchainImage, StorageImage, Dimensions};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::viewport::Viewport;
use vulkano::swapchain::{AcquireError, PresentMode, SurfaceTransform, Swapchain, SwapchainCreationError, ColorSpace, FullscreenExclusive};
use vulkano::swapchain;
use vulkano::format::Format;

use vulkano_win::VkSurfaceBuild;
use winit::window::{WindowBuilder, Window};
use winit::event_loop::{EventLoop, ControlFlow};
use winit::event::{Event, WindowEvent, VirtualKeyCode};

use winit_input_helper::WinitInputHelper;

use cgmath::prelude::*;
use cgmath::{Vector3, Quaternion, InnerSpace, Rotation3, Rad, Rotation};

use std::sync::Arc;
use std::time::{Instant, Duration};
use std::io::{stdout, Write};

use crossterm::{
    ExecutableCommand, execute, Result,
    cursor::MoveUp
};

use opensimplex::OsnContext;

fn main() {
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

    // build rendering compute pipeline
    let render_compute_pipeline = Arc::new({
        // raytracing shader
        use shaders::render_cs;

        let shader = render_cs::Shader::load(device.clone()).unwrap();
        ComputePipeline::new(device.clone(), &shader.main_entry_point(), &()).unwrap()
    });

    println!("Render Pipeline initialized");

    // build a swapchain compatible with the window surface we built earlier
    let (mut swapchain, mut images) = {
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

    let mut dynamic_state = DynamicState { line_width: None, viewports: None, scissors: None, compare_mask: None, write_mask: None, reference: None };

    update_dynamic_state(&images, &mut dynamic_state);

    let mut recreate_swapchain = false;

    let mut previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>);

    let mut fps = Timing::new();
    const FRAME_COUNTS : i32 = 144;
    let mut frame_counts = FRAME_COUNTS;
    let dimensions: [u32; 2] = surface.window().inner_size().into();
    let mut surface_width = dimensions[0];
    let mut surface_height = dimensions[1];
    let p_start = Instant::now();

    let mut render_pc = shaders::RenderPushConstantData {
        cam_o : [0.0, 0.0, 0.0],
        cam_f : [0.0, 0.0, 1.0],
        cam_u : [0.0, 1.0, 0.0],
        vox_chunk_dim : [4, 4, 4],
        render_dist : 1024,
        time : p_start.elapsed().as_secs_f32(),

        // dummy variables for alignment
        _dummy0 : [0;4],
        _dummy1 : [0;4],
    };

    // build the voxel data buffer
    let voxel_data_buffer = unsafe {

        let num_chunks = render_pc.vox_chunk_dim[0] * render_pc.vox_chunk_dim[1] * render_pc.vox_chunk_dim[2];

        // CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, data_iter).unwrap()
        CpuAccessibleBuffer::<[shaders::VoxelChunk]>::uninitialized_array(device.clone(), num_chunks as usize, BufferUsage::all(), false).unwrap()
    };

    println!("Voxel Buffer initialized");

    {
        let ctx = OsnContext::new(123).unwrap();

        let ns = noise::WorleyNoise3D::new(8);
        

        let mut lock = voxel_data_buffer.write().unwrap();

        let mut index = [0,0,0];

        for chunk in lock.iter_mut() {

            let mut chunk_data : Vec<u32> = Vec::with_capacity(64*64*64);
            for z in 0..64 {
                for y in 0..64 {
                    for x in 0..64 {

                        let xn = (x + index[0] * 64) as f64;
                        let yn = (y + index[1] * 64) as f64;
                        let zn = (z + index[2] * 64) as f64;

                        let noise_scale = 16.0;

                        let nx = ctx.noise3(xn / noise_scale, yn / noise_scale, zn / noise_scale);

                        // let nx = ns.sample(xn / noise_scale, yn / noise_scale, zn / noise_scale);

                        chunk_data.push(if nx > 0.0 {1} else {0});
                    }
                }
            }
    
            // let chunk_data = chunk_data.chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect::<Vec<_>>();

            chunk.mat.copy_from_slice(&chunk_data);

            index[0] += 1;
            if index[0] == render_pc.vox_chunk_dim[0] {
                index[0] = 0;
                index[1] += 1;
                if index[1] == render_pc.vox_chunk_dim[1] {
                    index[1] = 0;
                    index[2] += 1;
                }
            }
        }
    }


    println!("Voxel Data initialized");

    // create a list of materials to render
    let material_data_buffer = {
        use shaders::Material;

        let materials = [
            // air material
            Material {albedo : [0.0; 3], transparency: 1.0, emission: [0.0; 3], flags: 0b00000000, roughness: 0.0, _dummy0: [0;12]},
            // solid material
            Material {albedo : [1.0; 3], transparency: 0.0, emission: [0.0; 3], flags: 0b00000001, roughness: 0.0, _dummy0: [0;12]}
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
                intensity : 1.0,
                color : [1.0, 1.0, 1.0],
                _dummy0 : [0;4]
            }
        ];

        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, lights.iter().cloned()).unwrap()
    };

    
    println!("Light Data initialized");
    
    let mut input = WinitInputHelper::new();

    let mut forward = Vector3::new(0.0, 0.0, 1.0);
    let up = Vector3::new(0.0, 1.0, 0.0);
    let mut position = Vector3::new(0.0, 0.0, 0.0);
    let mut input_time = Instant::now();
    let mut pitch = 0.0;
    let mut yaw = 0.0;

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
                    images = new_images;

                    update_dynamic_state(&images, &mut dynamic_state);
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

                let layout = render_compute_pipeline.layout().descriptor_set_layout(0).unwrap();
                let swapchain_image_set = Arc::new(PersistentDescriptorSet::start(layout.clone())
                    .add_image(images[image_num].clone()).unwrap()
                    .add_buffer(voxel_data_buffer.clone()).unwrap()
                    .add_buffer(material_data_buffer.clone()).unwrap()
                    .add_buffer(point_light_data_buffer.clone()).unwrap()
                    .build().unwrap()
                );

                let raymarch_command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), compute_queue.family()).unwrap()
                    .dispatch([(surface_width - 1) / 32 + 1, (surface_height - 1) / 32 + 1, 1], render_compute_pipeline.clone(), swapchain_image_set.clone(), render_pc).unwrap()
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

                // FPS information
                fps.end_sample();

                frame_counts -= 1;
                if frame_counts <= 0 {
                    use std::f32::consts::PI;
                    frame_counts = FRAME_COUNTS;
                    let (t, t_var, fps, fps_var) = fps.stats();
                    stdout().execute(MoveUp(4)).unwrap();
                    println!("FPS: {:.2} +/- {:.2} ({:.3}ms +/- {:.3}ms)", fps, fps_var, t * 1000.0, t_var * 1000.0);
                    println!("Position: {:?}", position);
                    println!("Forward: {:?}", forward);
                    println!("P/Y: {}/{}", 180.0 * pitch / PI, 180.0 * yaw / PI);
                }
            },
            _ => ()
        }
        
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