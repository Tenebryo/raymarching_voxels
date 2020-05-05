// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

// This example demonstrates how to use the compute capabilities of Vulkan.
//
// While graphics cards have traditionally been used for graphical operations, over time they have
// been more or more used for general-purpose operations as well. This is called "General-Purpose
// GPU", or *GPGPU*. This is what this example demonstrates.

mod timing;
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
use winit::event::{Event, WindowEvent};

use std::sync::Arc;

fn main() {
    // As with other examples, the first step is to create an instance.
    let required_extensions = vulkano_win::required_extensions();
    let instance = Instance::new(None, &required_extensions, None).unwrap();

    // Choose which physical device to use.
    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();
    
    // The objective of this example is to draw a triangle on a window. To do so, we first need to
    // create the window.
    //
    // This is done by creating a `WindowBuilder` from the `winit` crate, then calling the
    // `build_vk_surface` method provided by the `VkSurfaceBuild` trait from `vulkano_win`. If you
    // ever get an error about `build_vk_surface` being undefined in one of your projects, this
    // probably means that you forgot to import this trait.
    //
    // This returns a `vulkano::swapchain::Surface` object that contains both a cross-platform winit
    // window and a cross-platform Vulkan surface that represents the surface of the window.
    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new().build_vk_surface(&event_loop, instance.clone()).unwrap();

    // Choose the queue of the physical device which is going to run our compute operation.
    //
    // The Vulkan specs guarantee that a compliant implementation must provide at least one queue
    // that supports compute operations.
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

    // Now let's get to the actual example.
    //
    // What we are going to do is very basic: we are going to fill a buffer with 64k integers
    // and ask the GPU to multiply each of them by 12.
    //
    // GPUs are very good at parallel computations (SIMD-like operations), and thus will do this
    // much more quickly than a CPU would do. While a CPU would typically multiply them one by one
    // or four by four, a GPU will do it by groups of 32 or 64.
    //
    // Note however that in a real-life situation for such a simple operation the cost of
    // accessing memory usually outweighs the benefits of a faster calculation. Since both the CPU
    // and the GPU will need to access data, there is no other choice but to transfer the data
    // through the slow PCI express bus.

    // We need to create the compute pipeline that describes our operation.
    //
    // If you are familiar with graphics pipeline, the principle is the same except that compute
    // pipelines are much simpler to create.
    let compute_pipeline = Arc::new({
        // raytracing shader
        mod cs {
            vulkano_shaders::shader!{
                ty: "compute",
                src: "
                    #version 450
                    layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;
                    layout(rgba32f, binding = 0) uniform image2D image;
                    layout(binding = 1) buffer VoxelOcc {
                        uint data[];
                    } occ;

                    bool visit(int x, int y, int z) {
                        return false;
                    }

                    void raymarch(vec3 o, vec3 d) {
                        float mx, my, mz, dx, dy, dz, t, dsx, dsy, dsz, ox, oy, oz;
                        int sx, sy, sz, hsx, hsy, hsz, x, y, z;
                        t = 0;
                        
                        sx = int(sign(d.x));
                        sy = int(sign(d.y));
                        sz = int(sign(d.z));
                        
                        x = int(floor(o.x));
                        y = int(floor(o.y));
                        z = int(floor(o.z));
                        
                        dx = 1.0 / abs(d.x);
                        dy = 1.0 / abs(d.y);
                        dz = 1.0 / abs(d.z);
                        
                        hsx = (1 + sx) / 2;
                        hsy = (1 + sy) / 2;
                        hsz = (1 + sz) / 2;
                        
                        mx = abs(x + hsx - o.x) * dx;
                        my = abs(y + hsy - o.y) * dy;
                        mz = abs(z + hsz - o.z) * dz;
                        
                        
                        int mask = 3;
                        
                        int n = 128;
                        int l = 0;
                        float s = 1.0;
                        int scale = 1;
                        
                        while (n >= 0) {
                          
                          if (mx < my) {
                            if (mx < mz) {
                              x += sx;
                              mx += dx;
                            } else {
                              z += sz;
                              mz += dz;
                            }
                          } else {
                            if (my < mz) {
                              y += sy;
                              my += dy;
                            } else {
                              z += sz;
                              mz += dz;
                            }
                          }

                          visit(x,y,z);
                        }
                    }

                    void main() {
                        uint idxX = gl_GlobalInvocationID.x;
                        uint idxY    = gl_GlobalInvocationID.y;

                        ivec2 size = imageSize(image);

                        if (idxX < size.x && idxY < size.y) {
                            imageStore(image, ivec2(idxX,idxY), vec4(idxX / float(size.x), idxY / float(size.y), 0.0, 1.0));
                        }
                    }
                "
            }
        }
        let shader = cs::Shader::load(device.clone()).unwrap();
        ComputePipeline::new(device.clone(), &shader.main_entry_point(), &()).unwrap()
    });

    // Before we can draw on the surface, we have to create what is called a swapchain. Creating
    // a swapchain allocates the color buffers that will contain the image that will ultimately
    // be visible on the screen. These images are returned alongside with the swapchain.
    let (mut swapchain, mut images) = {
        // Querying the capabilities of the surface. When we create the swapchain we can only
        // pass values that are allowed by the capabilities.
        let caps = surface.capabilities(physical).unwrap();
        let usage = caps.supported_usage_flags;

        // The alpha mode indicates how the alpha value of the final image will behave. For example
        // you can choose whether the window will be opaque or transparent.
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();

        // Choosing the internal format that the images will have.
        let format = caps.supported_formats[0].0;

        // The dimensions of the window, only used to initially setup the swapchain.
        // NOTE:
        // On some drivers the swapchain dimensions are specified by `caps.current_extent` and the
        // swapchain size must use these dimensions.
        // These dimensions are always the same as the window dimensions
        //
        // However other drivers dont specify a value i.e. `caps.current_extent` is `None`
        // These drivers will allow anything but the only sensible value is the window dimensions.
        //
        // Because for both of these cases, the swapchain needs to be the window dimensions, we just use that.
        let dimensions: [u32; 2] = surface.window().inner_size().into();

        // Please take a look at the docs for the meaning of the parameters we didn't mention.
        Swapchain::new(device.clone(), surface.clone(), caps.min_image_count, format,
            dimensions, 1, usage, &graphics_queue, SurfaceTransform::Identity, alpha,
            PresentMode::Fifo, FullscreenExclusive::Default, true, ColorSpace::SrgbNonLinear).unwrap()

    };

    // At this point, OpenGL initialization would be finished. However in Vulkan it is not. OpenGL
    // implicitly does a lot of computation whenever you draw. In Vulkan, you have to do all this
    // manually.

    // The next step is to create a *render pass*, which is an object that describes where the
    // output of the graphics pipeline will go. It describes the layout of the images
    // where the colors, depth and/or stencil information will be written.
    let render_pass = Arc::new(vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            // `color` is a custom name we give to the first and only attachment.
            color: {
                // `load: Clear` means that we ask the GPU to clear the content of this
                // attachment at the start of the drawing.
                load: Clear,
                // `store: Store` means that we ask the GPU to store the output of the draw
                // in the actual image. We could also ask it to discard the result.
                store: Store,
                // `format: <ty>` indicates the type of the format of the image. This has to
                // be one of the types of the `vulkano::format` module (or alternatively one
                // of your structs that implements the `FormatDesc` trait). Here we use the
                // same format as the swapchain.
                format: swapchain.format(),
                // TODO:
                samples: 1,
            }
        },
        pass: {
            // We use the attachment named `color` as the one and only color attachment.
            color: [color],
            // No depth-stencil attachment is indicated with empty brackets.
            depth_stencil: {}
        }
    ).unwrap());

    // Dynamic viewports allow us to recreate just the viewport when the window is resized
    // Otherwise we would have to recreate the whole pipeline.
    let mut dynamic_state = DynamicState { line_width: None, viewports: None, scissors: None, compare_mask: None, write_mask: None, reference: None };

    // The render pass we created above only describes the layout of our framebuffers. Before we
    // can draw we also need to create the actual framebuffers.
    //
    // Since we need to draw to multiple images, we are going to create a different framebuffer for
    // each image.
    let mut framebuffers = window_size_dependent_setup(&images, render_pass.clone(), &mut dynamic_state);

    // Initialization is finally finished!

    // In some situations, the swapchain will become invalid by itself. This includes for example
    // when the window is resized (as the images of the swapchain will no longer match the
    // window's) or, on Android, when the application went to the background and goes back to the
    // foreground.
    //
    // In this situation, acquiring a swapchain image or presenting it will return an error.
    // Rendering to an image of that swapchain will not produce any error, but may or may not work.
    // To continue rendering, we need to recreate the swapchain by creating a new swapchain.
    // Here, we remember that we need to do this for the next loop iteration.
    let mut recreate_swapchain = false;

    // In the loop below we are going to submit commands to the GPU. Submitting a command produces
    // an object that implements the `GpuFuture` trait, which holds the resources for as long as
    // they are in use by the GPU.
    //
    // Destroying the `GpuFuture` blocks until the GPU is finished executing it. In order to avoid
    // that, we store the submission of the previous frame here.
    let mut previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>);

    let mut fps = Timing::new();
    const FRAME_COUNTS : i32 = 144;
    let mut frame_counts = FRAME_COUNTS;
    let dimensions: [u32; 2] = surface.window().inner_size().into();
    let mut surface_width = dimensions[0];
    let mut surface_height = dimensions[1];

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                *control_flow = ControlFlow::Exit;
            },
            Event::WindowEvent { event: WindowEvent::Resized(_), .. } => {
                recreate_swapchain = true;
            },
            Event::RedrawEventsCleared => {

                fps.start_sample();
                // It is important to call this function from time to time, otherwise resources will keep
                // accumulating and you will eventually reach an out of memory error.
                // Calling this function polls various fences in order to determine what the GPU has
                // already processed, and frees the resources that are no longer needed.
                previous_frame_end.as_mut().unwrap().cleanup_finished();

                // Whenever the window resizes we need to recreate everything dependent on the window size.
                // In this example that includes the swapchain, the framebuffers and the dynamic state viewport.
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
                    // Because framebuffers contains an Arc on the old swapchain, we need to
                    // recreate framebuffers as well.
                    framebuffers = window_size_dependent_setup(&images, render_pass.clone(), &mut dynamic_state);
                    recreate_swapchain = false;
                }

                // Before we can draw on the output, we have to *acquire* an image from the swapchain. If
                // no image is available (which happens if you submit draw commands too quickly), then the
                // function will block.
                // This operation returns the index of the image that we are allowed to draw upon.
                //
                // This function can block if no image is available. The parameter is an optional timeout
                // after which the function call will return an error.
                let (image_num, suboptimal, acquire_future) = match swapchain::acquire_next_image(swapchain.clone(), None) {
                    Ok(r) => r,
                    Err(AcquireError::OutOfDate) => {
                        recreate_swapchain = true;
                        return;
                    },
                    Err(e) => panic!("Failed to acquire next image: {:?}", e)
                };

                // acquire_next_image can be successful, but suboptimal. This means that the swapchain image
                // will still work, but it may not display correctly. With some drivers this can be when
                // the window resizes, but it may not cause the swapchain to become out of date.
                if suboptimal {
                    recreate_swapchain = true;
                }

                let layout = compute_pipeline.layout().descriptor_set_layout(0).unwrap();
                let swapchain_image_set = Arc::new(PersistentDescriptorSet::start(layout.clone())
                    .add_image(images[image_num].clone()).unwrap()
                    .build().unwrap()
                );

                let raymarch_command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), compute_queue.family()).unwrap()
                    // .dispatch([32,32,1], compute_pipeline.clone(), swapchain_image_set.clone(), ()).unwrap()
                    // .dispatch([surface_width,surface_height,1], compute_pipeline.clone(), swapchain_image_set.clone(), ()).unwrap()
                    .dispatch([(surface_width - 1) / 32 + 1, (surface_height - 1) / 32 + 1, 1], compute_pipeline.clone(), swapchain_image_set.clone(), ()).unwrap()
                    .build().unwrap();

                let future = previous_frame_end.take().unwrap()
                    .join(acquire_future)
                    // rendering is actually done on a compute queue
                    .then_execute(compute_queue.clone(), raymarch_command_buffer).unwrap()

                    // The color output is now expected to contain our triangle. But in order to show it on
                    // the screen, we have to *present* the image by calling `present`.
                    //
                    // This function does not actually present the image immediately. Instead it submits a
                    // present command at the end of the queue. This means that it will only be presented once
                    // the GPU has finished executing the command buffer that draws the triangle.
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

                fps.end_sample();

                frame_counts -= 1;
                if frame_counts <= 0 {
                    frame_counts = FRAME_COUNTS;
                    let (t, t_var, fps, fps_var) = fps.stats();
                    println!("FPS: {:.2} +/- {:.2} ({:.3}ms +/- {:.3}ms)", fps, fps_var, t * 1000.0, t_var * 1000.0);
                }
            },
            _ => ()
        }
    });
}


/// This method is called once during initialization, then again whenever the window is resized
fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    dynamic_state: &mut DynamicState
) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
    let dimensions = images[0].dimensions();

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [dimensions[0] as f32, dimensions[1] as f32],
        depth_range: 0.0 .. 1.0,
    };
    dynamic_state.viewports = Some(vec!(viewport));

    images.iter().map(|image| {
        Arc::new(
            Framebuffer::start(render_pass.clone())
                .add(image.clone()).unwrap()
                .build().unwrap()
        ) as Arc<dyn FramebufferAbstract + Send + Sync>
    }).collect::<Vec<_>>()
}