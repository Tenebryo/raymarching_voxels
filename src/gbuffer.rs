
use vulkano::instance::QueueFamily;

use vulkano::device::Device;
use vulkano::image::{StorageImage, Dimensions};
use vulkano::format::Format;

use std::sync::Arc;

pub const PRE_DEPTH_ESTIMATE_PATCH : u32 = 8;

// container for all of the buffers needed for the rendering pipeline
pub struct GBuffer {
    pub temp_buffers : Vec<Arc<StorageImage<Format>>>,
    
    pub index_buffer : Arc<StorageImage<Format>>,
    pub seed_buffer : Arc<StorageImage<Format>>,

    pub pre_depth_buffer : Arc<StorageImage<Format>>,
    pub depth_buffer : Arc<StorageImage<Format>>,

    pub indirect_alpha_buffer : Arc<StorageImage<Format>>,
    pub direct_alpha_buffer : Arc<StorageImage<Format>>,

    pub position_buffer : Arc<StorageImage<Format>>,
    pub temporal_buffer : Arc<StorageImage<Format>>,
    pub normal_buffer : Arc<StorageImage<Format>>,
    pub indirect_lighting_buffer : Arc<StorageImage<Format>>,
    pub direct_lighting_buffer : Arc<StorageImage<Format>>,

    pub pre_trace_width : u32,
    pub pre_trace_height : u32,
}

impl GBuffer {
    // When the window is resized, the various screen-shaped buffers need to be resized
    // this is done often and is a bit repetitive, so it is its own function
    pub fn new_buffers(
        device : Arc<Device>, queue_family : QueueFamily, width : u32, height : u32, num_temp_buffers : usize,
    ) -> GBuffer {

        let temp_buffers = (0..num_temp_buffers)
                .map(|_| { StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, Format::R32G32B32A32Sfloat, [queue_family].iter().cloned()).unwrap() })
                .collect::<Vec<_>>();

        let pre_trace_width = (width / PRE_DEPTH_ESTIMATE_PATCH) + 1;
        let pre_trace_height = (height / PRE_DEPTH_ESTIMATE_PATCH) + 1;

        let pre_depth_buffer = StorageImage::new(device.clone(), Dimensions::Dim2d{width : pre_trace_width, height : pre_trace_height}, Format::R32Sfloat, [queue_family].iter().cloned()).unwrap();
        let depth_buffer = StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, Format::R32Sfloat, [queue_family].iter().cloned()).unwrap();
        let indirect_alpha_buffer = StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, Format::R32Sfloat, [queue_family].iter().cloned()).unwrap();
        let direct_alpha_buffer = StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, Format::R32Sfloat, [queue_family].iter().cloned()).unwrap();

        let index_buffer = StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, Format::R32Uint, [queue_family].iter().cloned()).unwrap();
        let seed_buffer = StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, Format::R32Uint, [queue_family].iter().cloned()).unwrap();

        let normal_buffer            = StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, Format::R32G32B32A32Sfloat, [queue_family].iter().cloned()).unwrap();
        let position_buffer          = StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, Format::R32G32B32A32Sfloat, [queue_family].iter().cloned()).unwrap();
        let temporal_buffer          = StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, Format::R32G32B32A32Sfloat, [queue_family].iter().cloned()).unwrap();
        let indirect_lighting_buffer = StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, Format::R32G32B32A32Sfloat, [queue_family].iter().cloned()).unwrap();
        let direct_lighting_buffer   = StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, Format::R32G32B32A32Sfloat, [queue_family].iter().cloned()).unwrap();


        GBuffer {
            temp_buffers,
            index_buffer,
            seed_buffer,
            pre_depth_buffer,
            depth_buffer,
            position_buffer,
            temporal_buffer,
            normal_buffer,
            indirect_alpha_buffer,
            direct_alpha_buffer,
            indirect_lighting_buffer,
            direct_lighting_buffer,
            pre_trace_width,
            pre_trace_height,
        }
    }
}