
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
    pub light_index_buffer : Arc<StorageImage<Format>>,
    pub noise_0_buffer : Arc<StorageImage<Format>>,
    pub noise_1_buffer : Arc<StorageImage<Format>>,
    pub noise_2_buffer : Arc<StorageImage<Format>>,

    pub pre_depth_buffer : Arc<StorageImage<Format>>,
    pub depth_buffer : Arc<StorageImage<Format>>,

    pub alpha_buffer : Arc<StorageImage<Format>>,

    pub material0_buffer : Arc<StorageImage<Format>>,
    pub material1_buffer : Arc<StorageImage<Format>>,
    pub position0_buffer : Arc<StorageImage<Format>>,
    pub position1_buffer : Arc<StorageImage<Format>>,
    pub temporal_buffer : Arc<StorageImage<Format>>,
    pub normal0_buffer : Arc<StorageImage<Format>>,
    pub normal0b_buffer : Arc<StorageImage<Format>>,
    pub normal1_buffer : Arc<StorageImage<Format>>,
    pub normal1b_buffer : Arc<StorageImage<Format>>,
    pub ldir0_buffer : Arc<StorageImage<Format>>,
    pub ldir1_buffer : Arc<StorageImage<Format>>,
    pub light0_buffer : Arc<StorageImage<Format>>,
    pub light1_buffer : Arc<StorageImage<Format>>,
    
    pub hdr_light_buffer : Arc<StorageImage<Format>>,

    pub luminance_buffer : Arc<StorageImage<Format>>,
    pub luminance_buffer_t : Arc<StorageImage<Format>>,

    pub reprojection_dist_buffer : Arc<StorageImage<Format>>,
    pub stratum_index_buffer : Arc<StorageImage<Format>>,
    pub stratum_pos_buffer : Arc<StorageImage<Format>>,

    pub prev_col_buffer : Arc<StorageImage<Format>>,
    pub prev_pos_buffer : Arc<StorageImage<Format>>,
    pub reprojected_col_buffer : Arc<StorageImage<Format>>,
    pub reprojected_pos_buffer : Arc<StorageImage<Format>>,
    pub prev_cnt_buffer : Arc<StorageImage<Format>>,
    pub reprojected_cnt_buffer : Arc<StorageImage<Format>>,
    pub postprocess_input_buffer : Arc<StorageImage<Format>>,

    pub prev_swapchain : Arc<StorageImage<Format>>,
    
    pub iteration_count_buffer : Arc<StorageImage<Format>>,

    pub output_buffer : Arc<StorageImage<Format>>,

    pub sample_mask_buffer : Arc<StorageImage<Format>>,
    
    pub pre_trace_width : u32,
    pub pre_trace_height : u32,
}

impl GBuffer {
    // When the window is resized, the various screen-shaped buffers need to be resized
    // this is done often and is a bit repetitive, so it is its own function
    pub fn new_buffers(
        device : Arc<Device>, queue_family : QueueFamily, width : u32, height : u32, num_temp_buffers : usize, stratum_size : u32,
    ) -> GBuffer {

        let temp_buffers = (0..num_temp_buffers)
                .map(|_| { StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, Format::R32G32B32A32Sfloat, [queue_family].iter().cloned()).unwrap() })
                .collect::<Vec<_>>();

        let pre_trace_width        = (width / PRE_DEPTH_ESTIMATE_PATCH) + 1;
        let pre_trace_height       = (height / PRE_DEPTH_ESTIMATE_PATCH) + 1;

        let pre_depth_buffer            = StorageImage::new(device.clone(), Dimensions::Dim2d{width : pre_trace_width, height : pre_trace_height}, Format::R32Sfloat, [queue_family].iter().cloned()).unwrap();
        let depth_buffer                = StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, Format::R32Sfloat, [queue_family].iter().cloned()).unwrap();
        let alpha_buffer                = StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, Format::R32Sfloat, [queue_family].iter().cloned()).unwrap();

        let index_buffer                = StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, Format::R32Uint, [queue_family].iter().cloned()).unwrap();
        let light_index_buffer          = StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, Format::R32Uint, [queue_family].iter().cloned()).unwrap();

        let noise_0_buffer              = StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, Format::R32G32B32A32Sfloat, [queue_family].iter().cloned()).unwrap();
        let noise_1_buffer              = StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, Format::R32G32B32A32Sfloat, [queue_family].iter().cloned()).unwrap();
        let noise_2_buffer              = StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, Format::R32G32B32A32Sfloat, [queue_family].iter().cloned()).unwrap();

        let material0_buffer            = StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, Format::R32Uint, [queue_family].iter().cloned()).unwrap();
        let material1_buffer            = StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, Format::R32Uint, [queue_family].iter().cloned()).unwrap();

        let normal0_buffer              = StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, Format::R32G32B32A32Sfloat, [queue_family].iter().cloned()).unwrap();
        let normal0b_buffer             = StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, Format::R32G32B32A32Sfloat, [queue_family].iter().cloned()).unwrap();
        let normal1_buffer              = StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, Format::R32G32B32A32Sfloat, [queue_family].iter().cloned()).unwrap();
        let normal1b_buffer             = StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, Format::R32G32B32A32Sfloat, [queue_family].iter().cloned()).unwrap();
        let ldir0_buffer                = StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, Format::R32G32B32A32Sfloat, [queue_family].iter().cloned()).unwrap();
        let ldir1_buffer                = StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, Format::R32G32B32A32Sfloat, [queue_family].iter().cloned()).unwrap();
        let position0_buffer            = StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, Format::R32G32B32A32Sfloat, [queue_family].iter().cloned()).unwrap();
        let position1_buffer            = StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, Format::R32G32B32A32Sfloat, [queue_family].iter().cloned()).unwrap();
        let temporal_buffer             = StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, Format::R32G32B32A32Sfloat, [queue_family].iter().cloned()).unwrap();
        let light0_buffer               = StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, Format::R32G32B32A32Sfloat, [queue_family].iter().cloned()).unwrap();
        let light1_buffer               = StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, Format::R32G32B32A32Sfloat, [queue_family].iter().cloned()).unwrap();

        let hdr_light_buffer            = StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, Format::R32G32B32A32Sfloat, [queue_family].iter().cloned()).unwrap();

        let luminance_buffer            = StorageImage::new(device.clone(), Dimensions::Dim2d{width : 32, height : 32}, Format::R32G32B32A32Sfloat, [queue_family].iter().cloned()).unwrap();
        let luminance_buffer_t          = StorageImage::new(device.clone(), Dimensions::Dim2d{width : 32, height : 32}, Format::R32G32B32A32Sfloat, [queue_family].iter().cloned()).unwrap();
        
        let reprojection_dist_buffer    = StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, Format::R32G32B32A32Sfloat, [queue_family].iter().cloned()).unwrap();

        let prev_pos_buffer             = StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, Format::R32G32B32A32Sfloat, [queue_family].iter().cloned()).unwrap();
        let reprojected_pos_buffer      = StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, Format::R32G32B32A32Sfloat, [queue_family].iter().cloned()).unwrap();
        
        let prev_col_buffer             = StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, Format::R32G32B32A32Sfloat, [queue_family].iter().cloned()).unwrap();
        let reprojected_col_buffer      = StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, Format::R32G32B32A32Sfloat, [queue_family].iter().cloned()).unwrap();
        
        let prev_cnt_buffer             = StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, Format::R32Uint, [queue_family].iter().cloned()).unwrap();
        let reprojected_cnt_buffer      = StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, Format::R32Uint, [queue_family].iter().cloned()).unwrap();

        let stratum_index_buffer        = StorageImage::new(device.clone(), Dimensions::Dim2d{width : width / stratum_size, height : height / stratum_size}, Format::R32G32Uint, [queue_family].iter().cloned()).unwrap();
        let stratum_pos_buffer          = StorageImage::new(device.clone(), Dimensions::Dim2d{width : width / stratum_size, height : height / stratum_size}, Format::R32G32B32A32Sfloat, [queue_family].iter().cloned()).unwrap();
        
        let postprocess_input_buffer    = StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, Format::R32G32B32A32Sfloat, [queue_family].iter().cloned()).unwrap();
        
        let iteration_count_buffer      = StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, Format::R32G32B32A32Uint, [queue_family].iter().cloned()).unwrap();

        let prev_swapchain              = StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, Format::R32G32B32A32Sfloat, [queue_family].iter().cloned()).unwrap();

        let output_buffer               = StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, Format::R32G32B32A32Sfloat, [queue_family].iter().cloned()).unwrap();

        let sample_mask_buffer                 = StorageImage::new(device.clone(), Dimensions::Dim2d{width, height}, Format::R32Uint, [queue_family].iter().cloned()).unwrap();

        GBuffer {
            temp_buffers,
            index_buffer,
            light_index_buffer,
            noise_0_buffer,
            noise_1_buffer,
            noise_2_buffer,
            pre_depth_buffer,
            depth_buffer,
            position0_buffer,
            position1_buffer,
            temporal_buffer,
            material0_buffer,
            material1_buffer,
            normal0_buffer,
            normal0b_buffer,
            normal1_buffer,
            normal1b_buffer,
            alpha_buffer,
            ldir0_buffer,
            ldir1_buffer,
            light0_buffer,
            light1_buffer,
            hdr_light_buffer,
            luminance_buffer,
            luminance_buffer_t,
            reprojection_dist_buffer,
            stratum_index_buffer,
            stratum_pos_buffer,
            prev_col_buffer,
            prev_pos_buffer,
            reprojected_col_buffer,
            reprojected_pos_buffer,
            prev_cnt_buffer,
            reprojected_cnt_buffer,
            postprocess_input_buffer,
            iteration_count_buffer,
            prev_swapchain,
            pre_trace_width,
            pre_trace_height,
            output_buffer,
            sample_mask_buffer,
        }
    }
}