use crate::vox::VChildDescriptor;
use crate::shaders;

use vulkano::buffer::CpuAccessibleBuffer;
use vulkano::image::ImmutableImage;
use vulkano::format::Format;
use vulkano::sampler::Sampler;

use std::sync::Arc;

/// data buffer struct for scene data and constant textures
/// generally, data that the renderer uses that doesn't need to be recreated when the render surface changes size
pub struct DBuffer {
    pub svdag_geometry_buffer    : Arc<CpuAccessibleBuffer<[VChildDescriptor]>>,
    pub svdag_material_buffer    : Arc<CpuAccessibleBuffer<[u32]>>,
    pub directional_light_buffer : Arc<CpuAccessibleBuffer<[shaders::DirectionalLight]>>,
    pub point_light_buffer       : Arc<CpuAccessibleBuffer<[shaders::PointLight]>>,
    pub spot_light_buffer        : Arc<CpuAccessibleBuffer<[shaders::SpotLight]>>,
    pub brdf_buffer              : Arc<CpuAccessibleBuffer<[f32]>>,
    pub material_buffer          : Arc<CpuAccessibleBuffer<[shaders::Material]>>,
    pub blue_noise_tex           : Arc<ImmutableImage<Format>>,
    pub nst_sampler              : Arc<Sampler>,
    pub skysphere_tex            : Arc<ImmutableImage<Format>>,
    pub lin_sampler              : Arc<Sampler>,
}