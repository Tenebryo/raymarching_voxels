use crate::shaders;

const CHUNK_DIM : usize = 64;

#[derive(Clone)]
pub struct VoxelChunk {
    mat : Vec<u32>,
}

impl VoxelChunk {
    pub fn empty() -> Self {
        Self {
            mat : (0..(CHUNK_DIM * CHUNK_DIM * CHUNK_DIM / 4)).map(|_| 0).collect(),
        }
    }

    pub fn into_shader_type(&self) -> shaders::VoxelChunk {
        let mut mat = [0; CHUNK_DIM * CHUNK_DIM * CHUNK_DIM / 4];

        mat.copy_from_slice(&self.mat);

        shaders::VoxelChunk {
            mat
        }
    }
}