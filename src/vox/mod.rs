mod raycast;
mod csg;
mod voxelize;
mod tests;

use cgmath::Vector2;
use cgmath::Vector3;
use cgmath::InnerSpace;
use cgmath::prelude::*;


use std::collections::HashMap;
use std::collections::HashSet;

use serde::{Serialize, Deserialize};

pub const MAX_DAG_DEPTH : usize = 16;

type Vec3 = Vector3<f32>;
type Vec2 = Vector2<f32>;

#[derive(Copy, Clone, PartialEq, Debug, Default, Serialize, Deserialize)]
pub struct Material {
    pub albedo : [f32; 3],
    pub metalness : f32,
    pub emission : [f32; 3],
    pub roughness : f32,
}

#[derive(Debug, Clone)]
pub struct Triangle {
    pub points : [Vec3; 3],
    pub uv : [Vec2; 3],
    pub normal : Vec3,
    pub mat    : u16,
}

impl Default for Triangle {
    fn default() -> Self {
        Triangle {
            points : [Vec3::zero(); 3],
            uv : [Vec2::zero(); 3],
            normal : Vec3::zero(),
            mat : 0,
        }
    }
}

impl Triangle {
    fn area(&self) -> f32 {
        // calculate the area of the triangle using heron's formula
        let a = self.points[0].distance(self.points[1]);
        let b = self.points[1].distance(self.points[2]);
        let c = self.points[2].distance(self.points[0]);

        let s = 0.5 * (a + b + c);

        (s * (s - a) * (s - b) * (s - c)).sqrt()
    }

    fn pos_center(&self) -> Vec3 {
        (self.points[0] + self.points[1] + self.points[2]) / 3.0
    }

    fn uv_center(&self) -> Vec2 {
        (self.uv[0] + self.uv[1] + self.uv[2]) / 3.0
    }
}

// this is a redefinition of the type in the voxel.glsl shader.
// these redefinition shenanigans are necessary because serde can't quite derive
// serialize/deserialize for types in another module that are used in Vec fields
#[derive(Copy, Clone, Serialize, Deserialize, Debug, Hash, PartialEq, Eq)]
#[repr(C)]
pub struct VChildDescriptor {
    pub sub_voxels : [i32; 8],
}

#[derive(Serialize, Deserialize, Clone)]
pub struct VoxelChunk {
    pub voxels : Vec<VChildDescriptor>,
    #[serde(default)]
    pub lod_materials : Vec<u32>,
}

const fn num_bits<T>() -> usize { std::mem::size_of::<T>() * 8 }

fn log_2(x: usize) -> u32 {
    assert!(x > 0);
    num_bits::<usize>() as u32 - x.leading_zeros() - 1
}

fn array3d_get<S : Copy>(a : &[S], d : [usize;3], i : [usize;3]) -> Option<S> {
    if i[0] < d[0] && i[1] < d[1] && i[2] < d[2] {
        Some(a[i[0] + d[0] * (i[1] + d[1] * i[2])])
    } else {
        None
    }
}

impl VoxelChunk {
    /// Construct an empty 
    pub fn empty() -> Self {
        // we must populate an empty root voxel;
        Self {
            voxels : vec![VChildDescriptor{
                sub_voxels : [0;8],
            }],
            lod_materials : vec![],
        }
    }

    /// Return the number of unique DAG nodes in this chunk
    pub fn len(&self) -> usize {
        self.voxels.len()
    }

    /// process a 3D array (`data` with dimensions `dim`) into a SVDAG, mapping values to materials using `f`
    pub fn from_dense_voxels(data : &[i32], dim : [usize; 3]) -> Self {
        
        assert!(dim[0] > 0 && dim[1] > 0 && dim[2] > 0);

        let depth = log_2(dim.iter().cloned().max().unwrap_or(0) - 1) as usize + 1;

        assert!(depth < MAX_DAG_DEPTH, "Depth is too large: {} >= {}", depth, MAX_DAG_DEPTH);

        let size = 1 << depth;

        fn recursive_create_dense(
            s : &mut VoxelChunk, d : usize, min : [usize; 3], size : usize, 
            data : &[i32], dim : [usize; 3],
            dedup : &mut HashMap<VChildDescriptor, i32>
        ) -> i32 {
            if min[0] >= dim[0] || min[1] >= dim[1] || min[2] >= dim[2] {
                // air if the voxel does not intersect the voxel data
                return 0;
            }
            
            if size <= 1 {
                // once we reach size 1, take the material from the data
                let v = data[min[0] + dim[0] * (min[1] + dim[1] * min[2])];
                // prevent awful fractal voxel graphs
                return -v.abs();
            }

            const BOX_OFFSETS : [[usize; 3]; 8] = [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1]
            ];

            let mut voxel = VChildDescriptor{
                sub_voxels : [0; 8],
            };

            let half_size = size >> 1;

            let mut is_uniform = true;

            for i in 0..8 {
                let bmin = [
                    min[0] + BOX_OFFSETS[i][0] * half_size,
                    min[1] + BOX_OFFSETS[i][1] * half_size,
                    min[2] + BOX_OFFSETS[i][2] * half_size
                ];

                voxel.sub_voxels[i] = recursive_create_dense(s, d - 1, bmin, half_size, data, dim, dedup);

                if voxel.sub_voxels[i] != voxel.sub_voxels[0] || voxel.sub_voxels[i] > 0 {
                    // the subvoxels are not all the same leaf node, so this voxel is not uniform
                    is_uniform = false;
                }
            }

            if is_uniform {
                return voxel.sub_voxels[0];
            }

            if let Some(&id) = dedup.get(&voxel) {
                // this node is a duplicate
                id
            } else {
                // this node is new, so add it
                s.voxels.push(voxel);
                let id = s.voxels.len() as i32;
                dedup.insert(voxel, id);
                id
            }
        }

        let mut chunk = VoxelChunk::empty();
        chunk.voxels.clear();
        // we build a list of unique voxels and store them in here
        let mut dedup : HashMap<VChildDescriptor, i32> = HashMap::new();

        recursive_create_dense(&mut chunk, depth, [0,0,0], size, data, dim, &mut dedup);

        chunk.voxels.reverse();

        //fixup the subvoxel pointers (we reversed the order)
        let n = chunk.voxels.len() as i32;
        for i in 0..(chunk.voxels.len()) {
            for j in 0..8 {
                let sv = chunk.voxels[i].sub_voxels[j];
                if sv > 0 {
                    let svi = n - sv + 1;
                    chunk.voxels[i].sub_voxels[j] = svi;
                }
            }
        }

        chunk
    }

    /// process an implicit 3D array into a DAG.
    pub fn from_dense_implicit<F : FnMut(usize, usize, usize) -> i32>(depth : usize, mut implicit : F) -> Self {

        assert!(depth < MAX_DAG_DEPTH, "Depth is too large: {} >= {}", depth, MAX_DAG_DEPTH);

        let size = 1 << depth;

        fn recursive_create_dense_implicit<F : FnMut(usize, usize, usize) -> i32>(
            s : &mut VoxelChunk, min : [usize; 3], size : usize, implicit : &mut F,
            dedup : &mut HashMap<VChildDescriptor, i32>
        ) -> i32 {

            if size <= 1 {
                // once we reach size 1, evaluate the material at the implicit surface
                let v = implicit(min[0], min[1], min[2]);
                // prevent awful fractal voxel graphs
                return -v.abs();
            }

            const BOX_OFFSETS : [[usize; 3]; 8] = [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1]
            ];

            let mut voxel = VChildDescriptor{
                sub_voxels : [0; 8],
            };

            let half_size = size >> 1;

            let mut is_uniform = true;

            for i in 0..8 {
                let bmin = [
                    min[0] + BOX_OFFSETS[i][0] * half_size,
                    min[1] + BOX_OFFSETS[i][1] * half_size,
                    min[2] + BOX_OFFSETS[i][2] * half_size
                ];

                voxel.sub_voxels[i] = recursive_create_dense_implicit(s, bmin, half_size, implicit, dedup);

                if voxel.sub_voxels[i] != voxel.sub_voxels[0] || voxel.sub_voxels[i] > 0 {
                    // the subvoxels are not all the same leaf node, so this voxel is not uniform
                    is_uniform = false;
                }
            }

            
            if is_uniform {
                return voxel.sub_voxels[0];
            }

            if let Some(&id) = dedup.get(&voxel) {
                // this node is a duplicate
                id
            } else {
                // this node is new, so add it
                s.voxels.push(voxel);
                let id = s.voxels.len() as i32;
                dedup.insert(voxel, id);
                id
            }
        }

        let mut chunk = VoxelChunk::empty();
        chunk.voxels.clear();
        // we build a list of unique voxels and store them in here
        let mut dedup : HashMap<VChildDescriptor, i32> = HashMap::new();

        recursive_create_dense_implicit(&mut chunk, [0,0,0], size, &mut implicit, &mut dedup);

        chunk.voxels.reverse();

        //fixup the subvoxel pointers (we reversed the order)
        let n = chunk.voxels.len() as i32;
        for i in 0..(chunk.voxels.len()) {
            for j in 0..8 {
                let sv = chunk.voxels[i].sub_voxels[j];
                if sv > 0 {
                    let svi = n - sv + 1;
                    chunk.voxels[i].sub_voxels[j] = svi;
                }
            }
        }

        chunk
    }

    /// process a distance equation array into a DAG.
    pub fn from_distance_equation<F : FnMut(f32, f32, f32) -> f32>(depth : usize, mut implicit : F) -> Self {

        assert!(depth < MAX_DAG_DEPTH, "Depth is too large: {} >= {}", depth, MAX_DAG_DEPTH);

        let size = 1 << depth;

        fn recurse_distance_equation<F : FnMut(f32, f32, f32) -> f32>(
            s : &mut VoxelChunk, min : [usize; 3], size : usize, implicit : &mut F, rscale : f32,
            dedup : &mut HashMap<VChildDescriptor, i32>
        ) -> i32 {

            const SQRT_THREE : f32 = 1.732050807568877293527446341505872366942805253810380628055;

            let v = implicit(
                rscale * (min[0] as f32 + 0.5 * size as f32),
                rscale * (min[1] as f32 + 0.5 * size as f32),
                rscale * (min[2] as f32 + 0.5 * size as f32)
            );

            let bounding_radius = rscale * size as f32 * SQRT_THREE;

            if size <= 1 {
                // once we reach size 1, check if the object intersects the implicit region
                if min[0] == 0 && min[1] == 0 && min[2] == 0 {
                    // println!("maybe intersection {} < {}", v, bounding_radius);
                }
                return if v < bounding_radius { -1 } else { 0 };
            }
            
            if v > bounding_radius {
                // the voxel does not intersect the cube at all based on the distance equation
                // println!("no intersection {} {}", v, bounding_radius);

                return 0;
            }

            const BOX_OFFSETS : [[usize; 3]; 8] = [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1]
            ];

            let mut voxel = VChildDescriptor{
                sub_voxels : [0; 8],
            };

            let half_size = size >> 1;

            let mut is_uniform = true;

            for i in 0..8 {
                let bmin = [
                    min[0] + BOX_OFFSETS[i][0] * half_size,
                    min[1] + BOX_OFFSETS[i][1] * half_size,
                    min[2] + BOX_OFFSETS[i][2] * half_size
                ];

                voxel.sub_voxels[i] = recurse_distance_equation(s, bmin, half_size, implicit, rscale, dedup);

                if voxel.sub_voxels[i] != voxel.sub_voxels[0] || voxel.sub_voxels[i] > 0 {
                    // the subvoxels are not all the same leaf node, so this voxel is not uniform
                    is_uniform = false;
                }
            }

            
            if is_uniform {
                return voxel.sub_voxels[0];
            }

            if let Some(&id) = dedup.get(&voxel) {
                // this node is a duplicate
                id
            } else {
                // this node is new, so add it
                s.voxels.push(voxel);
                let id = s.voxels.len() as i32;
                dedup.insert(voxel, id);
                id
            }
        }

        let mut chunk = VoxelChunk::empty();
        chunk.voxels.clear();
        // we build a list of unique voxels and store them in here
        let mut dedup : HashMap<VChildDescriptor, i32> = HashMap::new();

        recurse_distance_equation(&mut chunk, [0,0,0], size, &mut implicit, 1.0 / (size as f32), &mut dedup);

        chunk.voxels.reverse();

        //fixup the subvoxel pointers (we reversed the order)
        let n = chunk.voxels.len() as i32;
        for i in 0..(chunk.voxels.len()) {
            for j in 0..8 {
                let sv = chunk.voxels[i].sub_voxels[j];
                if sv > 0 {
                    let svi = n - sv + 1;
                    chunk.voxels[i].sub_voxels[j] = svi;
                }
            }
        }

        chunk
    }


    /// process a distance equation array into a DAG.
    pub fn from_intersection_test<F : FnMut(Vec3, f32) -> bool>(depth : usize, mut intersect_test : F) -> Self {

        assert!(depth < MAX_DAG_DEPTH, "Depth is too large: {} >= {}", depth, MAX_DAG_DEPTH);

        let size = 1 << depth;

        fn recurse_intersection_test<F : FnMut(Vec3, f32) -> bool>(
            s : &mut VoxelChunk, min : [usize; 3], size : usize, intersect_test : &mut F, rscale : f32,
            dedup : &mut HashMap<VChildDescriptor, i32>
        ) -> i32 {

            const SQRT_THREE : f32 = 1.732050807568877293527446341505872366942805253810380628055;

            let intersects = intersect_test(
                Vec3::new(
                    rscale * (min[0] as f32 + 0.5 * size as f32),
                    rscale * (min[1] as f32 + 0.5 * size as f32),
                    rscale * (min[2] as f32 + 0.5 * size as f32)
                ),
                0.5 * rscale * size as f32
            );

            if size <= 1 {
                // once we reach size 1, check if the object intersects the implicit region
                if min[0] == 0 && min[1] == 0 && min[2] == 0 {
                    // println!("maybe intersection {} < {}", v, bounding_radius);
                }
                return if intersects { -1 } else { 0 };
            }
            
            if !intersects {
                // the voxel does not intersect the cube at all based on the distance equation
                // println!("no intersection {} {}", v, bounding_radius);

                return 0;
            }

            const BOX_OFFSETS : [[usize; 3]; 8] = [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1]
            ];

            let mut voxel = VChildDescriptor{
                sub_voxels : [0; 8],
            };

            let half_size = size >> 1;

            let mut is_uniform = true;

            for i in 0..8 {
                let bmin = [
                    min[0] + BOX_OFFSETS[i][0] * half_size,
                    min[1] + BOX_OFFSETS[i][1] * half_size,
                    min[2] + BOX_OFFSETS[i][2] * half_size
                ];

                voxel.sub_voxels[i] = recurse_intersection_test(s, bmin, half_size, intersect_test, rscale, dedup);

                if voxel.sub_voxels[i] != voxel.sub_voxels[0] || voxel.sub_voxels[i] > 0 {
                    // the subvoxels are not all the same leaf node, so this voxel is not uniform
                    is_uniform = false;
                }
            }

            
            if is_uniform {
                return voxel.sub_voxels[0];
            }

            if let Some(&id) = dedup.get(&voxel) {
                // this node is a duplicate
                id
            } else {
                // this node is new, so add it
                s.voxels.push(voxel);
                let id = s.voxels.len() as i32;
                dedup.insert(voxel, id);
                id
            }
        }

        let mut chunk = VoxelChunk::empty();
        chunk.voxels.clear();
        // we build a list of unique voxels and store them in here
        let mut dedup : HashMap<VChildDescriptor, i32> = HashMap::new();

        recurse_intersection_test(&mut chunk, [0,0,0], size, &mut intersect_test, 1.0 / (size as f32), &mut dedup);

        chunk.voxels.reverse();

        //fixup the subvoxel pointers (we reversed the order)
        let n = chunk.voxels.len() as i32;
        for i in 0..(chunk.voxels.len()) {
            for j in 0..8 {
                let sv = chunk.voxels[i].sub_voxels[j];
                if sv > 0 {
                    let svi = n - sv + 1;
                    chunk.voxels[i].sub_voxels[j] = svi;
                }
            }
        }

        chunk
    }

    /// Convert an obj file into a voxel chunk format.
    /// Only places voxels intersect triangles will be made solid
    pub fn from_mesh<F : FnMut(u64)>(depth : usize, triangles: &[Triangle], corner : Vec3, size : f32, progress_callback : &mut F) -> VoxelChunk {
        assert!(depth < MAX_DAG_DEPTH, "Depth is too large: {} >= {}", depth, MAX_DAG_DEPTH);

        fn recursive_create_shell<F : FnMut(u64)>(
            s : &mut VoxelChunk, d : usize, md : usize, min : Vec3, size : f32, 
            tris : &[Triangle], indexes : &mut Vec<usize>, start : usize, 
            dedup : &mut HashMap<VChildDescriptor, i32>,
            counts : &mut HashMap<u16, i32>,
            progress_callback : &mut F
        ) -> i32 {

            if d == 0 {
                // if we reach the max resolution, check if there are intersecting triangles
                if start < indexes.len() {
                    // solid material

                    // let m = tris[indexes[start]].mat;

                    let m = mode(indexes[start..].iter().map(|&i| tris[i].mat));

                    return - (m as i32 + 1);
                } else {
                    // air
                    return 0;
                }
            }

            // there are no intersecting triangles, so the voxel is empty
            if start == indexes.len() {
                // air
                return 0;
            }

            const BOX_OFFSETS : [Vec3; 8] = [
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(1.0, 0.0, 1.0),
                Vec3::new(0.0, 1.0, 1.0),
                Vec3::new(1.0, 1.0, 1.0)
            ];

            let end = indexes.len();

            let mut voxel = VChildDescriptor{
                sub_voxels : [0; 8],
            };

            for i in 0..8 {
                let bmin = min + BOX_OFFSETS[i] * (size * 0.5);
                let bmax = bmin + Vec3::new(size * 0.5, size * 0.5, size * 0.5);

                for j in start..end {
                    if aabb_triangle_test(bmin, bmax, &tris[indexes[j]]) {
                        indexes.push(indexes[j]);
                    }
                }

                voxel.sub_voxels[i] = recursive_create_shell(s, d - 1, md, bmin, size * 0.5, tris, indexes, end, dedup, counts, progress_callback);

                indexes.truncate(end);
            }

            if md - d == 4 {
                progress_callback(8*8*8*8);
            }

            if let Some(&id) = dedup.get(&voxel) {
                // this node is a duplicate
                id
            } else {
                // this node is new, so add it
                s.voxels.push(voxel);
                let id = s.voxels.len() as i32;
                dedup.insert(voxel, id);
                id
            }
        }

        let mut chunk = VoxelChunk::empty();
        chunk.voxels.clear();
        // we build a list of unique voxels and store them in here
        let mut dedup : HashMap<VChildDescriptor, i32> = HashMap::new();
        // a reused hashmap to count most common material
        let mut counts : HashMap<u16, i32> = HashMap::new();
        // indexes acts as a simple growing allocator for the recursion
        let mut indexes = (0..(triangles.len())).collect::<Vec<_>>();

        recursive_create_shell(&mut chunk, depth, depth, corner, size, &triangles, &mut indexes, 0, &mut dedup, &mut counts, progress_callback);

        chunk.voxels.reverse();

        //fixup the subvoxel pointers (we reversed the order)
        let n = chunk.voxels.len() as i32;
        for i in 0..(chunk.voxels.len()) {
            for j in 0..8 {
                let sv = chunk.voxels[i].sub_voxels[j];
                if sv > 0 {
                    let svi = n - sv + 1;
                    chunk.voxels[i].sub_voxels[j] = svi;
                }
            }
        }

        chunk
    }

    pub fn shift_indexes(&mut self, s : usize) {
        for i in 0..(self.voxels.len()) {
            for j in 0..8 {
                // check that it is a subvoxel
                if self.voxels[i].sub_voxels[j] > 0 {
                    // permute the subvoxel index
                    self.voxels[i].sub_voxels[j] += s as i32;
                }
            }
        }
    }

    /// Permute the order of each node's subvoxels.
    /// This operation can be used to apply flips, rotates, and other operations
    pub fn permute(&mut self, permutation : [u8; 8]) {
        let perm = [
            permutation[0] as usize, 
            permutation[1] as usize, 
            permutation[2] as usize, 
            permutation[3] as usize, 
            permutation[4] as usize, 
            permutation[5] as usize, 
            permutation[6] as usize, 
            permutation[7] as usize
        ];

        for i in 0..(self.len()) {
            let mut j = 0;
            let t = self.voxels[i].sub_voxels[j];
            for _ in 0..8 {
                let k = perm[j];
                self.voxels[i].sub_voxels[j] = self.voxels[i].sub_voxels[k];
                j = k;
            }
            self.voxels[i].sub_voxels[j] = t;
        }
    }

    /// Combine 8 voxel chunks into a single chunk.
    /// This operation simply concatenates the data and does not compress it
    pub fn combine_8_voxel_chunks<'a>(subvoxels : [&'a VoxelChunk; 8]) -> VoxelChunk {
        let mut new_chunk = VoxelChunk::empty();

        let mut index = 1;
        for i in 0..8 {
            let mut vc : VoxelChunk = (*subvoxels[i]).clone();
            vc.shift_indexes(index);
            new_chunk.voxels.append(&mut vc.voxels);
            new_chunk.voxels[0].sub_voxels[i] = (index + 1) as i32;
            index += subvoxels[i].len();
        }

        new_chunk
    }

    /// Reduce the size of a voxel chunk by deduplicating voxels
    /// This function should run in linear time with the number DAG nodes
    pub fn compress(&mut self) {
        let n = self.len();
        // prevent reallocations
        let mut dedup = HashMap::with_capacity(n);
        let mut marked = (0..n).map(|_| false).collect::<Vec<_>>();

        // helper function to traverse the hierarchy in depth-first, post-traversal order
        // returns the index of the deduplicated voxel [1 indexed]
        fn recurse_dedup(s : &mut VoxelChunk, idx : i32, marks : &mut Vec<bool>, dedup : &mut HashMap<VChildDescriptor, i32>) -> i32 {
            for j in 0..8 {
                let sv = s.voxels[idx as usize].sub_voxels[j];

                // check if there is a subvoxel for this index
                if sv > 0 {
                    let svi = sv - 1;
                    // check if the subvoxel is a duplicate
                    if let Some(&nsvi) = dedup.get(&s.voxels[svi as usize]) {
                        // if it is, modify this voxel to point to the canonical version
                        s.voxels[idx as usize].sub_voxels[j] = nsvi;
                    } else {
                        // if the subvoxels is not (yet) a duplicate, try deduplicating it
                        let nsvi = recurse_dedup(s, svi, marks, dedup);
                        s.voxels[idx as usize].sub_voxels[j] = nsvi;
                    }
                }
            }

            let mut ret = idx + 1;
            dedup.entry(s.voxels[idx as usize])
                .and_modify(|nidx| {
                    // if this voxel is a now duplicate after deduplicating children,
                    // return the deduplicated index
                    ret = *nidx;
                })
                .or_insert_with(|| {
                    // otherwise, this is a now a unique voxel
                    marks[idx as usize] = true;
                    ret
                });

            ret
        }

        recurse_dedup(self, 0, &mut marked, &mut dedup);

        // compress the marked nodes to be contiguous, while recording their new index
        let mut new_idx = marked.iter().map(|_| -1).collect::<Vec<i32>>();
        let mut nlen = 0i32;
        for i in 0..n {
            if marked[i] {
                self.voxels[nlen as usize] = self.voxels[i];
                new_idx[i] = nlen;
                nlen += 1;
            }
        }

        // truncate the list of voxel nodes to delete the old ones
        self.voxels.truncate(nlen as usize);

        // convert all of the subvoxel indexes to the new addresses
        for i in 0..(self.voxels.len()) {
            for j in 0..8 {
                let sv = self.voxels[i].sub_voxels[j];
                if sv > 0 {
                    self.voxels[i].sub_voxels[j] = new_idx[(sv - 1) as usize] + 1;
                }
            }
        }
        
        //invalidate lod materials:
        self.lod_materials = vec![];
    }

    /// Permute the DAG nodes inplace. Note: the permutation array is overwritten.
    fn permute_node_indexes(&mut self, permutation : &mut [i32]) {
        let n = permutation.len();

        // first: update the subvoxel inputs
        for i in 0..n {
            for j in 0..8 {
                let sv = self.voxels[i].sub_voxels[j];
                if sv > 0 {
                    self.voxels[i].sub_voxels[j] = permutation[sv as usize - 1] as i32 + 1;
                }
            }
        }

        // second, permute the voxel nodes in place
        // I did some quick testing, and this seems to be faster than allocating a new array
        // and deallocating the old one
        for i in 0..n {
            let mut j = i;
            let mut temp = self.voxels[j];
            while permutation[j] < n as i32 {
                let k = permutation[j] as usize;
                let temp2 = self.voxels[k];
                self.voxels[k] = temp;
                temp = temp2;
                permutation[j] = n as i32;
                j = k;
            }
        }
    }

    /// Sort the DAG nodes in DFS order (hopefully makes the DAG more cache friendly)
    pub fn topological_sort(&mut self) {

        let mut new_indexes = self.voxels.iter()
            .map(|_| -1)
            .collect::<Vec<i32>>();

        fn recurse_traverse(voxels : &Vec<VChildDescriptor>, new_indexes : &mut Vec<i32>, i : usize, j : &mut i32) {
            for k in 0..8 {
                let sv = voxels[i].sub_voxels[k];

                if sv > 0 {
                    let svi = sv - 1;
                    if new_indexes[svi as usize] == -1 {
                        recurse_traverse(voxels, new_indexes, svi as usize, j);
                    }
                }
            }

            *j -= 1;
            new_indexes[i] = *j;
        }

        let mut j = self.len() as i32;

        recurse_traverse(&self.voxels, &mut new_indexes, 0, &mut j);

        // ensure every voxel is considered
        assert_eq!(0, j);
        // ensure the root stays the same
        assert_eq!(new_indexes[0], 0);

        self.permute_node_indexes(&mut new_indexes[..]);
    }

    pub fn duplicate_subvoxel(&mut self, i : usize, j : usize) -> Option<usize> {
        let subvoxel = self.voxels[i].sub_voxels[j];
        // check if it is a subvoxel and not a leaf
        if subvoxel > 0 {
            // append a new voxel that is a duplicate of the specified subvoxel and point the voxel to it
            self.voxels.push(self.voxels[subvoxel as usize - 1]);
            self.voxels[i].sub_voxels[j] = self.voxels.len() as i32;
            Some(self.voxels.len()) 
        } else {
            None
        }
    }

    pub fn subdivide_subvoxel(&mut self, i : usize, j : usize) -> usize {
        let subvoxel = self.voxels[i].sub_voxels[j];

        assert!(subvoxel <= 0);

        self.voxels.push(VChildDescriptor{sub_voxels : [subvoxel; 8]});
        self.voxels[i].sub_voxels[j] = self.voxels.len() as i32;
        self.voxels.len()
    }

    /// traverse the voxel data and determine the proper material to display for an LOD
    pub fn calculate_lod_materials(&mut self) {

        // Recursive helper function to calculate the most common material from each voxel's subvoxels
        fn recurse_calculate_lod_materials(s : &mut VoxelChunk, i : usize) -> u32 {
            let v = s.voxels[i];

            let mut mats : HashMap<u32, usize> = HashMap::new();

            // count the subvoxel materials
            for j in 0..8 {
                let sv = v.sub_voxels[j];
                let m = if sv > 0 {
                    let id = sv as usize -1;
                    if s.lod_materials[id] == std::u32::MAX {
                        recurse_calculate_lod_materials(s, id)
                    } else {
                        s.lod_materials[id]
                    }
                } else if sv == 0 {
                    0
                } else {
                    (-sv) as u32
                };

                if m == 0 {
                    continue;
                }

                mats.entry(m)
                    .and_modify(|x| *x += 1)
                    .or_insert(0usize);
            }

            let mut max_c = 0;
            let mut max_m = 0;
            for (&m, &c) in mats.iter() {
                if c > max_c {
                    max_c = c;
                    max_m = m;
                }
            }

            s.lod_materials[i] = max_m;
            max_m
        }

        self.lod_materials = self.voxels.iter().map(|_| std::u32::MAX).collect::<Vec<u32>>();
        recurse_calculate_lod_materials(self, 0);
    }

    /// traverse the voxel data and determine the proper material to display for an LOD
    pub fn detect_cycles(&self) -> bool {

        // Recursive helper function to calculate the most common material from each voxel's subvoxels
        fn recurse_detect_cycles(s : &VoxelChunk, i : usize, visited : &mut [bool], safe : &mut [bool], cycle : &mut Vec<usize>) -> bool {
            let v = s.voxels[i];

            if visited[i] {
                return true;
            }

            visited[i] = true;

            // count the subvoxel materials
            for j in 0..8 {
                let sv = v.sub_voxels[j];
                if sv > 0 {
                    let id = sv as usize - 1;

                    if safe[id] {
                        continue;
                    }

                    let b = recurse_detect_cycles(s, id, visited, safe, cycle);
                    if b {
                        return true;
                    }
                }
            }

            visited[i] = false;
            safe[i] = true;
            false
        }

        let mut visited = self.voxels.iter().map(|_| false).collect::<Vec<bool>>();
        let mut safe    = self.voxels.iter().map(|_| false).collect::<Vec<bool>>();
        let mut cycle = vec![];
        let has_cycle = recurse_detect_cycles(self, 0, &mut visited, &mut safe, &mut cycle);

        if has_cycle {
            println!("{} {:?}", cycle.len(), cycle);
        }

        has_cycle
    }

    /// Recursively calculate the DAG depth.
    pub fn depth(&self) -> u8 {
        fn recurse_depth(s : &VoxelChunk, i : usize, visited : &mut [bool], safe : &mut [u8]) -> u8 {
            let v = s.voxels[i];

            if visited[i] {
                return safe[i];
            }

            visited[i] = true;
            safe[i] = 255;

            let mut max_child_depth = 0u8;

            // count the subvoxel materials
            for j in 0..8 {
                let sv = v.sub_voxels[j];
                if sv > 0 {
                    let id = sv as usize - 1;

                    let d = recurse_depth(s, id, visited, safe);
                    max_child_depth = u8::max(max_child_depth, d);
                }
            }

            visited[i] = false;
            safe[i] = max_child_depth + 1;
            safe[i]
        }

        let mut visited = self.voxels.iter().map(|_| false).collect::<Vec<bool>>();
        let mut safe    = self.voxels.iter().map(|_| 0).collect::<Vec<u8>>();
        recurse_depth(self, 0, &mut visited, &mut safe)
    }
}

pub trait Integer : Into<u32> {}

impl Integer for u8 {}
impl Integer for u16 {}
impl Integer for u32 {}

// ###############################################################################################################################################
// ###############################################################################################################################################
// ###############################################################################################################################################
// Helper Functions
// ###############################################################################################################################################
// ###############################################################################################################################################
// ###############################################################################################################################################
use std::hash::Hash;

fn mode<T : Hash + Copy + Clone + Eq, I : IntoIterator<Item = T>>(numbers: I) -> T {
    let mut occurrences = HashMap::new();

    for value in numbers.into_iter() {
        *occurrences.entry(value).or_insert(0) += 1;
    }

    occurrences
        .into_iter()
        .max_by_key(|&(_, count)| count)
        .map(|(val, _)| val)
        .expect("Cannot compute the mode of zero numbers")
}

fn texture_lookup(img : &image::RgbImage, u : f32, v : f32) -> [u8; 3] {
    let (w, h) = img.dimensions();
    let image::Rgb(p) = img[((w as f32 * u) as u32 % w, (h as f32 * v) as u32 % h)];
    p
}

fn aabb_triangle_test(aabb_min : Vec3, aabb_max : Vec3, triangle : &Triangle) -> bool {

    let box_normals = [
        Vec3::new(1.0,0.0,0.0),
        Vec3::new(0.0,1.0,0.0),
        Vec3::new(0.0,0.0,1.0)
    ];

    let tri = &triangle.points;

    fn project(points : &[Vec3], axis : Vec3) -> (f32, f32) {
        let mut min = f32::MAX;
        let mut max = f32::MIN;

        for &v in points {
            let d = axis.dot(v);

            if d < min {min = d;}
            if d > max {max = d;}
        }

        (min, max)
    }

    for i in 0..3 {
        let (min, max) = project(tri, box_normals[i]);

        if max < aabb_min[i] || min > aabb_max[i] {
            return false; // No intersection possible.
        }
    }

    let box_vertices = [
        aabb_min,
        Vec3::new(aabb_max.x, aabb_min.y, aabb_min.z),
        Vec3::new(aabb_min.x, aabb_max.y, aabb_min.z),
        Vec3::new(aabb_max.x, aabb_max.y, aabb_min.z),
        Vec3::new(aabb_min.x, aabb_min.y, aabb_max.z),
        Vec3::new(aabb_max.x, aabb_min.y, aabb_max.z),
        Vec3::new(aabb_min.x, aabb_max.y, aabb_max.z),
        aabb_max
    ];

    // Test the triangle normal
    let tri_edges = [
        tri[0] - tri[1],
        tri[1] - tri[2],
        tri[2] - tri[0]
    ];

    let tri_norm = triangle.normal;

    let tri_offset = tri_norm.dot(tri[0]);
    let (min, max) = project(&box_vertices, tri_norm);

    if max < tri_offset || min > tri_offset {
        return false; // No intersection possible.
    }

    // Test the nine edge cross-products

    for i in 0..3 {
        for j in 0..3 {
            // The box normals are the same as it's edge tangents
            let axis = tri_edges[i].cross(box_normals[j]);
            let (bmin, bmax) = project(&box_vertices, axis);
            let (tmin, tmax) = project(tri, axis);
            if bmax < tmin || bmin > tmax {
                return false; // No intersection possible
            }
        }
    }

    // No separating axis found.
    return true;
}


fn cartesian_to_barycentric(tri : &Triangle, mut p : Vec3) -> Vec3 {
    // project point onto triangle:
    p *= tri.normal.dot(tri.points[0]) / tri.normal.dot(p);

    p
}