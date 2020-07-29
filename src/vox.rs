

use cgmath::Vector3;
use cgmath::InnerSpace;

use std::collections::HashMap;
use std::collections::HashSet;

use serde::{Serialize, Deserialize};

pub const MAX_DAG_DEPTH : usize = 16;

type Vec3 = Vector3<f32>;

/// This class is mostly used for the construction of SVDAGs from dense data
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub enum Voxel {
    Empty,
    Leaf(i32),
    Branch(i32)
}

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
    pub normal : Vec3,
    pub mat    : u16,
}

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

const CHILD_IDX : [[usize;3]; 8] = [
    [0,0,0], [0,0,1],
    [0,1,0], [0,1,1],
    [1,0,0], [1,0,1],
    [1,1,0], [1,1,1]
];

fn new_vchilddescriptor() -> VChildDescriptor {
    VChildDescriptor{
        sub_voxels : [0;8],
    }
}

fn build_child_descriptor_from_children(children : [Voxel; 8], ) -> VChildDescriptor {

    let mut descriptor = new_vchilddescriptor();

    for i in 0..8 {
        match children[i] {
            Voxel::Empty => {},
            Voxel::Leaf(material) => {
                descriptor.sub_voxels[i] = -material;
            },
            Voxel::Branch(voxel) => {
                descriptor.sub_voxels[i] = voxel + 1;
            }
        }
    }

    descriptor
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

            const box_offsets : [[usize; 3]; 8] = [
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
                    min[0] + box_offsets[i][0] * half_size,
                    min[1] + box_offsets[i][1] * half_size,
                    min[2] + box_offsets[i][2] * half_size
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

            const box_offsets : [[usize; 3]; 8] = [
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
                    min[0] + box_offsets[i][0] * half_size,
                    min[1] + box_offsets[i][1] * half_size,
                    min[2] + box_offsets[i][2] * half_size
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

            const box_offsets : [[usize; 3]; 8] = [
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
                    min[0] + box_offsets[i][0] * half_size,
                    min[1] + box_offsets[i][1] * half_size,
                    min[2] + box_offsets[i][2] * half_size
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

    /// Convert an obj file into a voxel chunk format.
    /// Only places voxels intersect triangles will be made solid
    pub fn from_mesh(depth : usize, triangles: &[Triangle], corner : Vec3, size : f32) -> VoxelChunk {
        assert!(depth < MAX_DAG_DEPTH, "Depth is too large: {} >= {}", depth, MAX_DAG_DEPTH);


        fn recursive_create_shell(
            s : &mut VoxelChunk, d : usize, min : Vec3, size : f32, 
            tris : &[Triangle], indexes : &mut Vec<usize>, start : usize, 
            dedup : &mut HashMap<VChildDescriptor, i32>,
            counts : &mut HashMap<u16, i32>
        ) -> i32 {

            if d == 0 {
                // if we reach the max resolution, check if there are intersecting triangles
                if start < indexes.len() {
                    // solid material
                    
                    let m = tris[indexes[start]].mat;

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

            const box_offsets : [Vec3; 8] = [
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
                let bmin = min + box_offsets[i] * (size * 0.5);
                let bmax = bmin + Vec3::new(size * 0.5, size * 0.5, size * 0.5);

                for j in start..end {
                    if aabb_triangle_test(bmin, bmax, &tris[indexes[j]]) {
                        indexes.push(indexes[j]);
                    }
                }

                voxel.sub_voxels[i] = recursive_create_shell(s, d - 1, bmin, size * 0.5, tris, indexes, end, dedup, counts);

                indexes.truncate(end);
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

        recursive_create_shell(&mut chunk, depth, corner, size, &triangles, &mut indexes, 0, &mut dedup, &mut counts);

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

    pub fn duplicate_subvoxel(&mut self, i : usize, j : usize) {
        let subvoxel = self.voxels[i].sub_voxels[j];
        // check if it is a subvoxel and not a leaf
        if subvoxel > 0 {
            // append a new voxel that is a duplicate of the specified subvoxel and point the voxel to it
            self.voxels.push(self.voxels[subvoxel as usize - 1]);
            self.voxels[i].sub_voxels[i] = self.voxels.len() as i32;
        }
    }

    pub fn subdivide_subvoxel(&mut self, i : usize, j : usize) {
        let subvoxel = self.voxels[i].sub_voxels[j];
        self.voxels.push(VChildDescriptor{sub_voxels : [subvoxel; 8]});
        self.voxels[i].sub_voxels[i] = self.voxels.len() as i32;
    }

    /// Make an `x` by `y` by `z` grid of the current voxel chunk
    /// Makes a small number of duplicated voxels in the
    pub fn grid(&mut self, x : usize, y : usize, z : usize) {
        let max_dim = {
            use std::cmp::max;
            max(x, max(y,z))
        };

        let s = log_2(max_dim - 1) + 1;

        self.shift_indexes(s as usize);

        self.voxels.splice(
            0..0,
            (0..s).map(|i| VChildDescriptor{sub_voxels: [i as i32 + 2; 8]})
        );

        fn recursive_restrict(s : &mut VoxelChunk, i : usize, x : usize, y : usize, z : usize, scale : usize) {

            // base case, the current voxel is entirely contained in the grid
            if x >= scale && y >= scale && z >= scale {
                return;
            }

            let half_scale = scale >> 1;

            for j in 0..8 {
                let xn = if j & 0b001 == 0 { 0 } else {half_scale};
                let yn = if j & 0b010 == 0 { 0 } else {half_scale};
                let zn = if j & 0b100 == 0 { 0 } else {half_scale};

                if xn >= x || yn >= y || zn >= z {
                    // clear the subvoxel if the subvoxel is outside the grid
                    s.voxels[i].sub_voxels[j] = 0;
                } else {
                    // further process the subvoxel
                    s.duplicate_subvoxel(i, j);
                    recursive_restrict(s, s.voxels[i].sub_voxels[j] as usize - 1, x - xn, y - yn, z - zn, half_scale);
                }
            }
        }

        recursive_restrict(self, 0, x, y, z, 1 << s);
    }

    /// Translates the voxel chunk in multiples of its size in a new larger space
    pub fn translate_integral(&mut self, x : usize, y : usize, z : usize) {
        let max_dim = {
            use std::cmp::max;
            max(x, max(y,z))
        };

        let s = log_2(max_dim - 1) + 1;

        self.shift_indexes(s as usize);
        
        self.voxels.splice(
            0..0,
            (0..s).map(|i| {
                let mut sub_voxels = [0i32; 8];
                let j = s - i - 1;

                // calculate the index of the next child at depth i
                let xo = if x & (1 << j) == 0 {0} else {1};
                let yo = if y & (1 << j) == 0 {0} else {2};
                let zo = if z & (1 << j) == 0 {0} else {4};

                sub_voxels[xo + yo + zo] = i as i32 + 2;
                
                VChildDescriptor{sub_voxels}
            })
        );
    }

    /// Translates the voxel chunk by fractions of its size
    pub fn translate_fractional(&mut self, x : usize, y : usize, z : usize) {
        unimplemented!()
    }

    /// Writes the other voxel chunk into this one. Whether the other voxels overwrite or not is
    /// controlled by the `overwrite` parameter
    pub fn combine(&mut self, other : &VoxelChunk, overwrite : bool) {
        let n = self.len();

        {
            let mut other_clone : VoxelChunk = other.clone();
            other_clone.shift_indexes(n);
            self.voxels.extend(other_clone.voxels);
        }

        fn recursive_combine(s : &mut VoxelChunk, overwrite : bool, i : usize, j : usize) {
            for k in 0..8 {
                let sv0 = s.voxels[i].sub_voxels[k];
                let sv1 = s.voxels[j].sub_voxels[k];

                if sv0 == 0  {
                    s.voxels[i].sub_voxels[k] = sv1;
                }
                if sv0 > 0 && sv1 > 0 {
                    recursive_combine(s, overwrite, sv0 as usize - 1, sv1 as usize - 1);
                }
                if overwrite {
                    if sv1 < 0 {
                        s.voxels[i].sub_voxels[k] = sv1;
                    }
                }
            }
        }

        recursive_combine(self, overwrite, 0, n);

        self.compress();
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

use std::path::Path;

use image;
fn read_tga<P : AsRef<Path>>(path : P) -> image::DynamicImage {
    let bytes = std::fs::read(path).unwrap();

    
    let byte_stream = std::io::Cursor::new(&bytes);

    let mut reader = image::io::Reader::new(byte_stream);

    reader.set_format(image::ImageFormat::Tga);

    let image = reader.decode().unwrap();

    image    
}


// ###############################################################################################################################################
// ###############################################################################################################################################
// ###############################################################################################################################################
// Tests
// ###############################################################################################################################################
// ###############################################################################################################################################
// ###############################################################################################################################################


#[test]
fn test_voxel_dag_obj_shell_teapot() {
    use obj::*;
    use std::path::Path;
    use std::fs;

    let obj_data = Obj::load(&Path::new("./data/obj/teapot.obj")).expect("Failed to load obj file");


    let mut triangles = vec![];

    for o in 0..(obj_data.data.objects.len()) {
        let object = &obj_data.data.objects[o];
        for g in 0..(object.groups.len()) {
            let group = &object.groups[g];
            for p in 0..(group.polys.len()) {
                let poly = &group.polys[p];
                for v in 2..(poly.0.len()) {
                    let v0 = obj_data.data.position[poly.0[0].0];
                    let v1 = obj_data.data.position[poly.0[v-1].0];
                    let v2 = obj_data.data.position[poly.0[v].0];


                    let v0 = Vec3::new(v0[0], v0[1], v0[2]);
                    let v1 = Vec3::new(v1[0], v1[1], v1[2]);
                    let v2 = Vec3::new(v2[0], v2[1], v2[2]);
                    
                    triangles.push(Triangle{
                        points : [v0, v1, v2],
                        normal : (v0 - v1).cross(v1 - v2),
                        mat    : 1,
                    });
                }
            }
        }
    }

    let mut min = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
    let mut max = Vec3::new(f32::MIN, f32::MIN, f32::MIN);

    for [x,y,z] in obj_data.data.position {
        if x < min.x {min.x = x;}
        if y < min.y {min.y = y;}
        if z < min.z {min.z = z;}
        if x > max.x {max.x = x;}
        if y > max.y {max.y = y;}
        if z > max.z {max.z = z;}
    }

    let size = max - min;
    let mut max_size = size.x;
    if size.y > max_size {max_size = size.y;}
    if size.z > max_size {max_size = size.z;}

    println!("Triangles: {}", triangles.len());

    use std::time::*;

    let start = Instant::now();
    let mut vchunk = VoxelChunk::from_mesh(9, &triangles, min, max_size);
    let elapsed = start.elapsed();

    println!("Time to voxelize: {:?}", elapsed);
    println!("DAG nodes: {}", vchunk.len());

    let serialized = bincode::serialize(&vchunk).unwrap();

    fs::write("./data/dag/teapot.svdag", serialized).unwrap();

}

#[test]
fn test_voxel_dag_obj_shell_sponza() {
    use obj::Obj;
    use std::path::Path;
    use std::fs;

    let mut obj_data = Obj::load(&Path::new("./data/obj/Sponza/sponza.obj")).expect("Failed to load obj file");


    let mut triangles = vec![];

    let mut materials = HashMap::new();

    const DEPTH : usize = 12;

    use std::path::PathBuf;
    let obj_root = PathBuf::from("./data/obj/Sponza/");
    let mut material_list = vec![Material::default(); materials.len()];

    for mtl in obj_data.data.material_libs.iter_mut() {
        use std::io::Read;
        use std::fs::File;

        mtl.reload(File::open(&obj_root.join(mtl.filename.clone())).unwrap()).unwrap();

        for mat in &mtl.materials {
            let kd = if let Some(kd_tex_file) = &mat.map_kd {
                
                println!("Loading texture: {:?}", kd_tex_file);

                let img = read_tga(obj_root.join(kd_tex_file));

                println!("  img: {:?}", img.color());

                let img = img.into_rgb();

                let (w, h) = img.dimensions();

                println!("  Averaging...");

                let mut color = [0.0; 3];

                for (_,_,&image::Rgb(p)) in img.enumerate_pixels() {
                    color[0] += p[0] as f32;
                    color[1] += p[1] as f32;
                    color[2] += p[2] as f32;
                }

                color[0] /= (w * h * 255) as f32;
                color[1] /= (w * h * 255) as f32;
                color[2] /= (w * h * 255) as f32;

                println!("  Color: {:?}", color);

                color
            } else {
                [1.0; 3]
            };

            let next =  materials.len();
            let id   = *materials.entry(mat.name.clone()).or_insert(next);

            let mut kdd = mat.kd.unwrap_or([0.0; 3]);

            kdd[0] *= kd[0];
            kdd[1] *= kd[1];
            kdd[2] *= kd[2];
        
            material_list[id] = Material {
                albedo : kdd,
                metalness : mat.km.unwrap_or(0.0),
                emission : mat.ke.unwrap_or([0.0; 3]),
                roughness : 0.3,
            };
        }
    }

    println!("Material Count: {}", materials.len());

    println!("Processing Triangles...");

    for o in 0..(obj_data.data.objects.len()) {
        let object = &obj_data.data.objects[o];
        for g in 0..(object.groups.len()) {
            let group = &object.groups[g];

            let next = materials.len();

            let id = if let Some(obj::ObjMaterial::Ref(s)) = &group.material {
                *materials.entry(s.clone()).or_insert(next) as u16
            } else {
                0
            };

            for p in 0..(group.polys.len()) {
                let poly = &group.polys[p];
                for v in 2..(poly.0.len()) {
                    let v0 = obj_data.data.position[poly.0[0].0];
                    let v1 = obj_data.data.position[poly.0[v-1].0];
                    let v2 = obj_data.data.position[poly.0[v].0];


                    let v0 = Vec3::new(v0[0], v0[1], v0[2]);
                    let v1 = Vec3::new(v1[0], v1[1], v1[2]);
                    let v2 = Vec3::new(v2[0], v2[1], v2[2]);
                    
                    triangles.push(Triangle{
                        points : [v0, v1, v2],
                        normal : (v0 - v1).cross(v1 - v2),
                        // mat    : 0,
                        mat    : id,
                    });
                }
            }
        }
    }

    println!("Triangles: {}", triangles.len());


    let mut min = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
    let mut max = Vec3::new(f32::MIN, f32::MIN, f32::MIN);

    for [x,y,z] in obj_data.data.position {
        if x < min.x {min.x = x;}
        if y < min.y {min.y = y;}
        if z < min.z {min.z = z;}
        if x > max.x {max.x = x;}
        if y > max.y {max.y = y;}
        if z > max.z {max.z = z;}
    }

    let size = max - min;
    let mut max_size = size.x;
    if size.y > max_size {max_size = size.y;}
    if size.z > max_size {max_size = size.z;}

    
    println!("Constructing SVDAG...");
    use std::time::*;
    
    let start = Instant::now();
    let vchunk = VoxelChunk::from_mesh(DEPTH, &triangles, min, max_size);
    let elapsed = start.elapsed();
    
    println!("Time to voxelize: {:?}", elapsed);
    println!("DAG nodes: {}", vchunk.len());
    
    let serialized = bincode::serialize(&vchunk).unwrap();
    let serialized_mats = bincode::serialize(&material_list).unwrap();

    fs::write(format!("./data/dag/sponza_mats.svdag"), serialized).unwrap();
    fs::write("./data/dag/sponza_mats.mats", serialized_mats).unwrap();
}


#[test]
fn test_voxel_dag_obj_shell_hairball() {
    use obj::Obj;
    use std::path::Path;
    use std::fs;

    let obj_data = Obj::load(&Path::new("./data/obj/hairball.obj")).expect("Failed to load obj file");

    let mut triangles = vec![];

    // let mut materials = HashMap::new();

    const DEPTH : usize = 10;

    println!("Processing Triangles...");

    for o in 0..(obj_data.data.objects.len()) {
        let object = &obj_data.data.objects[o];
        for g in 0..(object.groups.len()) {
            let group = &object.groups[g];

            // let next = materials.len();

            // let id = if let Some(obj::ObjMaterial::Ref(s)) = &group.material {
            //     *materials.entry(s.clone()).or_insert(next) as u16
            // } else {
            //     0
            // };

            for p in 0..(group.polys.len()) {
                let poly = &group.polys[p];
                for v in 2..(poly.0.len()) {
                    let v0 = obj_data.data.position[poly.0[0].0];
                    let v1 = obj_data.data.position[poly.0[v-1].0];
                    let v2 = obj_data.data.position[poly.0[v].0];


                    let v0 = Vec3::new(v0[0], v0[1], v0[2]);
                    let v1 = Vec3::new(v1[0], v1[1], v1[2]);
                    let v2 = Vec3::new(v2[0], v2[1], v2[2]);
                    
                    triangles.push(Triangle{
                        points : [v0, v1, v2],
                        normal : (v0 - v1).cross(v1 - v2),
                        mat    : 0,
                        // mat    : id,
                    });
                }
            }
        }
    }

    println!("Triangles: {}", triangles.len());


    let mut min = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
    let mut max = Vec3::new(f32::MIN, f32::MIN, f32::MIN);

    for [x,y,z] in obj_data.data.position {
        if x < min.x {min.x = x;}
        if y < min.y {min.y = y;}
        if z < min.z {min.z = z;}
        if x > max.x {max.x = x;}
        if y > max.y {max.y = y;}
        if z > max.z {max.z = z;}
    }

    let size = max - min;
    let mut max_size = size.x;
    if size.y > max_size {max_size = size.y;}
    if size.z > max_size {max_size = size.z;}

    
    println!("Constructing SVDAG...");
    use std::time::*;
    
    let start = Instant::now();
    let vchunk = VoxelChunk::from_mesh(DEPTH, &triangles, min, max_size);
    let elapsed = start.elapsed();
    
    println!("Time to voxelize: {:?}", elapsed);
    println!("DAG nodes: {}", vchunk.len());
    
    let serialized = bincode::serialize(&vchunk).unwrap();

    fs::write(format!("./data/dag/hairball.svdag"), serialized).unwrap();
}


#[test]
fn test_voxel_dag_tri_shell() {
    use std::path::Path;
    use std::fs;
    
    let v0 = Vec3::new(1.0, 0.0, 0.0);
    let v1 = Vec3::new(0.0, 1.0, 0.0);
    let v2 = Vec3::new(0.0, 0.0, 1.0);

    let mut triangles = vec![
        Triangle{
            points : [v0, v1, v2],
            normal : (v0 - v1).cross(v1 - v2),
            mat    : 1,
        }
    ];

    let min = Vec3::new(0.0, 0.0, 0.0);
    let size = 1.0;

    println!("Triangles: {}", triangles.len());

    use std::time::*;

    let start = Instant::now();
    let vchunk = VoxelChunk::from_mesh(8, &triangles, min, size);
    let elapsed = start.elapsed();

    println!("DAG nodes: {}", vchunk.len());

    println!("Time to assemble: {:?}", elapsed);

    let serialized = bincode::serialize(&vchunk).unwrap();

    fs::write("./data/dag/tri.svdag", serialized).unwrap();

}

/// Construct an SVDAG of a ct-scan of the stanford bunny;
#[test]
fn test_voxel_dag_bunny() {
    let mut data : Vec<u16> = Vec::with_capacity(512 *  512 * 361);

    use std::path::Path;
    use std::fs;
    use std::u16;
    use std::time::*;

    let dir = Path::new("./data/dense/bunny/");
    if dir.is_dir() {
        for entry in fs::read_dir(dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();

            if !path.is_dir() {
                println!("Loading {:?}", path);
                let slice = fs::read(path).unwrap();

                assert_eq!(slice.len(), 512*512*2);

                data.append(&mut slice.chunks_exact(2).map(|v| u16::from_be_bytes([v[0],v[1]])).collect::<Vec<_>>());
            }
        }
    }

    assert_eq!(data.len(), 512 * 512 * 361);

    println!("Converting...");

    // scan data has a solid cylinder around the bunny, so this code removes that.
    for z in 0..361 {
        for y in 0..512 {
            for x in 0..512 {
                let dx = x as i32 - 256;
                let dy = y as i32 - 256;
                let i  = x + 512 * (y + 512 * z);

                if dx * dx + dy * dy > 255 * 255 {
                    data[i] = 0;
                }
            }
        }
    }

    // threshold for the ct scan data
    let data = data.iter().map(|&v| if v > 0x06ff {1i32} else {0i32}).collect::<Vec<i32>>();

    let start = Instant::now();

    let mut chunk = VoxelChunk::from_dense_voxels(&data, [512, 512, 361]);

    let runtime = start.elapsed();

    
    println!("Compression took {:?}", runtime);
    chunk.topological_sort();

    let out_path = Path::new("./data/dag/bunny.svdag");
        
    println!("Writing File... ({:?})", out_path);

    use bincode;

    let serialized = bincode::serialize(&chunk).unwrap();

    fs::write(out_path, serialized).unwrap();

    println!("Num Voxels: {} (from {})", chunk.voxels.len(), 512*512*361);
}

#[test]
fn test_voxel_dag_implicit() {
    use std::time::*;
    use std::fs;
    use bincode;
    use std::path::*;

    println!("Compressing implicit gyroid...");

    let start = Instant::now();

    let chunk = VoxelChunk::from_dense_implicit(8, |x, y, z| {
        let scale = 1.0/16.0;
        let threshold = 0.05;

        let x = x as f32 * scale;
        let y = y as f32 * scale;
        let z = z as f32 * scale;

        let sdf = x.sin() * y.cos() + y.sin() * z.cos() + z.sin() * x.cos();

        if sdf.abs() < threshold {
            1
        } else {
            0
        }
    });

    let runtime = start.elapsed();
    
    println!("Compression took {:?}", runtime);

    let out_path = Path::new("./data/dag/gyroid.svdag");
        
    println!("Writing File... ({:?})", out_path);

    let serialized = bincode::serialize(&chunk).unwrap();

    fs::write(out_path, serialized).unwrap();

    println!("Num Voxels: {} (from {})", chunk.voxels.len(), 512*512*512);
}


#[test]
fn test_voxel_dag_de_gyroid() {
    use std::time::*;
    use std::fs;
    use bincode;
    use std::path::*;

    println!("Compressing DE gyroid...");

    let start = Instant::now();

    let chunk = VoxelChunk::from_distance_equation(10, |x, y, z| {
        let scale = std::f32::consts::PI * 4.0;

        let x = x as f32 * scale;
        let y = y as f32 * scale;
        let z = z as f32 * scale;

        let sdf = x.sin() * y.cos() + y.sin() * z.cos() + z.sin() * x.cos();

        sdf.abs() / scale
    });

    let runtime = start.elapsed();

    println!("Compression took {:?}", runtime);
    
    println!("Num Voxels: {} (uncompress: {})", chunk.voxels.len(), 512*512*512);

    assert!(!chunk.detect_cycles(), "Cycle Detected!");

    let out_path = Path::new("./data/dag/gyroid_de.svdag");

    println!("Writing File... ({:?})", out_path);

    let serialized = bincode::serialize(&chunk).unwrap();

    fs::write(out_path, serialized).unwrap();
}

#[test]
fn test_voxel_dag_de_mandelbulb() {
    use std::time::*;
    use std::fs;
    use bincode;
    use std::path::*;

    println!("Compressing DE mandelbulb...");

    let start = Instant::now();
    const DEPTH : usize = 11;
    const DIM : usize = 1 << DEPTH;

    let chunk = VoxelChunk::from_distance_equation(DEPTH, |x, y, z| {
        const SCALE : f32 = 4.0;

        let pos = Vec3::new(x - 0.5, y - 0.5, z - 0.5) * SCALE;
        let mut z = pos;
        let mut dr = 1.0;
        let mut r = 0.0;

        const ITERS : usize = 8;
        const POWER : f32 = 8.0;
        const BAILOUT : f32 = 2.0;

        for _ in 0..ITERS {
            r = z.magnitude(); //length(z);
            if r>BAILOUT {
                break
            };
            
            // convert to polar coordinates
            let mut theta = (z.z/r).acos();
            let mut phi = (z.y).atan2(z.x);
            dr =  r.powf( POWER -1.0) * POWER * dr + 1.0;
            
            // scale and rotate the point
            let zr = r.powf(POWER);
            theta = theta*POWER;
            phi = phi*POWER;
            
            // convert back to cartesian coordinates
            z = zr*Vec3::new(theta.sin()*phi.cos(), phi.sin()*theta.sin(), theta.cos());
            z += pos;
        }
        return 0.5*r.ln()*r/dr / SCALE;
    });

    let runtime = start.elapsed();

    println!("Compression took {:?}", runtime);
    
    println!("Num Voxels: {} (uncompressed: {} ({:2.1}%))", chunk.voxels.len(), DIM*DIM*DIM,chunk.voxels.len() as f32 / (DIM*DIM*DIM) as f32);

    assert!(!chunk.detect_cycles(), "Cycle Detected!");

    let out_path = Path::new("./data/dag/mandelbulb.svdag");

    println!("Writing File... ({:?})", out_path);

    let serialized = bincode::serialize(&chunk).unwrap();

    fs::write(out_path, serialized).unwrap();
}


/// This test constructs a simple sphere as a test
#[test]
fn test_voxel_dag_sphere() {

    use std::path::Path;
    use std::fs;

    let data : [i32; 8*8*8]= [
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,

        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0,
        0, 0, 1, 1, 1, 1, 0, 0,
        0, 0, 1, 1, 1, 1, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,

        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0,
        0, 0, 1, 1, 1, 1, 0, 0,
        0, 1, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 1, 0,
        0, 0, 1, 1, 1, 1, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,

        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 1, 1, 1, 0, 0,
        0, 1, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 1, 0,
        0, 0, 1, 1, 1, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,

        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 1, 1, 1, 0, 0,
        0, 1, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 1, 0,
        0, 0, 1, 1, 1, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,

        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0,
        0, 0, 1, 1, 1, 1, 0, 0,
        0, 1, 1, 1, 1, 1, 1, 0,
        0, 1, 1, 1, 1, 1, 1, 0,
        0, 0, 1, 1, 1, 1, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,

        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0,
        0, 0, 1, 1, 1, 1, 0, 0,
        0, 0, 1, 1, 1, 1, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,

        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0
    ];

    let chunk = VoxelChunk::from_dense_voxels(&data, [8, 8, 8]);

    let out_path = Path::new("./data/dag/sphere.svdag");

    {
        use bincode;
        use serde_json::to_string_pretty;

        let serialized = bincode::serialize(&chunk).unwrap();

        fs::write(out_path, serialized).unwrap();

        println!("{}", to_string_pretty(&chunk).unwrap());
    }

    println!("Num Voxels: {} (from {})", chunk.voxels.len(), 8*8*8);
}

/// This test constructs a simple sphere as a test
#[test]
fn test_voxel_dag_checkers() {

    use std::path::Path;
    use std::fs;
    for i in 1..6 {
        let d = 1 << i;
        let data : Vec<i32> = (0..d).map(|z| {
            (0..d).map(move |y| {
                (0..d).map(move |x| (x + y + z) % 2)
            }).flatten()
        }).flatten().collect();

        let chunk = VoxelChunk::from_dense_voxels(&data, [d as usize, d as usize, d as usize]);

        let path = format!("./data/dag/checker{:0>2}.svdag", d);

        let out_path = Path::new(&path);

        {
            use bincode;
            use serde_json::to_string_pretty;

            let serialized = bincode::serialize(&chunk).unwrap();

            fs::write(out_path, serialized).unwrap();

            println!("Checker (d={})", d);
            println!("{}", to_string_pretty(&chunk).unwrap());
            println!("Num Voxels: {} (from {})", chunk.voxels.len(), d*d*d);
        }

    }
}

#[test]
fn test_voxel_dag_octohedron() {

    use std::path::Path;
    use std::fs;
    for i in 1..6 {
        let d = 1 << i;
        let dd = d >> 1;
        let data : Vec<i32> = (0..d).map(|z| {
            (0..d).map(move |y| {
                (0..d).map(move |x| if (x == dd || x == dd - 1) && (y == dd || y == dd - 1) && (z == dd || z == dd - 1) {1} else {0})
            }).flatten()
        }).flatten().collect();

        let chunk = VoxelChunk::from_dense_voxels(&data, [d as usize, d as usize, d as usize]);

        let path = format!("./data/dag/octohedron{:0>2}.svdag", d);

        let out_path = Path::new(&path);

        {
            use bincode;
            use serde_json::to_string_pretty;

            let serialized = bincode::serialize(&chunk).unwrap();

            fs::write(out_path, serialized).unwrap();

            println!("Checker (d={})", d);
            println!("{}", to_string_pretty(&chunk).unwrap());
            println!("Num Voxels: {} (from {})", chunk.voxels.len(), d*d*d);
        }

    }
}