

use cgmath::Vector3;
use cgmath::InnerSpace;

use std::collections::HashMap;
use std::collections::HashSet;

use serde::{Serialize, Deserialize};

const CHUNK_DIM : usize = 64;

type Vec3 = Vector3<f32>;

/// This class is mostly used for the construction of SVDAGs from dense data
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub enum Voxel {
    Empty,
    Leaf(i32),
    Branch(i32)
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
    pub fn from_dense_data_fn<S, F>(data : &[S], dim : [usize; 3], f_mat : F) -> Self 
    where S : Integer + Copy, F : Fn(S) -> Voxel {
        
        let depth = log_2(dim.iter().cloned().max().unwrap_or(0));

        let msize = 1 << depth;

        // hashmap is used to deduplicate identical voxel sub-DAGs
        let mut dedup_vox = HashMap::new();

        // cache stores the previous
        let mut cache = data.iter().map(|&x| f_mat(x)).collect::<Vec<Voxel>>();
        let unique_materials = cache.iter().fold(HashSet::new(), |mut s, &x| {s.insert(x); s});

        // insert deduplication entries for leaf voxels
        // dedup_vox.insert([Voxel::Empty; 8], Voxel::Empty);
        for mat in unique_materials {
            dedup_vox.insert([mat; 8], mat);
        }
    

        // the starting ID for voxels
        let mut id = 0i32;

        let mut voxels = vec![];
        let mut edges = vec![];
        let mut no_edges = vec![];
        let mut parents : Vec<Vec<i32>> = vec![];

        // first step: deduplicate voxels. 
        for lsize in 1..=depth {
            let s = 1 << lsize;
            let mut level_cache = vec![];

            let ldim = [(dim[0] - 1) * 2 / s + 1, (dim[1] - 1) * 2 / s + 1, (dim[2] - 1) * 2 / s + 1];

            for z in 0..(msize / s) {
                for y in 0..(msize / s) {
                    for x in 0..(msize / s) {

                        let x = x + x;
                        let y = y + y;
                        let z = z + z;

                        let children = [
                            array3d_get(&cache, ldim, [x,   y,   z  ]).unwrap_or(Voxel::Empty),
                            array3d_get(&cache, ldim, [x+1, y,   z  ]).unwrap_or(Voxel::Empty),
                            array3d_get(&cache, ldim, [x,   y+1, z  ]).unwrap_or(Voxel::Empty),
                            array3d_get(&cache, ldim, [x+1, y+1, z  ]).unwrap_or(Voxel::Empty),
                            array3d_get(&cache, ldim, [x,   y,   z+1]).unwrap_or(Voxel::Empty),
                            array3d_get(&cache, ldim, [x+1, y,   z+1]).unwrap_or(Voxel::Empty),
                            array3d_get(&cache, ldim, [x,   y+1, z+1]).unwrap_or(Voxel::Empty),
                            array3d_get(&cache, ldim, [x+1, y+1, z+1]).unwrap_or(Voxel::Empty)
                        ];

                        dedup_vox.entry(children)
                            .and_modify(|id| level_cache.push(*id))
                            .or_insert_with(|| {
                                let mut ve = 0;
                                for c in children.iter() {
                                    match *c {
                                        Voxel::Empty | Voxel::Leaf(_) => (),
                                        Voxel::Branch(cid) => {
                                            ve += 1;
                                            parents[cid as usize].push(id);
                                        },
                                    }
                                }
                                voxels.push(build_child_descriptor_from_children(children));
                                edges.push(ve);
                                parents.push(vec![]);
                                if ve == 0 {
                                    no_edges.push(id);
                                }

                                let new_id = Voxel::Branch(id);
                                id += 1;
                                level_cache.push(new_id);

                                new_id
                            });
                    }
                }
            }

            cache = level_cache;
        }


        let n = voxels.len();

        let mut topo_perm = vec![];
        let mut topo_perm_inv = (0..n).map(|_| 0i32).collect::<Vec<_>>();

        let mut j = n as i32 - 1;

        // second step: voxel topological sorting to get final layout (particularly with root voxel at 0).
        while let Some(i) = no_edges.pop() {
            topo_perm.push(i);
            topo_perm_inv[i as usize] = j;
            j -= 1;
            for &p in &parents[i as usize] {
                // check valid = 1 and leaf = 0
                edges[p as usize] -= 1;
                if edges[p as usize] == 0 {
                    no_edges.push(p);
                }
            }
        }

        // third step: permute all the voxels and their child indices based on their topological sort
        let old_voxels = voxels;
        voxels = Vec::with_capacity(n);

        for &i in topo_perm.iter().rev() {
            let mut v = old_voxels[i as usize];
            for i in 0..8 {
                // check that it is a subvoxel
                if v.sub_voxels[i] > 0 {
                    // permute the subvoxel index
                    v.sub_voxels[i] = topo_perm_inv[v.sub_voxels[i] as usize - 1] + 1;
                }
            }
            voxels.push(v);
        }

        Self {voxels,lod_materials : vec![]}
    }

    /// Convert an obj file into a voxel chunk format.
    /// Only places voxels intersect triangles will be made solid
    pub fn from_obj_shell(depth : usize, triangles: &[[Vec3;3]], corner : Vec3, size : f32) -> VoxelChunk {
        assert!(depth < 16, "Depth is too large: {} >= 16", depth);


        fn recursive_create_shell(
            s : &mut VoxelChunk, d : usize, min : Vec3, size : f32, 
            tris : &[[Vec3; 3]], indexes : &mut Vec<usize>, start : usize, 
            dedup : &mut HashMap<VChildDescriptor, i32>
        ) -> i32 {

            if d == 0 {
                // if we reach the max resolution, check if there are intersecting triangles
                if start < indexes.len() {
                    // solid material
                    return -1;
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
                    if aabb_triangle_test(bmin, bmax, tris[indexes[j]]) {
                        indexes.push(indexes[j]);
                    }
                }

                voxel.sub_voxels[i] = recursive_create_shell(s, d - 1, bmin, size * 0.5, tris, indexes, end, dedup);

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
        let mut dedup : HashMap<VChildDescriptor, i32> = HashMap::new();
        let mut indexes = (0..(triangles.len())).collect::<Vec<_>>();

        recursive_create_shell(&mut chunk, depth, corner, size, &triangles, &mut indexes, 0, &mut dedup);

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

            let mut ret = idx;
            dedup.entry(s.voxels[idx as usize])
                .and_modify(|nidx| {
                    // if this voxel is a now duplicate after deduplicating children,
                    // return the deduplicated index
                    ret = *nidx;
                })
                .or_insert_with(|| {
                    // otherwise, this is a now a unique voxel
                    marks[idx as usize] = true;
                    idx
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

fn aabb_triangle_test(aabb_min : Vec3, aabb_max : Vec3, tri : [Vec3; 3]) -> bool {

    let box_normals = [
        Vec3::new(1.0,0.0,0.0),
        Vec3::new(0.0,1.0,0.0),
        Vec3::new(0.0,0.0,1.0)
    ];


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
        let (min, max) = project(&tri, box_normals[i]);

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

    let tri_norm = tri_edges[0].cross(tri_edges[1]).normalize();

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
            let (tmin, tmax) = project(&tri, axis);
            if bmax < tmin || bmin > tmax {
                return false; // No intersection possible
            }
        }
    }

    // No separating axis found.
    return true;
}


        /*
        
bool IsIntersecting(IAABox box, ITriangle triangle)
{
    double triangleMin, triangleMax;
    double boxMin, boxMax;

    // Test the box normals (x-, y- and z-axes)
    var boxNormals = new IVector[] {
        new Vector(1,0,0),
        new Vector(0,1,0),
        new Vector(0,0,1)
    };
    for (int i = 0; i < 3; i++)
    {
        IVector n = boxNormals[i];
        Project(triangle.Vertices, boxNormals[i], out triangleMin, out triangleMax);
        if (triangleMax < box.Start.Coords[i] || triangleMin > box.End.Coords[i])
            return false; // No intersection possible.
    }

    // Test the triangle normal
    double triangleOffset = triangle.Normal.Dot(triangle.A);
    Project(box.Vertices, triangle.Normal, out boxMin, out boxMax);
    if (boxMax < triangleOffset || boxMin > triangleOffset)
        return false; // No intersection possible.

    // Test the nine edge cross-products
    IVector[] triangleEdges = new IVector[] {
        triangle.A.Minus(triangle.B),
        triangle.B.Minus(triangle.C),
        triangle.C.Minus(triangle.A)
    };
    for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
    {
        // The box normals are the same as it's edge tangents
        IVector axis = triangleEdges[i].Cross(boxNormals[j]);
        Project(box.Vertices, axis, out boxMin, out boxMax);
        Project(triangle.Vertices, axis, out triangleMin, out triangleMax);
        if (boxMax <= triangleMin || boxMin >= triangleMax)
            return false; // No intersection possible
    }

    // No separating axis found.
    return true;
}

         */


// ###############################################################################################################################################
// ###############################################################################################################################################
// ###############################################################################################################################################
// Tests
// ###############################################################################################################################################
// ###############################################################################################################################################
// ###############################################################################################################################################


#[test]
fn voxel_chunk_construct_obj_shell() {
    use obj::*;
    use std::path::Path;
    use std::fs;

    let obj_data = Obj::load(&Path::new("./data/teapot.obj")).expect("Failed to load obj file");


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

                    triangles.push([
                        Vec3::new(v0[0], v0[1], v0[2]),
                        Vec3::new(v1[0], v1[1], v1[2]),
                        Vec3::new(v2[0], v2[1], v2[2])
                    ]);
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
    let vchunk = VoxelChunk::from_obj_shell(9, &triangles, min, max_size);
    let elapsed = start.elapsed();

    println!("DAG nodes: {}", vchunk.len());

    println!("Time to assemble: {:?}", elapsed);

    let serialized = bincode::serialize(&vchunk).unwrap();

    fs::write("./data/teapot.svdag", serialized).unwrap();

}

#[test]
fn test_aabb_tri_intersection() {
    {
        let min = Vec3::new(-10.0, -10.0, -10.0);
        let max = Vec3::new(10.0, 10.0, 10.0);

        let tri = [
            Vec3::new(12.0,9.0,9.0),
            Vec3::new(9.0,12.0,9.0),
            Vec3::new(19.0,19.0,20.0)
        ];

        assert!(!aabb_triangle_test(min, max, tri));
    }
    
    {
        let min = Vec3::new(0.0, 0.0, 0.0);
        let max = Vec3::new(0.25, 0.25, 0.25);

        let tri = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0)
        ];

        assert!(!aabb_triangle_test(min, max, tri));
    }
}

#[test]
fn voxel_chunk_construct_tri_shell() {
    use std::path::Path;
    use std::fs;


    let mut triangles = vec![[
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0)
    ]];

    let min = Vec3::new(0.0, 0.0, 0.0);
    let size = 1.0;

    println!("Triangles: {}", triangles.len());

    use std::time::*;

    let start = Instant::now();
    let vchunk = VoxelChunk::from_obj_shell(8, &triangles, min, size);
    let elapsed = start.elapsed();

    println!("DAG nodes: {}", vchunk.len());

    println!("Time to assemble: {:?}", elapsed);

    let serialized = bincode::serialize(&vchunk).unwrap();

    fs::write("./data/tri.svdag", serialized).unwrap();

}

/// Construct an SVDAG of a ct-scan of the stanford bunny;
#[test]
fn voxel_chunk_compress_bunny() {
    let mut data : Vec<u16> = Vec::with_capacity(512 *  512 * 361);

    use std::path::Path;
    use std::fs;
    use std::u16;

    let dir = Path::new("./data/bunny/");
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

    let chunk = VoxelChunk::from_dense_data_fn(&data, [512, 512, 361], |v| if v > 0x06ff {Voxel::Leaf(1)} else {Voxel::Empty});

    let out_path = Path::new("./data/bunny.svdag");
        
    println!("Writing File... ({:?})", out_path);

    use bincode;

    let serialized = bincode::serialize(&chunk).unwrap();

    fs::write(out_path, serialized).unwrap();

    println!("Num Voxels: {} (from {})", chunk.voxels.len(), 512*512*361);
}


/// This test constructs a simple sphere as a test
#[test]
fn voxel_chunk_compress_sphere() {

    use std::path::Path;
    use std::fs;

    let data : [u8; 8*8*8]= [
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

    let chunk = VoxelChunk::from_dense_data_fn(&data, [8, 8, 8], |v| if v == 1 {Voxel::Leaf(1)} else {Voxel::Empty});

    let out_path = Path::new("./data/sphere.svdag");

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
fn voxel_chunk_compress_checkers() {

    use std::path::Path;
    use std::fs;
    for i in 1..6 {
        let d = 1 << i;
        let data : Vec<u16> = (0..d).map(|z| {
            (0..d).map(move |y| {
                (0..d).map(move |x| (x + y + z) % 2)
            }).flatten()
        }).flatten().collect();

        let chunk = VoxelChunk::from_dense_data_fn(&data, [d as usize, d as usize, d as usize], |v| if v == 1 {Voxel::Leaf(1)} else {Voxel::Empty});

        let path = format!("./data/checker{:0>2}.svdag", d);

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

/// This test constructs a simple sphere as a test
#[test]
fn voxel_chunk_compress_inner_cube() {

    use std::path::Path;
    use std::fs;
    for i in 1..6 {
        let d = 1 << i;
        let dd = d >> 1;
        let data : Vec<u16> = (0..d).map(|z| {
            (0..d).map(move |y| {
                (0..d).map(move |x| if (x == dd || x == dd - 1) && (y == dd || y == dd - 1) && (z == dd || z == dd - 1) {1} else {0})
            }).flatten()
        }).flatten().collect();

        let chunk = VoxelChunk::from_dense_data_fn(&data, [d as usize, d as usize, d as usize], |v| if v == 1 {Voxel::Leaf(1)} else {Voxel::Empty});

        let path = format!("./data/inner_cube{:0>2}.svdag", d);

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