use crate::shaders::{VChildDescriptor};

use std::collections::HashMap;
use std::collections::HashSet;

use serde::{Serialize, Deserialize};

const CHUNK_DIM : usize = 64;

/// This class is mostly used for the construction of SVDAGs from dense data
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub enum Voxel {
    Empty,
    Leaf(i32),
    Branch(i32)
}

#[derive(Serialize, Deserialize)]
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
    pub fn empty() -> Self {
        // we must populate an empty root voxel;
        Self {
            voxels : vec![VChildDescriptor{
                sub_voxels : [0;8],
            }],
            lod_materials : vec![],
        }
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

    /// recursive helper function to calculate the most common material from each voxel's subvoxels
    fn recurse_calculate_lod_materials(&mut self, i : usize) -> u32 {
        let v = self.voxels[i];

        let mut mats : HashMap<u32, usize> = HashMap::new();

        for j in 0..8 {
            let sv = v.sub_voxels[j];
            let m = if sv > 0 {
                self.recurse_calculate_lod_materials(sv as usize -1)
            } else if sv == 0 {
                0
            } else {
                (-sv) as u32
            };

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

        self.lod_materials[i] = max_m;
        max_m
    }

    /// traverse the voxel data and determine the proper material to display for an LOD
    pub fn calculate_lod_materials(&mut self) {
        self.lod_materials = self.voxels.iter().map(|_| 0).collect::<Vec<u32>>();
        self.recurse_calculate_lod_materials(0);
    }
}

pub trait Integer : Into<u32> {}

impl Integer for u8 {}
impl Integer for u16 {}
impl Integer for u32 {}

/// Load a ct-scan of the stanford bunny;
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