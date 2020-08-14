use super::*;

use super::voxelize::*;

use pbr::ProgressBar;

use noise;
use noise::NoiseFn;

use std::path::PathBuf;

#[test]
fn test_voxel_dag_obj_shell_buff_doge() {
    println!("Converting {:?}", "./data/obj/BuffDoge.OBJ");

    convert_obj_file(
        PathBuf::from("./data/obj/BuffDoge.OBJ"),
        PathBuf::from("./data/dag/BuffDoge.svdag"),
        12
    );

    println!("");
    println!("Converting {:?}", "./data/obj/BuffDoge.OBJ");

    convert_obj_file(
        PathBuf::from("./data/obj/MegaBuffDoge.OBJ"),
        PathBuf::from("./data/dag/MegaBuffDoge.svdag"),
        12
    );

    println!("");
    println!("Converting {:?}", "./data/obj/BuffDoge.OBJ");
    
    convert_obj_file(
        PathBuf::from("./data/obj/Cheem.OBJ"),
        PathBuf::from("./data/dag/Cheem.svdag"),
        12
    );
}

#[test]
fn test_voxel_dag_obj_shell_teapot() {

    convert_obj_file(
        PathBuf::from("./data/obj/teapot.obj"),
        PathBuf::from("./data/dag/teapot.svdag"),
        12
    );

}

#[test]
fn test_voxel_dag_obj_shell_sponza() {

    convert_obj_file_with_materials(
        PathBuf::from("./data/obj/Sponza/sponza.obj"),
        PathBuf::from("./data/dag/sponza_mats.svdag"),
        PathBuf::from("./data/dag/sponza_mats.mats"),
        12
    );
}

#[test]
fn test_voxel_dag_obj_shell_sponza_textured() {

    convert_obj_file_textured(
        PathBuf::from("./data/obj/sponza-modified/sponza.obj"),
        PathBuf::from("./data/dag/sponza_tex_1k.svdag"),
        PathBuf::from("./data/dag/sponza_tex_1k.mats"),
        10
    );
}

#[test]
fn test_voxel_dag_obj_shell_sibenik() {
    convert_obj_file_with_materials(
        PathBuf::from("./data/obj/sibenik/sibenik.obj"),
        PathBuf::from("./data/dag/sibenik_mats.svdag"),
        PathBuf::from("./data/dag/sibenik_mats.mats"),
        10
    );
}


#[test]
fn test_voxel_dag_obj_shell_hairball() {
    
    convert_obj_file(
        PathBuf::from("./data/obj/hairball.obj"),
        PathBuf::from("./data/dag/hairball.svdag"),
        9
    );

}


#[test]
fn test_voxel_dag_tri_shell() {
    use std::path::Path;
    use std::fs;
    
    let v0 = Vec3::new(1.0, 0.0, 0.0);
    let v1 = Vec3::new(0.0, 1.0, 0.0);
    let v2 = Vec3::new(0.0, 0.0, 1.0);

    let triangles = vec![
        Triangle{
            points : [v0, v1, v2],
            normal : (v0 - v1).cross(v1 - v2),
            mat    : 1,
            ..Default::default()
        }
    ];

    let min = Vec3::new(0.0, 0.0, 0.0);
    let size = 1.0;

    println!("Triangles: {}", triangles.len());

    use std::time::*;

    let mut pb = ProgressBar::new(8*8*8*8);

    let start = Instant::now();
    let vchunk = VoxelChunk::from_mesh(8, &triangles, min, size, &mut |t| { pb.total = t; pb.inc(); });
    let elapsed = start.elapsed();
    
    pb.finish();

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
fn test_voxel_dag_intersection_sphere() {
    use std::time::*;
    use std::fs;
    use bincode;
    use std::path::*;

    println!("Creating SVDAG from sphere intersection test...");
    
    let start = Instant::now();

    let sc = Vec3::new(0.5, 0.5, 0.5);
    let sr = 0.01;

    let depth = 10;
    let uncompressed_size = (1 << depth) * (1 << depth) * (1 << depth);

    let chunk = VoxelChunk::from_intersection_test(depth, |v, s| {

        let mut sq_dist = 0.0;

        if sc.x < v.x - s { sq_dist += (v.x - s - sc.x).powi(2); }
        if sc.x > v.x + s { sq_dist += (v.x + s - sc.x).powi(2); }
        
        if sc.y < v.y - s { sq_dist += (v.y - s - sc.y).powi(2); }
        if sc.y > v.y + s { sq_dist += (v.y + s - sc.y).powi(2); }
        
        if sc.z < v.z - s { sq_dist += (v.z - s - sc.z).powi(2); }
        if sc.z > v.z + s { sq_dist += (v.z + s - sc.z).powi(2); }

        sq_dist < sr * sr
    });

    let runtime = start.elapsed();

    println!("Compression took {:?}", runtime);
    
    println!("Num Voxels: {} (uncompressed: {} - ) [{:3.3}%]", chunk.voxels.len(), uncompressed_size, 100.0 * chunk.voxels.len() as f32 / uncompressed_size as f32);

    assert!(!chunk.detect_cycles(), "Cycle Detected!");

    let out_path = Path::new("./data/dag/sphere.svdag");

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