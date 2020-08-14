use super::*;

use pbr::ProgressBar;

use std::path::{Path, PathBuf};

fn recursively_subdivide(triangle : Triangle, area_cutoff : f32, buf : &mut Vec<Triangle>) {
    if triangle.area() < area_cutoff {
        buf.push(triangle);
    } else {

        //
        //         a
        //        / \
        //       d---f
        //      / \ / \
        //     b---e---c
        //

        let a = triangle.points[0];
        let b = triangle.points[1];
        let c = triangle.points[2];

        let d = 0.5 * (a + b);
        let e = 0.5 * (b + c);
        let f = 0.5 * (c + a);

        let ta = triangle.uv[0];
        let tb = triangle.uv[1];
        let tc = triangle.uv[2];

        let td = 0.5 * (ta + tb);
        let te = 0.5 * (tb + tc);
        let tf = 0.5 * (tc + ta);

        let t_adf = Triangle { points : [a, d, f], uv : [ta, td, tf], ..triangle };
        let t_bed = Triangle { points : [b, e, d], uv : [tb, te, td], ..triangle };
        let t_def = Triangle { points : [d, e, f], uv : [td, te, tf], ..triangle };
        let t_cfe = Triangle { points : [c, f, e], uv : [tc, tf, te], ..triangle };
        
        recursively_subdivide(t_adf, area_cutoff, buf);
        recursively_subdivide(t_bed, area_cutoff, buf);
        recursively_subdivide(t_def, area_cutoff, buf);
        recursively_subdivide(t_cfe, area_cutoff, buf);
    }
}
use obj::Obj;

pub fn convert_obj_file_textured(obj_file : PathBuf, svdag_file : PathBuf, mat_file : PathBuf, depth : usize){
    use std::path::Path;
    use std::fs;

    let mut obj_data : Obj = Obj::load(&obj_file).expect("Failed to load obj file");


    let mut triangles = vec![];

    let mut materials = HashMap::new();

    use std::path::PathBuf;
    let mut obj_root = obj_file.clone();
    obj_root.set_file_name("");
    let mut material_mat_list = vec![];
    let mut material_idx_list = vec![];
    let mut material_tex_list = vec![];
    let mut material_col_list = vec![];

    let mut next_material = 0;

    for mtl in obj_data.data.material_libs.iter_mut() {
        use std::io::Read;
        use std::fs::File;

        println!("Reloading: {:?}", mtl.filename);

        mtl.reload(File::open(&obj_root.join(&mtl.filename)).unwrap()).unwrap();

        for mat in &mtl.materials {

            let nid = materials.len();
            materials.entry(mat.name.clone()).or_insert(nid);
            material_idx_list.push(next_material);

            let mat_offset = next_material;

            let mut unique_colors = HashSet::new();

            if let Some(kd_tex_file) = &mat.map_kd {
                
                println!("Loading texture: {:?}", kd_tex_file);

                let img = read_image_maybe_tga(obj_root.join(kd_tex_file));

                let img = img.into_rgb();
                
                println!("  Finding Unique Colors...");

                for (_,_,&image::Rgb(p)) in img.enumerate_pixels() {
                    unique_colors.insert(p);
                }

                println!("  Unique Colors: {}", unique_colors.len());
                
                next_material += unique_colors.len();

                material_tex_list.push(Some(img));
                material_col_list.push(Some(unique_colors.iter().cloned().collect::<Vec<_>>()));
            } else {
                next_material += 1;

                material_tex_list.push(None);
                material_col_list.push(None);
            }

            println!("  Material Offset: {}", mat_offset);

            let kdd = mat.kd.unwrap_or([0.0; 3]);

            material_mat_list.push(Material {
                albedo : kdd,
                metalness : mat.km.unwrap_or(0.0),
                emission : mat.ke.unwrap_or([0.0; 3]),
                roughness : 0.3,
            });

        }
    }


    println!("Material Count: {}", materials.len());

    println!("Processing Triangles...");
    
    let mut min = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
    let mut max = Vec3::new(f32::MIN, f32::MIN, f32::MIN);

    for &[x,y,z] in &obj_data.data.position {
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

    println!("Max Size: {}", max_size);

    let area_cutoff = 4.0 * (max_size * max_size) / (4.0f32.powf(depth as f32));

    println!("Area Cutoff: {}", area_cutoff);

    for o in 0..(obj_data.data.objects.len()) {
        let object = &obj_data.data.objects[o];
        for g in 0..(object.groups.len()) {
            let group = &object.groups[g];

            let next = materials.len();

            let id = if let Some(obj::ObjMaterial::Ref(s)) = &group.material {
                *materials.entry(s.clone()).or_insert(next)
            } else {
                0
            };

            for p in 0..(group.polys.len()) {
                let poly = &group.polys[p];
                for v in 2..(poly.0.len()) {
                    let v0 = obj_data.data.position[poly.0[0].0];
                    let v1 = obj_data.data.position[poly.0[v-1].0];
                    let v2 = obj_data.data.position[poly.0[v].0];
                    
                    let t0 = poly.0[0].1  .map(|i| obj_data.data.texture[i]).unwrap_or([0.0,0.0]);
                    let t1 = poly.0[v-1].1.map(|i| obj_data.data.texture[i]).unwrap_or([0.0,0.0]);
                    let t2 = poly.0[v].1  .map(|i| obj_data.data.texture[i]).unwrap_or([0.0,0.0]);

                    let v0 = Vec3::new(v0[0], v0[1], v0[2]);
                    let v1 = Vec3::new(v1[0], v1[1], v1[2]);
                    let v2 = Vec3::new(v2[0], v2[1], v2[2]);
                    
                    let t0 = Vec2::new(t0[0], t0[1]);
                    let t1 = Vec2::new(t1[0], t1[1]);
                    let t2 = Vec2::new(t2[0], t2[1]);
                    
                    let t = Triangle{
                        points : [v0, v1, v2],
                        normal : (v0 - v1).cross(v1 - v2),
                        uv : [t0, t1, t2],
                        mat    : material_idx_list[id] as u16,
                        ..Default::default()
                    };

                    let t_start = triangles.len();

                    recursively_subdivide(t, area_cutoff, &mut triangles);

                    let ref tex = material_tex_list[id];
                    let ref col = material_col_list[id];

                    if let (Some(t), Some(col)) = (tex, col) {
                        for tri in triangles[t_start..].iter_mut() {
                            let cuv = tri.uv_center();
                            let c = texture_lookup(&t, cuv.x, cuv.y);
                            let i = col.iter().enumerate().find_map(|(i, p)| if *p == c {Some(i)} else {None}).unwrap();
                            tri.mat += i as u16;
                        }
                    }
                }
            }
        }
    }

    println!("Triangles: {}", triangles.len());

    let material_list = (0..(material_mat_list.len()))
        .flat_map(|i| {
            let m = &material_mat_list[i];
            if let Some(ref c) = material_col_list[i] {
                c.iter()
                    .map(|c| Material {
                        albedo : [
                            c[0] as f32 / 255.0,
                            c[1] as f32 / 255.0,
                            c[2] as f32 / 255.0
                        ],
                        ..*m
                    })
                    .collect::<Vec<_>>()
            } else {
                vec![*m]
            }
        })
        .collect::<Vec<_>>();

    println!("Material Count: {}", material_list.len());
    
    println!("Constructing SVDAG...");
    use std::time::*;
    
    let mut pb = ProgressBar::new(8*8*8*8);

    let start = Instant::now();
    let vchunk = VoxelChunk::from_mesh(depth, &triangles, min, max_size, &mut |t| { pb.total = t; pb.inc(); });
    let elapsed = start.elapsed();

    pb.finish();
    
    println!("Time to voxelize: {:?}", elapsed);
    println!("DAG nodes: {}", vchunk.len());
    
    let serialized = bincode::serialize(&vchunk).unwrap();
    let serialized_mats = bincode::serialize(&material_list).unwrap();

    fs::write(svdag_file, serialized).unwrap();
    fs::write(mat_file, serialized_mats).unwrap();
}

pub fn convert_obj_file_with_materials(obj_file : PathBuf, svdag_file : PathBuf, mat_file : PathBuf, depth : usize){
    use obj::Obj;
    use std::path::Path;
    use std::fs;

    let mut obj_data = Obj::load(&obj_file).expect("Failed to load obj file");


    let mut triangles = vec![];

    let mut materials = HashMap::new();

    use std::path::PathBuf;
    let mut obj_root = obj_file.clone();
    obj_root.set_file_name("");
    let mut material_list = vec![Material::default(); materials.len()];

    for mtl in obj_data.data.material_libs.iter_mut() {
        use std::io::Read;
        use std::fs::File;

        println!("Reloading: {:?}", mtl.filename);

        mtl.reload(File::open(&obj_root.join(&mtl.filename)).unwrap()).unwrap();

        for mat in &mtl.materials {
            let kd = if let Some(kd_tex_file) = &mat.map_kd {
                
                println!("Loading texture: {:?}", kd_tex_file);

                let img = read_image_maybe_tga(obj_root.join(kd_tex_file));

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
            let _id   = *materials.entry(mat.name.clone()).or_insert(next);

            let mut kdd = mat.kd.unwrap_or([0.0; 3]);

            kdd[0] *= kd[0];
            kdd[1] *= kd[1];
            kdd[2] *= kd[2];

            material_list.push(Material {
                albedo : kdd,
                metalness : mat.km.unwrap_or(0.0),
                emission : mat.ke.unwrap_or([0.0; 3]),
                roughness : 0.3,
            });
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
                        ..Default::default()
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
    let mut pb = ProgressBar::new(8*8*8*8);
    
    let start = Instant::now();
    let vchunk = VoxelChunk::from_mesh(depth, &triangles, min, max_size, &mut |t| { pb.total = t; pb.inc(); });
    let elapsed = start.elapsed();
    
    pb.finish();
    
    println!("Time to voxelize: {:?}", elapsed);
    println!("DAG nodes: {}", vchunk.len());
    
    let serialized = bincode::serialize(&vchunk).unwrap();
    let serialized_mats = bincode::serialize(&material_list).unwrap();

    fs::write(svdag_file, serialized).unwrap();
    fs::write(mat_file, serialized_mats).unwrap();
}

pub fn convert_obj_file(obj_file : PathBuf, svdag_file : PathBuf, depth : usize){
    use obj::Obj;
    use std::path::Path;
    use std::fs;

    let obj_data = Obj::load(&obj_file).expect("Failed to load obj file");

    let mut triangles = vec![];

    use std::path::PathBuf;

    println!("Processing Triangles...");

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
                        mat    : 0,
                        ..Default::default()
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

    let mut pb = ProgressBar::new(8*8*8*8);
    
    let start = Instant::now();
    let vchunk = VoxelChunk::from_mesh(depth, &triangles, min, max_size, &mut |t| { pb.total = t; pb.inc(); });
    let elapsed = start.elapsed();

    pb.finish();
    
    println!("Time to voxelize: {:?}", elapsed);
    println!("DAG nodes: {}", vchunk.len());
    
    let serialized = bincode::serialize(&vchunk).unwrap();

    fs::write(svdag_file, serialized).unwrap();
}

use image;
fn read_image_maybe_tga<P : AsRef<Path>>(path : P) -> image::DynamicImage {
    let path : &Path = path.as_ref();
    let bytes = std::fs::read(path).unwrap();

    let byte_stream = std::io::Cursor::new(&bytes);

    let mut reader = image::io::Reader::new(byte_stream);

    // somewhat sketchy logic to deal with some tga files I had
    if path.extension().map(|ext| ext.to_string_lossy().to_string()) == Some("tga".to_string()) {
        reader.set_format(image::ImageFormat::Tga);
    } else {
        reader = reader.with_guessed_format().unwrap();
    }

    let image = reader.decode().unwrap();

    image    
}

