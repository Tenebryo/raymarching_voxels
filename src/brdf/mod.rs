use na::DMatrix;
use na::Vector3;
use serde::{Serialize, Deserialize};
use std::path::Path;
use std::io::prelude::*;
use std::fs::File;
use crate::*;

use rand::prelude::*;

use vulkano::buffer::cpu_access::WriteLock;

use std::f32::consts::PI;

mod matusik;



#[derive(Clone, Serialize, Deserialize)]
pub struct BRDF {
    n : [usize;4],
    j : usize,
    k : usize,
    /// F factor matrix (n.0 * n.1 * j * k)
    f : Vec<f32>,
    /// u factor matrix (j * k * n.2)
    u : Vec<f32>,
    /// v factor matrix (j * k * n.3)
    v : Vec<f32>,
}

impl BRDF {
    /// creates a Phong BRDF, then computes the factored form for the
    pub fn phong_brdf(ks : f32, kd : f32, ka : f32, a : f32, shape : [usize;4], j : usize, k : usize) -> Self {
        let mut sampled_data = Vec::with_capacity(shape[0] * shape[1] * shape[2] * shape[3]);

        for theta_out_i in 0..shape[0] {
            println!("theta_out_i : {}", theta_out_i);
            let theta_out = theta_angle(theta_out_i as isize, shape[0] as isize);
            for phi_out_i in 0..shape[1] {
                let phi_out = phi_angle(phi_out_i as isize, shape[1] as isize);
                for theta_half_i in 0..shape[2] {
                    let theta_half = theta_half_angle(theta_half_i as isize, shape[2] as isize);
                    for phi_half_i in 0..shape[3] {
                        let phi_half = phi_angle(phi_half_i as isize, shape[3] as isize);

                        let (theta_in, phi_in) = get_angles_from_out_half_angles(theta_out, phi_out, theta_half, phi_half);

                        // calculate phong brdf value
                        // compute in vector
                        let in_vec_z = theta_in.cos();
                        let proj_in_vec = theta_in.sin();
                        let in_vec_x = proj_in_vec*(PI + phi_in).cos();
                        let in_vec_y = proj_in_vec*(PI + phi_in).sin();
                        let mut iv = [in_vec_x,in_vec_y,in_vec_z];
                        let in_mag = (iv[0] * iv[0] + iv[1] * iv[1] + iv[2] * iv[2]).sqrt();
                        iv[0] /= in_mag;
                        iv[1] /= in_mag;
                        iv[2] /= in_mag;


                        // compute out vector
                        let out_vec_z = theta_out.cos();
                        let proj_out_vec = theta_out.sin();
                        let out_vec_x = proj_out_vec*phi_out.cos();
                        let out_vec_y = proj_out_vec*phi_out.sin();
                        let mut ov = [out_vec_x,out_vec_y,out_vec_z];
                        let out_mag = (ov[0] * ov[0] + ov[1] * ov[1] + ov[2] * ov[2]).sqrt();
                        ov[0] /= out_mag;
                        ov[1] /= out_mag;
                        ov[2] /= out_mag;

                        let mut dot = ov[0] * iv[0] + ov[1] * iv[1] + ov[2] * iv[2];
                        if dot < 0.0001 {
                            dot = 0.0001;
                        }

                        let mut cos = theta_in.cos();
                        if cos < 0.0001 {
                            cos = 0.0001;
                        }

                        let illumination = kd * cos + ks * dot.powf(a);

                        // println!("{}", illumination);

                        sampled_data.push(illumination);
                    }
                }
            }
        }

        Self::factor_measured_brdf(&sampled_data, shape, j, k, 32)
    }

    /// reads in a file in the format described by Matusik, resampled to a new resolution using 
    pub fn read_matusik_brdf_file<P: AsRef<Path>>(path: P, shape : [usize; 4], j : usize, k : usize) -> std::io::Result<Self> {

        const BRDF_SAMPLING_RES_THETA_H : usize = 90;
        const BRDF_SAMPLING_RES_THETA_D : usize = 90;
        const BRDF_SAMPLING_RES_PHI_D   : usize = 360;

        // read data in from file
        let mut file = File::open(path)?;

        let mut data = Vec::with_capacity(file.metadata()?.len() as usize);
        let n_read = file.read_to_end(&mut data)?;

        assert!(n_read >= 12);

        let n_theta_h = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let n_theta_d = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
        let n_phi_d   = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;

        let n = n_theta_h * n_theta_d * n_phi_d;

        assert_eq!(n, BRDF_SAMPLING_RES_THETA_H * BRDF_SAMPLING_RES_THETA_D * BRDF_SAMPLING_RES_PHI_D / 2);

        // this isn't very efficient (should just reinterpret the data), but this is easier to understand
        let brdf_data = data[12..(3*std::mem::size_of::<f64>()*n+12)]
            .chunks_exact(std::mem::size_of::<f64>())
            .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
            .collect::<Vec<_>>();


        // matusik parameterized the BRDF data by [theta_half, theta_diff, phi_diff] (isotropic brdf), but we want to parameterize it
        // by [theta_out, phi_out, theta_half, phi_half], so we have to resample the brdf data
        let nn = shape[0] * shape[1] * shape[2] * shape[3];

        let mut resampled_data = Vec::with_capacity(nn);

        for theta_out_i in 0..shape[0] {
            println!("theta_out_i : {}", theta_out_i);
            let theta_out = theta_angle(theta_out_i as isize, shape[0] as isize);
            for phi_out_i in 0..shape[1] {
                let phi_out = phi_angle(phi_out_i as isize, shape[1] as isize);
                for theta_half_i in 0..shape[2] {
                    let theta_half = theta_half_angle(theta_half_i as isize, shape[2] as isize);
                    for phi_half_i in 0..shape[3] {
                        let phi_half = phi_angle(phi_half_i as isize, shape[3] as isize);

                        let (theta_in, phi_in) = get_angles_from_out_half_angles(theta_out, phi_out, theta_half, phi_half);

                        let (r, g, b) = matusik::lookup_brdf_val(&brdf_data, theta_in, phi_in, theta_out, phi_out);

                        resampled_data.push((r + g + b) / 3.0);
                    }
                }
            }
        }

        Ok(Self::factor_measured_brdf(&resampled_data, shape, j, k, 32))
    }

    /// Take a fully measured BRDF in [theta_out, phi_out, theta_half, phi_half] layout, and factor it using the algorithm
    /// from https://gfx.cs.princeton.edu/gfx/proj/brdf/brdf.pdf
    pub fn factor_measured_brdf(data : &[f32], shape : [usize; 4], j : usize, k : usize, iterations : usize) -> Self {

        let wo_n = shape[0] * shape[1];
        let wi_n = shape[2] * shape[3];
        let l_n = j * k;

        //data has shape [n_th_o, n_ph_o, n_th_i, n_ph_i]

        //data is reshaped to [n_th_o * n_ph_o, n_th_i * n_ph_i]


        //f has shape [n_th_o * n_ph_o, j]
        //g has shape [j, n_th_i * n_ph_i]
        // this function call is the big expense since it is factoring a large matrix
        let (f, g) = nmf(DMatrix::from_row_slice(wo_n, wi_n, data), j, iterations);

        assert_eq!(f.shape(), (wo_n, j));
        assert_eq!(g.shape(), (j, wi_n));

        // u_data has shape [j, k, n_th_i]
        // v_data has shape [j, k, n_ph_i]
        let mut u_data = Vec::with_capacity(shape[2] * j * k);
        let mut v_data = Vec::with_capacity(shape[3] * j * k);

        for ji in 0..j {

            // reshape each column of g to [n_th_u, n_phi_i] and factor

            let (u,v) = nmf(DMatrix::from_row_slice(shape[2], shape[3], &g.row(ji).iter().cloned().collect::<Vec<_>>()), k, iterations);

            assert_eq!(u.shape(), (shape[2], k));
            assert_eq!(v.shape(), (k, shape[3]));
            //u has shape [n_th_i, k]
            //v has shape [k, n_ph_i]

            u_data.extend_from_slice(u.transpose().as_slice());
            v_data.extend_from_slice(v.as_slice());
        }


        // f_data has shape [j*k, n_th_o * n_ph_o]
        let mut f_data = (0..(wo_n * l_n)).map(|_| 0.0).collect::<Vec<f32>>();

        // *jk_means have shape [j,k]
        let mut ujk_means = Vec::with_capacity(j * k);
        let mut vjk_means = Vec::with_capacity(j * k);

        // normalize matrices
        for ji in 0..j {
            for ki in 0..k {
                let mut ujk_sum = 0.0;
                let mut vjk_sum = 0.0;

                let li = ki + k * ji;

                // calculate row means
                for i in 0..shape[2] {
                    ujk_sum += u_data[i + shape[2] * li];
                }
                
                for i in 0..shape[3] {
                    vjk_sum += v_data[i + shape[3] * li];
                }

                ujk_sum /= shape[2] as f32;
                vjk_sum /= shape[3] as f32;
                
                ujk_means.push(ujk_sum);
                vjk_means.push(vjk_sum);

                for i in 0..wo_n {
                    f_data[li + l_n * i] = ujk_sum * vjk_sum * f[(i, ji)];
                }

                for i in 0..shape[2] {
                    u_data[i + shape[2] * li] /= ujk_sum;
                }
                
                for i in 0..shape[3] {
                    v_data[i + shape[3] * li] /= vjk_sum;
                }
            }
        }

        Self {
            n : shape,
            j,
            k,
            /// F factor matrix (n.0 * n.1 * j * k)
            f : f_data,
            /// u factor matrix (j * k * n.2)
            u : u_data,
            /// v factor matrix (j * k * n.3)
            v : v_data,
        }
    }

    /// Write the factored BRDF to a linear buffer for use in a compute shader.
    /// returns the sizes of each of the matrix buffers
    pub fn write_to_buffer(&self, buf : &mut WriteLock<[f32]>, i : usize) {
        let nf = self.f.len();
        let nu = self.u.len();
        let nv = self.v.len();

        buf[i..(i+nf)].clone_from_slice(&self.f);
        buf[(i+nf)..(i+nf+nu)].clone_from_slice(&self.u);
        buf[(i+nf+nu)..(i+nf+nu+nv)].clone_from_slice(&self.v);
    }

    // pub fn create_shader_type(&self, offset : usize) -> shaders::BRDF {

    //     shaders::BRDF {
    //         n : [self.n[0] as u32, self.n[1] as u32, self.n[2] as u32, self.n[3] as u32],
    //         l : (self.j * self.k) as u32,
    //         f_idx : (offset) as u32,
    //         u_idx : (offset + self.f.len()) as u32,
    //         v_idx : (offset + self.f.len() + self.u.len()) as u32,
    //     }
    // }

    pub fn len(&self) -> usize {
        self.f.len() + self.u.len() + self.v.len()
    }
}

/// Lookup theta_half index from theta half angle (non-linear map)
/// In:  [0 .. pi/2]
/// Out: [0 .. n]
#[inline]
fn theta_half_index(theta_half : f32, n : isize) -> usize {
	if theta_half <= 0.0 {
        return 0;
    }
	let theta_half_n = (theta_half / (PI/2.0)) * n as f32;
    let temp = (theta_half_n * n as f32).sqrt();
    
	let mut ret_val = temp.floor() as isize;
	if ret_val < 0 {
        ret_val = 0;
    }
	if ret_val >= n {
        ret_val = n - 1;
    }
	return ret_val as usize;
}

/// Lookup theta_half angle from theta half index (non-linear map)
/// In:  [0 .. n]
/// Out: [0 .. pi/2]
#[inline]
fn theta_half_angle(theta_half : isize, n : isize) -> f32 {
	if theta_half <= 0 {
        return 0.0;
    }
    if theta_half >= n - 1 {
        return PI/2.0;
    }

    return (PI / 2.0) * (theta_half * theta_half) as f32 / ((n * n) as f32);
}

/// Lookup theta index from theta angle (linear map)
/// In:  [0 .. pi/2]
/// Out: [0 .. n]
#[inline]
fn theta_index(theta : f32, n : isize) -> usize {
	if theta <= 0.0 {
        return 0;
    }
	let theta_n = (theta / (PI/2.0)) * n as f32;
    
	let mut ret_val = theta_n.floor() as isize;
	if ret_val < 0 {
        ret_val = 0;
    }
	if ret_val >= n {
        ret_val = n - 1;
    }
	return ret_val as usize;
}

/// Lookup theta angle from theta index (linear map)
/// In:  [0 .. n]
/// Out: [0 .. pi/2]
#[inline]
fn theta_angle(theta : isize, n : isize) -> f32 {
	if theta <= 0 {
        return 0.0;
    }
    if theta >= n - 1 {
        return PI/2.0;
    }

    return (PI / 2.0) * theta as f32 / (n as f32);
}

/// Lookup phi index from phi half angle (linear map)
/// In:  [0 .. 2*pi]
/// Out: [0 .. n]
#[inline]
fn phi_index(theta_half : f32, n : isize) -> usize {
	if theta_half <= 0.0 {
        return 0;
    }
	let theta_half_n = (theta_half / (PI*2.0)) * n as f32;
    
	let mut ret_val = theta_half_n.floor() as isize;
	if ret_val < 0 {
        ret_val = 0;
    }
	if ret_val >= n {
        ret_val = n - 1;
    }
	return ret_val as usize;
}

/// Lookup theta_half angle from theta half index (linear map)
/// In:  [0 .. n]
/// Out: [0 .. 2*pi]
#[inline]
fn phi_angle(theta_half : isize, n : isize) -> f32 {
	if theta_half <= 0 {
        return 0.0;
    }
    if theta_half >= n - 1 {
        return PI * 2.0;
    }

    return (PI * 2.0) * theta_half as f32 / (n as f32);
}

/// calculates the incoming incident ray from the outbound ray and the half angle vector (angle formats)
fn get_angles_from_out_half_angles(theta_out : f32, phi_out : f32, theta_half : f32, phi_half : f32) -> (f32, f32) {

    // out vector
	let out_vec_z = theta_out.cos();
	let proj_out_vec = theta_out.sin();
	let out_vec_x = proj_out_vec*phi_out.cos();
	let out_vec_y = proj_out_vec*phi_out.sin();
    let out = Vector3::new(out_vec_x,out_vec_y,out_vec_z).normalize();
    
    // half vector
	let hlf_vec_z = theta_half.cos();
	let proj_hlf_vec = theta_half.sin();
	let hlf_vec_x = proj_hlf_vec*phi_half.cos();
	let hlf_vec_y = proj_hlf_vec*phi_half.sin();
    let hlf = Vector3::new(hlf_vec_x,hlf_vec_y,hlf_vec_z).normalize();

    let dot = out.dot(&hlf);

    let in_vec = out - 2.0 * (out - dot * hlf);

    let theta_in = in_vec[2].acos();
    let phi_in = in_vec[1].atan2(in_vec[0]);

    (theta_in, phi_in)
}

/// factors a n*m matrix into a n*j matrix G and a j*m matrix F 
fn nmf(y : DMatrix<f32>, factors : usize, iterations : usize) -> (DMatrix<f32>, DMatrix<f32>) {
    const RANDOM_SCALE : f32 = 0.01;

    let nr = y.nrows();
    let nc = y.ncols();

    let mut rng = thread_rng();

    // initialize with non-negative random values
    let f_init = (0..(factors * nc)).map(|_| rng.gen_range(0.0,RANDOM_SCALE)).collect::<Vec<_>>();
    let g_init = (0..(factors * nr)).map(|_| rng.gen_range(0.0,RANDOM_SCALE)).collect::<Vec<_>>();

    let mut f = DMatrix::from_row_slice(factors, nc, &f_init);
    let mut g = DMatrix::from_row_slice(nr, factors, &g_init);
    for _ in 0..iterations {

        let gf = g.clone() * f.clone();

        for i in 0..factors {
            for j in 0..nc {
                let mut sum = 0.0;
                for k in 0..nr {
                    sum += g[(k, i)] * y[(k, j)] / gf[(k,j)];
                }
                f[(i,j)] *= sum;
            }
        }

        let gf = g.clone() * f.clone();

        for i in 0..nr {
            for j in 0..factors {
                let mut sum = 0.0;
                for k in 0..nc {
                    sum += f[(j, k)] * y[(i, k)] / gf[(i,k)];
                }
                g[(i,j)] *= sum;
            }
        }
        
        for j in 0..factors {
            let mut sum = 0.0;
            for i in 0..nr {
                sum += g[(i,j)];
            }

            for i in 0..nr {
                g[(i,j)] /= sum;
            }
        }

    }

    (g, f)
}


#[test]
fn brdf_nmf_factoring_test() {
    let mut rng = thread_rng();

    let y = DMatrix::from_row_slice(256, 128, &(0..(256*128)).map(|_| rng.gen_range(0.0, 1.0)).collect::<Vec<_>>());

    use std::time::*;

    let start = Instant::now();

    let (g,f) = nmf(y.clone(), 16, 32);

    let dur = start.elapsed();

    let gf = (&g) * (&f);

    let mut sum = 0.0;
    for r in 0..256 {
        for c in 0..128 {
            // calculate error
            let e = y[(r,c)] * (y[(r,c)] / gf[(r,c)]).ln() - y[(r,c)] + gf[(r,c)];
            sum += (e * e).sqrt();
        }
    }

    println!("Error: {}", sum / (256.0 * 128.0));
    println!("Time: {:?}", dur);
}


#[test]
fn brdf_matusik_convert_test() {
    use std::time::Instant;

    let shape = [16,16,128,16];

    let start = Instant::now();
    let brdf = brdf::BRDF::read_matusik_brdf_file(
        Path::new("data/teflon.bin"), 
        shape, 2, 2
    ).expect("could not load brdf file");
    let elapsed = start.elapsed();

    let mut output_file = File::create(
        format!("data/teflon-{}-{}-{}-{}.brdf", shape[0], shape[1], shape[2], shape[3]))
        .expect("could not create output file");

    output_file.write_all(&bincode::serialize(&brdf).expect("failed to serialize brdf")).expect("failed to write to output file");

    println!("f: {:?}", brdf.f);
    println!("u: {:?}", brdf.u);
    println!("v: {:?}", brdf.v);

    println!("Time to process: {:?}", elapsed);
    println!("BRDF Size: {}B", brdf.len() * 4);
}

#[test]
fn brdf_phong_test() {
    use std::time::Instant;

    let shape = [16,16,128,16];

    let start = Instant::now();
    let brdf = brdf::BRDF::phong_brdf(
        0.51, 0.51, 0.51, 0.51,
        shape, 1, 1
    );
    let elapsed = start.elapsed();

    let mut output_file = File::create(
        format!("data/phong-{}-{}-{}-{}.brdf", shape[0], shape[1], shape[2], shape[3]))
        .expect("could not create output file");

    output_file.write_all(&bincode::serialize(&brdf).expect("failed to serialize brdf")).expect("failed to write to output file");

    println!("f: {:?}", brdf.f);
    println!("u: {:?}", brdf.u);
    println!("v: {:?}", brdf.v);

    println!("Time to process: {:?}", elapsed);
    println!("BRDF Size: {}B", brdf.len() * 4);
}
