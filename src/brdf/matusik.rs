
const BRDF_SAMPLING_RES_THETA_H : usize = 90;
const BRDF_SAMPLING_RES_THETA_D : usize = 90;
const BRDF_SAMPLING_RES_PHI_D   : usize = 360;

const RED_SCALE : f32 = (1.0/1500.0);
const GREEN_SCALE : f32 = (1.15/1500.0);
const BLUE_SCALE : f32 = (1.66/1500.0);
const M_PI : f32 = 3.1415926535897932384626433832795;

// cross product of two vectors
fn cross_product (v1 : [f32; 3], v2 : [f32; 3]) -> [f32;3]
{
    [
        v1[1]*v2[2] - v1[2]*v2[1],
        v1[2]*v2[0] - v1[0]*v2[2],
        v1[0]*v2[1] - v1[1]*v2[0]
    ]
}

// normalize vector
fn normalize(v : &mut [f32; 3])
{
	// normalize
	let len = (v[0]*v[0]+v[1]*v[1]+v[2]*v[2]).sqrt();
	v[0] = v[0] / len;
	v[1] = v[1] / len;
	v[2] = v[2] / len;
}

// rotate vector along one axis
fn rotate_vector(vector : [f32; 3], axis : [f32; 3], angle : f32) -> [f32; 3]
{
    let mut out = [0.0; 3];
	let mut temp;
	let cos_ang = angle.cos();
	let sin_ang = angle.sin();

	out[0] = vector[0] * cos_ang;
	out[1] = vector[1] * cos_ang;
	out[2] = vector[2] * cos_ang;

	temp = axis[0]*vector[0]+axis[1]*vector[1]+axis[2]*vector[2];
	temp = temp*(1.0-cos_ang);

	out[0] += axis[0] * temp;
	out[1] += axis[1] * temp;
	out[2] += axis[2] * temp;

	let cross = cross_product (axis,vector);
	
	out[0] += cross[0] * sin_ang;
	out[1] += cross[1] * sin_ang;
    out[2] += cross[2] * sin_ang;
    
    out
}


// convert standard coordinates to half vector/difference vector coordinates
fn std_coords_to_half_diff_coords(theta_in : f32, fi_in : f32, theta_out : f32, fi_out : f32)
    -> (f32, f32, f32, f32)
{

	// compute in vector
	let in_vec_z = theta_in.cos();
	let proj_in_vec = theta_in.sin();
	let in_vec_x = proj_in_vec*fi_in.cos();
	let in_vec_y = proj_in_vec*fi_in.sin();
	let mut in_vec = [in_vec_x,in_vec_y,in_vec_z];
	normalize(&mut in_vec);


	// compute out vector
	let out_vec_z = theta_out.cos();
	let proj_out_vec = theta_out.sin();
	let out_vec_x = proj_out_vec*fi_out.cos();
	let out_vec_y = proj_out_vec*fi_out.sin();
	let mut out = [out_vec_x,out_vec_y,out_vec_z];
	normalize(&mut out);


	// compute halfway vector
	let half_x = (in_vec_x + out_vec_x)/2.0;
	let half_y = (in_vec_y + out_vec_y)/2.0;
	let half_z = (in_vec_z + out_vec_z)/2.0;
	let mut half = [half_x,half_y,half_z];
	normalize(&mut half);

	// compute  theta_half, fi_half
	let theta_half = half[2].acos();
	let fi_half = half[1].atan2(half[0]);


	let bi_normal = [0.0, 1.0, 0.0];
	let normal = [0.0, 0.0, 1.0];

	// compute diff vector
	let temp = rotate_vector(in_vec, normal , -fi_half);
	let diff = rotate_vector(temp, bi_normal, -theta_half);
	
	// compute  theta_diff, fi_diff	
	let theta_diff = diff[2].acos();
	let fi_diff = diff[1].atan2(diff[0]);

    (theta_half, fi_half, theta_diff, fi_diff)
}


// Lookup theta_half index
// This is a non-linear mapping!
// In:  [0 .. pi/2]
// Out: [0 .. 89]
#[inline]
fn theta_half_index(theta_half : f32) -> usize
{
	if theta_half <= 0.0 {
        return 0;
    }
	let theta_half_deg = (theta_half / (M_PI/2.0)) * BRDF_SAMPLING_RES_THETA_H as f32;
    let temp = (theta_half_deg * BRDF_SAMPLING_RES_THETA_H as f32).sqrt();
    
	let mut ret_val = temp.floor() as isize;
	if ret_val < 0 {
        ret_val = 0;
    }
	if ret_val >= BRDF_SAMPLING_RES_THETA_H as isize {
        ret_val = BRDF_SAMPLING_RES_THETA_H as isize - 1;
    }
	return ret_val as usize;
}


// Lookup theta_diff index
// In:  [0 .. pi/2]
// Out: [0 .. 89]
#[inline]
fn theta_diff_index(theta_diff : f32) -> usize
{
	let tmp = (theta_diff / (M_PI * 0.5) * BRDF_SAMPLING_RES_THETA_D as f32).floor() as isize;
	if tmp < 0 {
        return 0;
    } else if tmp < BRDF_SAMPLING_RES_THETA_D as isize - 1 {
        return tmp as usize;
    } else {
        return BRDF_SAMPLING_RES_THETA_D - 1;
    }
}


// Lookup phi_diff index
#[inline]
fn phi_diff_index(mut phi_diff : f32) -> usize
{
	// Because of reciprocity, the BRDF is unchanged under
	// phi_diff -> phi_diff + M_PI
	if phi_diff < 0.0 {
        phi_diff += M_PI;
    }

	// In: phi_diff in [0 .. pi]
	// Out: tmp in [0 .. 179]
	let tmp = (phi_diff / M_PI * BRDF_SAMPLING_RES_PHI_D as f32 / 2.0).floor() as isize;
	if tmp < 0 {	
        return 0;
    } else if tmp < BRDF_SAMPLING_RES_PHI_D as isize / 2 - 1 {
		return tmp as usize;
    } else {
        return BRDF_SAMPLING_RES_PHI_D / 2 - 1;
    }
}


// Given a pair of incoming/outgoing angles, look up the BRDF.
pub fn lookup_brdf_val(brdf : &[f32], theta_in : f32, fi_in : f32, theta_out : f32, fi_out : f32)
    -> (f32, f32, f32)
{
	// Convert to halfangle / difference angle coordinates
	
	let (theta_half, fi_half, theta_diff, fi_diff) = std_coords_to_half_diff_coords(theta_in, fi_in, theta_out, fi_out);


	// Find index.
	// Note that phi_half is ignored, since isotropic BRDFs are assumed
	let ind = phi_diff_index(fi_diff) +
		  theta_diff_index(theta_diff) * BRDF_SAMPLING_RES_PHI_D / 2 +
		  theta_half_index(theta_half) * BRDF_SAMPLING_RES_PHI_D / 2 *
					         BRDF_SAMPLING_RES_THETA_D;

	let red_val = brdf[ind] * RED_SCALE;
	let green_val = brdf[ind + BRDF_SAMPLING_RES_THETA_H*BRDF_SAMPLING_RES_THETA_D*BRDF_SAMPLING_RES_PHI_D/2] * GREEN_SCALE;
	let blue_val = brdf[ind + BRDF_SAMPLING_RES_THETA_H*BRDF_SAMPLING_RES_THETA_D*BRDF_SAMPLING_RES_PHI_D] * BLUE_SCALE;

	
	if red_val < 0.0 || green_val < 0.0 || blue_val < 0.0 {
        eprintln!("Below horizon.");
    }

    (red_val, green_val, blue_val)
}