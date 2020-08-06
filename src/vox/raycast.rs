// this file is a roughly 1-to-1 translation of the GPU shader for use on the CPU

use crate::vox::VChildDescriptor;

use cgmath::prelude::*;
use cgmath::Vector2;
use cgmath::Vector3;

type Vec2 = Vector2<f32>;
type Vec3 = Vector3<f32>;

type UVec2 = Vector2<u32>;
type UVec3 = Vector3<u32>;


const AXIS_X_MASK : u32 = 1;
const AXIS_Y_MASK : u32 = 2;
const AXIS_Z_MASK : u32 = 4;

const AXIS_MASK_VEC : UVec3 = UVec3::new(AXIS_X_MASK, AXIS_Y_MASK, AXIS_Z_MASK);

const INCIDENCE_X : u32 = 0;
const INCIDENCE_Y : u32 = 1;
const INCIDENCE_Z : u32 = 2;

const INCIDENCE_NORMALS : [Vec3; 3] = [
    Vec3::new(1.0, 0.0, 0.0),
    Vec3::new(0.0, 1.0, 0.0),
    Vec3::new(0.0, 0.0, 1.0)
];

const LOD_CUTOFF_CONSTANT : f32 = 0.002;

fn fma(a : Vec3, b : Vec3, c : Vec3) -> Vec3 {
    a.mul_element_wise(b) + c
}

fn idot(a : UVec3, b : UVec3) -> u32 {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

fn project_cube(id : Vec3, od : Vec3, mn : Vec3, mx : Vec3) -> (u32, Vec2) {

    let tmn = fma(id, mn, od);
    let tmx = fma(id, mx, od);



    let ts = f32::max(tmn.x, f32::max(tmn.y, tmn.z));

    let te = f32::min(tmx.x, f32::min(tmx.y, tmx.z));

    let mut incidence = 0;

    if te == tmx.x {incidence = INCIDENCE_X;}
    if te == tmx.y {incidence = INCIDENCE_Y;}
    if te == tmx.z {incidence = INCIDENCE_Z;}
    

    (incidence, Vec2::new(ts, te))
}

fn subvoxel_valid(sv : i32)    -> bool {sv != 0}
fn subvoxel_leaf(sv : i32)     -> bool {sv < 0}
fn subvoxel_empty(sv : i32)    -> bool {sv != 0}
fn subvoxel_child(sv : i32)    -> u32  {sv as u32 - 1}
fn subvoxel_material(sv : i32) -> u32  {(-sv) as u32}

fn voxel_get_subvoxel(voxels : &[VChildDescriptor], parent : u32, idx : u32) -> i32 {
    return voxels[parent as usize].sub_voxels[idx as usize];
}

fn interval_nonempty(t : Vec2) -> bool {
    return t.x < t.y;
}

fn interval_intersect(a : Vec2, b : Vec2) -> Vec2 {
    return Vec2::new(f32::max(a.x,b.x), f32::min(a.y, b.y));
    // return ((b.x > a.y || a.x > b.y) ? vec2(1,0) : vec2(max(a.x,b.x), min(a.y, b.y)));
}

fn select_child(pos : Vec3, scale : f32, o : Vec3, d : Vec3, t : f32) -> u32 {
    let p = fma(d, Vec3::from_value(t), o) - pos - Vec3::from_value(scale);
    // vec3 p = o + d * t - pos - scale;

    let less = UVec3::new(
        if p.x < 0.0 { 1 } else { 0 },
        if p.y < 0.0 { 1 } else { 0 },
        if p.z < 0.0 { 1 } else { 0 }
    );

    idot(less, AXIS_MASK_VEC)
}

fn select_child_bit(pos : Vec3, scale : f32, o : Vec3, d : Vec3, t : f32) -> u32 {
    let p = fma(d, Vec3::new(t,t,t), o) - pos - Vec3::from_value(scale);

    let s = UVec3::new(
        (p.x > 0.0) as u32,
        (p.y > 0.0) as u32,
        (p.z > 0.0) as u32
    );

    idot(s, AXIS_MASK_VEC)
}

fn child_cube(pos : UVec3, scale : u32, idx : u32) -> UVec3 {


    const CUBE_OFFSETS : [UVec3; 8] = [
        UVec3::new(0,0,0),
        UVec3::new(1,0,0),
        UVec3::new(0,1,0),
        UVec3::new(1,1,0),
        UVec3::new(0,0,1),
        UVec3::new(1,0,1),
        UVec3::new(0,1,1),
        UVec3::new(1,1,1)
    ];

    return pos + (scale * CUBE_OFFSETS[idx as usize]);
}

fn highest_differing_bit(a : UVec3, b : UVec3) -> u32 {
    let t : u32 = (a.x ^ b.x) | (a.y ^ b.y) | (a.z ^ b.z);

    31 - t.leading_zeros()
}

fn extract_child_slot(pos : UVec3, scale : u32) -> u32 {

    let  d = UVec3::new(
        (pos.x & scale == 0) as u32,
        (pos.y & scale == 0) as u32,
        (pos.z & scale == 0) as u32
    );

    idot(d, AXIS_MASK_VEC)
}

fn extract_child_slot_bfe(pos : UVec3, depth : u32) -> u32 {

    let d = UVec3::new(
        (pos.x >> depth) & 1,
        (pos.y >> depth) & 1,
        (pos.z >> depth) & 1
    );

    // return AXIS_X_MASK * d.x + AXIS_Y_MASK * d.y + AXIS_Z_MASK * d.z;
    return idot(d, AXIS_MASK_VEC);
}
#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub enum RaycastTermination {
    Miss,
    Hit,
    MaxDepth,
    LoD,
    MaxDist,
    Error,
    LoopEnd,
}

impl Default for RaycastTermination {
    fn default() -> Self {
        RaycastTermination::Miss
    }
}

#[derive(Copy, Clone, Default, Debug)]
pub struct Raycast {
    pub hit : bool,
    pub dist : f32,
    pub incidence : u32,
    pub material : u32,
    pub voxel_id : u32,
    pub termination : RaycastTermination,
    pub iterations : u32,
}

pub fn voxel_march(voxels : &[VChildDescriptor], lod_materials : &[u32], mut o : Vec3, mut d : Vec3, max_depth : u32, max_dist : f32) -> Raycast {

    let mut res = Raycast::default();

    const MAX_DAG_DEPTH : u32 = 16;
    const MAX_SCALE : u32 = 1 << MAX_DAG_DEPTH;

    let mut pstack = [0u32; MAX_DAG_DEPTH as usize];
    let mut tstack = [0.0f32; MAX_DAG_DEPTH as usize];
    let mut dmask = 0;

    d.x = if d.x == 0.0 { 1e-6 } else { d.x };
    d.y = if d.y == 0.0 { 1e-6 } else { d.y };
    d.z = if d.z == 0.0 { 1e-6 } else { d.z };

    let ds = Vec3::new(d.x.signum(), d.y.signum(), d.z.signum());

    d.mul_assign_element_wise(ds);
    // o = o * ds + (1 - ds) * 0.5;
    o = fma(o, ds, (Vec3::from_value(1.0) - ds) * 0.5);

    o *= MAX_SCALE as f32;
    d *= MAX_SCALE as f32;

    dmask |= if ds.x < 0.0 { AXIS_X_MASK } else { 0 };
    dmask |= if ds.y < 0.0 { AXIS_Y_MASK } else { 0 };
    dmask |= if ds.z < 0.0 { AXIS_Z_MASK } else { 0 };

    let id = 1.0 / d;
    let od = - o.mul_element_wise(id);

    let mut t = Vec2::new(0.0, max_dist);

    let mut h = t.y;

    // fix initial position
    let mut pos = UVec3::from_value(0);

    let mut parent = 0;
    let mut idx;

    let mut scale = 1 << MAX_DAG_DEPTH;
    let mut depth = 1;

    let (mut incidence, tp) = project_cube(id, od, pos.cast().unwrap(), pos.add_element_wise(scale).cast().unwrap());

    t = interval_intersect(t, tp);

    res.iterations = 0;

    if !interval_nonempty(t) {
        // we didn't hit the bounding cube
        res.termination = RaycastTermination::Miss;
        res.hit = false;
        return res;
    }

    scale = scale >> 1;
    // idx = select_child(pos, scale, o, d, t.x);
    idx = select_child_bit(pos.cast().unwrap(), scale as f32, o, d, t.x);
    pos = child_cube(pos, scale, idx);

    pstack[0] = parent;
    tstack[0] = t.y;
    let mut tv;

    // very hot loop
    while res.iterations < 2048 {
        res.iterations += 1;


        let (new_incidence, tc) = project_cube(id, od, pos.cast().unwrap(), pos.add_element_wise(scale).cast().unwrap());

        let subvoxel = voxel_get_subvoxel(voxels, parent, dmask ^ idx);

        if subvoxel_valid(subvoxel) && interval_nonempty(t) {

            if scale as f32 <= tc.x * LOD_CUTOFF_CONSTANT as f32 || depth >= max_depth {

                // voxel is too small
                res.dist = t.x;
                res.termination = if depth >= max_depth { RaycastTermination::MaxDepth } else { RaycastTermination::LoD };
                res.material = lod_materials[parent as usize];
                res.voxel_id = (parent << 3) | (dmask ^ idx);
                res.incidence = incidence;
                res.hit = true;
                return res;
            }

            if tc.x > max_dist {
                // voxel is beyond the render distance
                res.termination = RaycastTermination::MaxDist;
                res.hit = false;
                return res;
            }

            tv = interval_intersect(tc, t);

            if interval_nonempty(tv) {
                if subvoxel_leaf(subvoxel) {
                    res.dist = tv.x;
                    res.voxel_id = (parent << 3) | (dmask ^ idx);
                    res.termination = RaycastTermination::Hit;
                    res.material = subvoxel_material(subvoxel);
                    res.incidence = incidence;
                    res.hit = true;
                    return res;
                }
                // descend:
                if tc.y < h {
                    pstack[depth as usize] = parent;
                    tstack[depth as usize] = t.y;
                }
                depth += 1;

                h = tc.y;
                scale = scale >> 1;
                parent = subvoxel_child(subvoxel);
                // idx = select_child(pos, scale, o, d, tv.x);
                idx = select_child_bit(pos.cast().unwrap(), scale as f32, o, d, tv.x);
                t = tv;
                pos = child_cube(pos, scale, idx);

                continue;
            }
        }

        incidence = new_incidence;

        // advance
        t.x = tc.y;

        let incidence_mask = UVec3::new(
            (incidence == INCIDENCE_X) as u32,
            (incidence == INCIDENCE_Y) as u32,
            (incidence == INCIDENCE_Z) as u32
        );

        let p = pos + UVec3::from_value(scale);
        let bit_diff = highest_differing_bit(p, pos);
        pos += scale * incidence_mask;

        // bit_diff = p.x | p.y | p.z;

        let mask = 1 << incidence;
        idx ^= mask;

        // idx bits should only ever flip 0->1 because we force the ray direction to always be in the (1,1,1) quadrant
        if (idx & mask) == 0 {
            // ascend

            // highest differing bit
            // depth = ilog2(bit_diff);
            let idepth = bit_diff;

            // check if we exited voxel tree
            if idepth >= MAX_DAG_DEPTH {
                res.termination = RaycastTermination::Miss;
                res.hit = false;
                return res;
            }

            depth = MAX_DAG_DEPTH - idepth;

            scale = MAX_SCALE >> depth;
            // scale = 1 << (MAX_DAG_DEPTH - 1 - depth);

            parent = pstack[depth as usize];
            t.y = tstack[depth as usize];

            // round position to correct voxel (mask out low bits)
            pos.x &= 0xFFFFFFFF ^ (scale - 1);
            pos.y &= 0xFFFFFFFF ^ (scale - 1);
            pos.z &= 0xFFFFFFFF ^ (scale - 1);
            // pos = bitfieldInsert(pos, uvec3(0), 0, int(idepth));
            
            // get the idx of the child at the new depth
            // idx = extract_child_slot(pos, scale);
            idx = extract_child_slot_bfe(pos, idepth);

            h = 0.0;
        }

    }

    res.termination = RaycastTermination::LoopEnd;
    res.hit = false;
    return res;
}