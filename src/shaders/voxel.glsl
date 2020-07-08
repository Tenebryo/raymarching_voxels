
// this seems like a pretty hacky solution to getting access to voxel data when included
#ifndef VOXEL_BINDING_OFFSET
#define VOXEL_BINDING_OFFSET (0)
#endif
/// datastructures and methods for processing voxel data

#ifndef MAX_DAG_DEPTH
#define MAX_DAG_DEPTH (16)
#endif


#define AXIS_X_MASK 1
#define AXIS_Y_MASK 2
#define AXIS_Z_MASK 4

#define INCIDENCE_X 0
#define INCIDENCE_Y 1
#define INCIDENCE_Z 2

#define CHILD_OFFSET_BITS 4
#define CHILD_OFFSET_MASK 3

struct VChildDescriptor {
    // if a sub-DAG: positive 1-index pointers to children
    // if a leaf: negative voxel material id
    // if empty: 0
    int sub_voxels[8];
};

layout(binding = VOXEL_BINDING_OFFSET) buffer VoxelChildData {
    VChildDescriptor voxels[];
};
layout(binding = VOXEL_BINDING_OFFSET + 1) buffer VoxelMaterialData {
    uint lod_materials[];
};

vec2 project_cube(vec3 id, vec3 od, vec3 mn, vec3 mx, out uint incidence_min, out uint incidence_max) {

    vec3 tmn = fma(id, mn, od);
    vec3 tmx = fma(id, mx, od);

    float ts;
    if (tmn.x > tmn.y) {
        if (tmn.x > tmn.z) {
            incidence_min = INCIDENCE_X;
            ts = tmn.x;
        } else {
            incidence_min = INCIDENCE_Z;
            ts = tmn.z;
        }
    } else {
        if (tmn.y > tmn.z) {
            incidence_min = INCIDENCE_Y;
            ts = tmn.y;
        } else {
            incidence_min = INCIDENCE_Z;
            ts = tmn.z;
        }
    }

    float te;
    if (tmx.x < tmx.y) {
        if (tmx.x < tmx.z) {
            incidence_max = INCIDENCE_X;
            te = tmx.x;
        } else {
            incidence_max = INCIDENCE_Z;
            te = tmx.z;
        }
    } else {
        if (tmx.y < tmx.z) {
            incidence_max = INCIDENCE_Y;
            te = tmx.y;
        } else {
            incidence_max = INCIDENCE_Z;
            te = tmx.z;
        }
    }

    return vec2(ts, te);
}

bool voxel_valid_bit(uint parent, uint idx) {
    return voxels[parent].sub_voxels[idx] != 0;
}

bool voxel_leaf_bit(uint parent, uint idx) {
    return voxels[parent].sub_voxels[idx] < 0;
}

bool voxel_empty(uint parent, uint idx) {
    return voxels[parent].sub_voxels[idx] == 0;
}

uint voxel_get_child(uint parent, uint idx) {
    return voxels[parent].sub_voxels[idx] - 1;
}

uint voxel_get_material(uint parent, uint idx) {
    return -voxels[parent].sub_voxels[idx];
}

bool interval_nonempty(vec2 t) {
    return t.x < t.y;
}
vec2 interval_intersect(vec2 a, vec2 b) {
    return ((b.x > a.y || a.x > b.y) ? vec2(1,0) : vec2(max(a.x,b.x), min(a.y, b.y)));
}

uint select_child(vec3 pos, float scale, vec3 o, vec3 d, float t) {
    vec3 p = o + d * t - pos - scale;
    uint idx = 0;

    idx |= p.x < 0 ? 0 : AXIS_X_MASK;
    idx |= p.y < 0 ? 0 : AXIS_Y_MASK;
    idx |= p.z < 0 ? 0 : AXIS_Z_MASK;

    return idx;
}

uvec3 child_cube( uvec3 pos, uint scale, uint idx) {
    const uvec3 child_pos_offsets[8] = {
        uvec3(0,0,0),
        uvec3(1,0,0),
        uvec3(0,1,0),
        uvec3(1,1,0),
        uvec3(0,0,1),
        uvec3(1,0,1),
        uvec3(0,1,1),
        uvec3(1,1,1)
    };

    return pos + (scale * child_pos_offsets[idx]);
}

uint ilog2(uint v) {
#define LT(n) n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n
    const uint LogTable256[256] = {
        -1, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
        LT(4), LT(5), LT(5), LT(6), LT(6), LT(6), LT(6),
        LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7)
    };

    uint r;     // r will be lg(v)
    uint t, tt; // temporaries
    tt = v >> 16;
    if (tt != 0)
    {
        t = tt >> 8;
        r = t != 0 ? 24 + LogTable256[t] : 16 + LogTable256[tt];
    }
    else 
    {
        t = v >> 8;
        r = t != 0 ? 8 + LogTable256[t] : LogTable256[v];
    }

    return r;
}

uint highest_differing_bit(uvec3 a, uvec3 b) {
    uvec3 t = a ^ b;

    return ilog2(t.x | t.y | t.z);
}

uint extract_child_slot(uvec3 pos, uint scale) {
    uvec3 d = pos & scale;

    uint idx = 0;
    
    idx |= (d.x == 0) ? 0 : AXIS_X_MASK;
    idx |= (d.y == 0) ? 0 : AXIS_Y_MASK;
    idx |= (d.z == 0) ? 0 : AXIS_Z_MASK;

    return idx;
}

#define VOXEL_MARCH_MISS 0
#define VOXEL_MARCH_HIT 1
#define VOXEL_MARCH_MAX_DEPTH 2
#define VOXEL_MARCH_LOD 3
#define VOXEL_MARCH_MAX_DIST 4

bool voxel_march(vec3 o, vec3 d, uint max_depth, float max_dist, out float dist, out uint incidence, out uint vid, out uint material, out uint return_state, out uint iterations) {

    const uint MAX_SCALE = (1<<MAX_DAG_DEPTH);


    const ivec3 incidence_axis[3] = {
        ivec3(1,0,0),
        ivec3(0,1,0),
        ivec3(0,0,1)
    };
    const uint incidence_mask[3] = {
        AXIS_X_MASK,
        AXIS_Y_MASK,
        AXIS_Z_MASK
    };

    uint pstack[MAX_DAG_DEPTH];
    float tstack[MAX_DAG_DEPTH];
    uint dmask = 0;

    vec3 ds = sign(d);

    d *= ds;
    o = o * ds + (1 - ds) / 2;

    o *= MAX_SCALE;

    dmask |= ds.x < 0 ? AXIS_X_MASK : 0;
    dmask |= ds.y < 0 ? AXIS_Y_MASK : 0;
    dmask |= ds.z < 0 ? AXIS_Z_MASK : 0;

    float min_size = 0.00001;

    vec3 id = 1.0 / d;
    vec3 od = - o * id;

    vec2 t = vec2(0,MAX_SCALE * 100);

    float h = t.y;

    // fix initial position
    uvec3 pos = ivec3(0);
    uvec3 old_pos = pos;

    uint parent = 0;
    uint idx = 0;

    uint scale = 1 << MAX_DAG_DEPTH;
    uint depth = 1;

    uint incidence_min;
    t = interval_intersect(t, project_cube(id, od, pos, pos + scale, incidence_min, incidence));

    if (!interval_nonempty(t)) {
        // we didn't hit the bounding cube
        return false;
    }

    idx = select_child(pos, scale, o, d, t.x);

    scale >>= 1;

    iterations = 0;

    return_state = VOXEL_MARCH_MISS;

    pstack[0] = parent;
    tstack[0] = t.y;


    // very hot loop
    while (iterations < 512) {
        iterations += 1;

        vec2 tc = project_cube(id, od, pos, pos + scale, incidence_min, incidence);

        if (voxel_valid_bit(parent, dmask ^ idx) && interval_nonempty(t)) {

            if (scale <= tc.x * 0.005 || depth >= max_depth) {
                // voxel is too small
                dist = t.x / MAX_SCALE;
                return_state = depth >= max_depth ? VOXEL_MARCH_MAX_DEPTH : VOXEL_MARCH_LOD;
                material = lod_materials[parent];
                return true;
            }

            if (tc.x > max_dist * MAX_SCALE) {
                // voxel is beyond the render distance
                return_state = VOXEL_MARCH_MAX_DIST;
                return false;
            }

            vec2 tv = interval_intersect(tc, t);

            if (interval_nonempty(tv)) {
                if (voxel_leaf_bit(parent, dmask ^ idx)) {
                    dist = tv.x / MAX_SCALE;
                    vid = (parent << 3) | (dmask ^ idx);
                    return_state = VOXEL_MARCH_HIT;
                    material = voxel_get_material(parent, dmask ^ idx);
                    return true;
                }
                // descend:
                if (tc.y < h) {
                    pstack[depth] = parent;
                    tstack[depth] = t.y;
                }
                depth += 1;

                h = tc.y;
                scale = scale >> 1;
                parent = voxel_get_child(parent, dmask ^ idx);
                idx = select_child(pos, scale, o, d, tv.x);
                t = tv;
                pos = child_cube(pos, scale, idx);

                continue;
            }
        }

        // advance
        t.x = tc.y;

        uint mask = 0;
        uint bit_diff = 0;
        if (incidence == INCIDENCE_X) {
            uint px = pos.x;
            pos.x += scale;
            bit_diff = px ^ pos.x;
            mask = AXIS_X_MASK;
        } else if (incidence == INCIDENCE_Y) {
            uint py = pos.y;
            pos.y += scale;
            bit_diff = py ^ pos.y;
            mask = AXIS_Y_MASK;
        } else {
            uint pz = pos.z;
            pos.z += scale;
            bit_diff = pz ^ pos.z;
            mask = AXIS_Z_MASK;
        }

        idx ^= mask;


        // idx bits should only ever flip 0->1 because we force the ray direction to always be in the (1,1,1) quadrant
        if ((idx & mask) == 0) {
            // ascend

            // highest differing bit
            depth = ilog2(bit_diff);

            // check if we exited voxel tree
            if (depth >= MAX_DAG_DEPTH) {
                return_state = VOXEL_MARCH_MISS;
                return false;
            }

            depth = MAX_DAG_DEPTH - depth;

            scale = MAX_SCALE >> depth;
            // scale = 1 << (MAX_DAG_DEPTH - 1 - depth);

            parent = pstack[depth];
            t.y = tstack[depth];

            // round position to correct voxel (mask out low bits)
            pos &= 0xFFFFFFFF ^ (scale - 1);
            
            // get the idx of the child at the new depth
            idx = extract_child_slot(pos, scale);

            h = 0;
        }

    }

    return false;
}