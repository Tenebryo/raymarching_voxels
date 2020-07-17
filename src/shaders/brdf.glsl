
#ifndef BRDF_BINDING_OFFSET
#define BRDF_BINDING_OFFSET (0)
#endif

/// BRDF 
struct BRDF {
    /// resolution of brdf
    uvec4 n;
    /// factor terms
    uint l;

    // pdf and cdf indices
    /// index of the 2*n*n*l outgoing ray matrix
    uint f_idx;
    /// index of the 2*l*n incoming ray matrix theta factor
    uint u_idx;
    /// index of the 2*l*n incoming ray matrix phi factor
    uint v_idx;
};

layout(binding = BRDF_BINDING_OFFSET) readonly buffer BRDFMetaData {
    BRDF brdfs[];
};
layout(binding = BRDF_BINDING_OFFSET+1) readonly buffer BRDFData {
    float brdf_values[];
};

// brdf indexing functions
uint brdf_index_f_pdf(BRDF brdf, uint ti, uint pi, uint li) {
    uint n = brdf.n.y;
    uint l = brdf.l;
    return brdf.f_idx + li + l * (pi + n * ti);
}

uint brdf_index_f_cdf(BRDF brdf, uint ti, uint pi, uint li) {
    uint n = brdf.n.y;
    uint l = brdf.l;
    return brdf.f_idx + li + l * (pi + n * (ti + n));
}

uint brdf_index_u_pdf(BRDF brdf, uint ti, uint li) {
    uint n = brdf.n.z;
    uint l = brdf.l;
    return brdf.u_idx + ti + n * li;
}

uint brdf_index_u_cdf(BRDF brdf, uint ti, uint li) {
    uint n = brdf.n.z;
    uint l = brdf.l;
    return brdf.u_idx + ti + n * (li + l);
}

uint brdf_index_v_pdf(BRDF brdf, uint pi, uint li) {
    uint n = brdf.n.w;
    uint l = brdf.l;
    return brdf.v_idx + pi + n * li;
}

uint brdf_index_v_cdf(BRDF brdf, uint pi, uint li) {
    uint n = brdf.n.w;
    uint l = brdf.l;
    return brdf.v_idx + pi + n * (li + l);
}

// brdf value functions (gets the value of the BDRF factors at certain points)
float brdf_value_f_pdf(BRDF brdf, uint ti, uint pi, uint li) {
    return brdf_values[brdf_index_f_pdf(brdf, ti, pi, li)];
}

float brdf_value_f_cdf(BRDF brdf, uint ti, uint pi, uint li) {
    return brdf_values[brdf_index_f_cdf(brdf, ti, pi, li)];
}

float brdf_value_u_pdf(BRDF brdf, uint ti, uint li) {
    return brdf_values[brdf_index_u_pdf(brdf, ti, li)];
}

float brdf_value_u_cdf(BRDF brdf, uint ti, uint li) {
    return brdf_values[brdf_index_u_cdf(brdf, ti, li)];
}

float brdf_value_v_pdf(BRDF brdf, uint pi, uint li) {
    return brdf_values[brdf_index_v_pdf(brdf, pi, li)];
}

float brdf_value_v_cdf(BRDF brdf, uint pi, uint li) {
    return brdf_values[brdf_index_v_cdf(brdf, pi, li)];
}

/// get the density of the BRDF with the outbound and parameterized (half angle) incoming ray
float brdf_pdf_p(BRDF brdf, float theta_o, float phi_o, float theta_p, float phi_p) {

    int theta_o_i = int(theta_o / PI * brdf.n);
    int theta_p_i = int(theta_p / PI * brdf.n);
    
    int phi_o_i = int(phi_o / PI * 0.5 * brdf.n);
    int phi_p_i = int(phi_p / PI * 0.5 * brdf.n);

    float nsum = 0;
    float dsum = 0;

    for (uint li = 0; li < brdf.l; li++) {
        float fv = brdf_value_f_pdf(brdf, theta_o_i, phi_o_i, li);
        float uv = brdf_value_u_pdf(brdf, theta_p_i, li);
        float vv = brdf_value_u_pdf(brdf, phi_p_i, li);

        nsum += fv * uv * vv;
        dsum += fv;
    }

    return nsum / dsum;
}

/// calculate the incidence angle from factorization index.
///  * `theta` : the `theta` parameter
///  * `phi` : the `phi` parameter
///  * `theta_i` : out parameter for the `u` factor index
///  * `phi_i` : out parameter for the `v` factor index
void brdf_angle_to_index(BRDF brdf, float theta, float phi, out uint theta_i, out uint phi_i) {
    theta_i = int(theta / PI * brdf.n);
    phi_i = int(phi / PI * 0.5 * brdf.n);
}

/// calculate the half-angle parameterization from factorization index.
///  * `z` : the `z` parameter
///  * `phi` : the `phi` parameter
///  * `zi` : out parameter for the `u` factor index
///  * `phi_i` : out parameter for the `v` factor index
void brdf_half_angle_to_index(BRDF brdf, float z, float phi, out uint zi, out uint phi_i) {
    zi = int(brdf.n * (z + 1) * 0.5);
    phi_i = int(phi / PI * 0.5 * brdf.n);
}

/// calculate the incidence angle from factorization index.
///  * `theta_i` : the `u` factor index
///  * `phi_i` : the `v` factor index
///  * `theta` : out parameter for the `theta` parameter
///  * `phi` : out parameter for the `phi` parameter
void brdf_index_to_angle(BRDF brdf, uint theta_i, uint phi_i, out float theta, out float phi) {
    theta = PI * theta_i / float(brdf.n);
    phi = 2.0 * PI * phi_i / float(brdf.n);
}

/// calculate the half-angle parameterization from factorization index.
///  * `zi` : the `u` factor index
///  * `phi_i` : the `v` factor index
///  * `z` : out parameter for the `z` parameter
///  * `phi` : out parameter for the `phi` parameter
void brdf_index_to_half_angle(BRDF brdf, uint zi, uint phi_i, out float z, out float phi) {
    z = 2.0 * zi / float(brdf.n) - 1;
    phi = 2.0 * PI * phi_i / float(brdf.n);
}


/// Sample a vector from the BRDF conditioned on the outbound ray defined by azimuthal and elevation angles
/// and 3 random numbers in the range [0,1)
/// An incoming ray is sampled and returned in the same format vec2(theta, phi);
vec2 sample_brdf(BRDF brdf, float theta_o, float phi_o, float r0, float r1, float r2) {
    // TODO: replace linear search with binary search. this is currently not a huge bottleneck because raymarching is much slower
    uint fi, ui, vi;

    uint theta_i;
    uint phi_i;

    brdf_angle_to_index(brdf, theta_o, phi_o, theta_i, phi_i);

    // use first random number to select the factor index:
    for(fi = 0; fi < brdf.l; fi++) {
        float fv = brdf_value_f_cdf(brdf, theta_i, phi_i, fi);
        if (r0 < fv) {
            break;
        }
    }

    // use the second and third random numbers to select the ray direction
    for(ui = 0; ui < brdf.n.z; ui++) {
        float uv = brdf_value_u_cdf(brdf, ui, fi);
        if (r1 < uv) {
            break;
        }
    }
    
    for(vi = 0; vi < brdf.n.w; vi++) {
        float vv = brdf_value_v_cdf(brdf, vi, fi);
        if (r2 < vv) {
            break;
        }
    }
    

    // convert the sampled indices to a ray vector 
    float z, phi;
    brdf_index_to_half_angle(brdf, ui, vi, z, phi);

    float theta = acos(z);

    return vec2(theta, phi);
}