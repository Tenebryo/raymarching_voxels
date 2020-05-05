pub mod cs {
    vulkano_shaders::shader!{
        ty: "compute",
        src: "
            #version 450
            layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;
            layout(rgba32f, binding = 0) uniform image2D image;

            struct VoxelChunk {
                uint occ[8192];
            };

            layout(binding = 1) buffer VoxelOcc {
                VoxelChunk chunks[];
            } vox;

            layout(push_constant) uniform PushConstantData {
                vec3 cam_o;
                vec3 cam_f;
                vec3 cam_u;
                ivec3 vox_dim;
            } pc;

            bool check_voxel(int x, int y, int z) {
                

                return false;
            }

            void raymarch(vec3 o, vec3 d) {
                float mx, my, mz, dx, dy, dz, t, dsx, dsy, dsz, ox, oy, oz;
                int sx, sy, sz, hsx, hsy, hsz, x, y, z;
                t = 0;
                
                sx = int(sign(d.x));
                sy = int(sign(d.y));
                sz = int(sign(d.z));
                
                x = int(floor(o.x));
                y = int(floor(o.y));
                z = int(floor(o.z));
                
                dx = 1.0 / abs(d.x);
                dy = 1.0 / abs(d.y);
                dz = 1.0 / abs(d.z);
                
                hsx = (1 + sx) / 2;
                hsy = (1 + sy) / 2;
                hsz = (1 + sz) / 2;
                
                mx = abs(x + hsx - o.x) * dx;
                my = abs(y + hsy - o.y) * dy;
                mz = abs(z + hsz - o.z) * dz;
                
                int n = 64;
                
                while (n >= 0) {
                  
                  if (mx < my) {
                    if (mx < mz) {
                      x += sx;
                      mx += dx;
                    } else {
                      z += sz;
                      mz += dz;
                    }
                  } else {
                    if (my < mz) {
                      y += sy;
                      my += dy;
                    } else {
                      z += sz;
                      mz += dz;
                    }
                  }

                  if (check_voxel(x,y,z)) {
                    break;
                  }

                  n -= 1;
                }
            }

            void main() {
                uint idxX = gl_GlobalInvocationID.x;
                uint idxY = gl_GlobalInvocationID.y;

                ivec2 size = imageSize(image);

                if (idxX < size.x && idxY < size.y) {
                    imageStore(image, ivec2(idxX,idxY), vec4(idxX / float(size.x), idxY / float(size.y), 0.0, 1.0));
                }
            }
        "
    }
}

pub use cs::ty::PushConstantData;
pub use cs::ty::VoxelChunk;