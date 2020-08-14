use super::*;

impl VoxelChunk {
    
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
    pub fn translate_integral(&mut self, x : usize, y : usize, z : usize, size : usize) {
        let max_coord = {
            use std::cmp::max;
            max(x, max(y,z))
        };

        assert!(max_coord < size);

        let s = log_2(size - 1) + 1;

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
    pub fn translate_fractional(&mut self, _x : usize, _y : usize, _z : usize) {
        unimplemented!()
    }

    /// Writes the other voxel chunk into this one. Whether the other voxels overwrite or not is
    /// controlled by the `overwrite` parameter
    pub fn combine(&mut self, other : &VoxelChunk, overwrite : bool, recompress : bool) {
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

                if sv1 == 0 {
                    continue;
                }
                if sv0 == 0 {
                    s.voxels[i].sub_voxels[k] = sv1;
                    continue;
                }
                if overwrite {
                    if sv1 < 0 {
                        s.voxels[i].sub_voxels[k] = sv1;
                        continue;
                    }
                    if sv0 < 0 {
                        if sv1 > 0 {
                            let sv0 = s.subdivide_subvoxel(i, k);
                            recursive_combine(s, overwrite, sv0 - 1, sv1 as usize - 1);
                            continue;
                        } else {
                            s.voxels[i].sub_voxels[k] = sv1;
                        }
                    }
                } else {
                    if sv0 < 0 {
                        continue;
                    }
                    if sv1 < 0 {
                        let sv1 = s.subdivide_subvoxel(j, k);
                        let sv0 = s.duplicate_subvoxel(i, k).unwrap();
                        recursive_combine(s, overwrite, sv0 as usize - 1, sv1 as usize - 1);
                        continue;
                    }
                }
                if sv0 > 0 && sv1 > 0 {
                    let sv0 = s.duplicate_subvoxel(i, k).unwrap();
                    recursive_combine(s, overwrite, sv0 - 1, sv1 as usize - 1);
                    continue;
                }
            }
        }

        recursive_combine(self, overwrite, 0, n);

        if recompress {
            self.compress();
        }
    }
    
    /// Writes the other voxel chunk into this one. Whether the other voxels overwrite or not is
    /// controlled by the `overwrite` parameter
    pub fn subtract(&mut self, other : &VoxelChunk, recompress : bool) {
        let n = self.len();

        {
            let mut other_clone : VoxelChunk = other.clone();
            other_clone.shift_indexes(n);
            self.voxels.extend(other_clone.voxels);
        }

        fn recursive_subtract(s : &mut VoxelChunk, i : usize, j : usize) {
            for k in 0..8 {
                let sv0 = s.voxels[i].sub_voxels[k];
                let sv1 = s.voxels[j].sub_voxels[k];

                if sv1 == 0 {
                    continue;
                }
                if sv0 == 0 {
                    continue;
                }
                if sv1 < 0 {
                    s.voxels[i].sub_voxels[k] = 0;
                    continue;
                }
                if sv0 < 0 {
                    if sv1 > 0 {
                        let sv0 = s.subdivide_subvoxel(i, k);
                        recursive_subtract(s, sv0 - 1, sv1 as usize - 1);
                        continue;
                    } else {
                        s.voxels[i].sub_voxels[k] = 0;
                    }
                }
                if sv0 > 0 && sv1 > 0 {
                    let sv0 = s.duplicate_subvoxel(i, k).unwrap();
                    recursive_subtract(s, sv0 - 1, sv1 as usize - 1);
                    continue;
                }
            }
        }

        recursive_subtract(self, 0, n);

        if recompress {
            self.compress();
        }
    }
}