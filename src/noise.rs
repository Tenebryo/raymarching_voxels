use rand::prelude::*;

pub struct WorleyNoise3D {
    dim: usize,
    points: Vec<[f32; 3]>,
}

impl WorleyNoise3D {
    pub fn new(dim : usize) -> Self {
        let points = 
            (0..dim).map(move |z| 
                (0..dim).map(move |y|
                    (0..dim).map(move |x| {
                        let mut rng = thread_rng();
                        [
                            (rng.gen_range(0.0, 1.0) + x as f32) / dim as f32,
                            (rng.gen_range(0.0, 1.0) + y as f32) / dim as f32,
                            (rng.gen_range(0.0, 1.0) + z as f32) / dim as f32
                        ]
                    })
                )
            )
            .flatten()
            .flatten()
            .collect::<Vec<[f32;3]>>();

        Self {
            dim,
            points
        }
    }

    pub fn sample(&self, x : f32, y : f32, z : f32) -> f32 {
        let x = x % 1.0;
        let y = y % 1.0;
        let z = z % 1.0;

        let d = self.dim;

        let xi = (x * d as f32).floor() as isize;
        let yi = (y * d as f32).floor() as isize;
        let zi = (z * d as f32).floor() as isize;

        // println!("test: {:.3} {:.3} {:.3}", xi, yi, zi);

        // max dist is (very loosely) bounded by 1.0;
        // closer bound is bounded by 1 / dim;
        let mut max_dist = 1.0;

        let d = d as isize;

        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let mut xj = xi + dx;
                    let mut yj = yi + dy;
                    let mut zj = zi + dz;

                    let mut px = x;
                    let mut py = y;
                    let mut pz = z;

                    if xj < 0 { xj = d - 1; px += 1.0; } else if xj >= d { xj = 0; px -= 1.0; }
                    if yj < 0 { yj = d - 1; py += 1.0; } else if yj >= d { yj = 0; py -= 1.0; }
                    if zj < 0 { zj = d - 1; pz += 1.0; } else if zj >= d { zj = 0; pz -= 1.0; }

                    let i = (xj + d * (yj + d * zj)) as usize;

                    let ddx = self.points[i][0] - px;
                    let ddy = self.points[i][1] - py;
                    let ddz = self.points[i][2] - pz;

                    let dist = (ddx * ddx + ddy * ddy + ddz * ddz).sqrt();

                    if dist < max_dist {
                        max_dist = dist;
                    }
                }
            }
        }

        // d as f32 * max_dist
        max_dist
    }
}