use std::time::Instant;

pub struct Timing {
    current_sample : Instant,
    time_samples : Vec<f64>,
    time_mean : f64,
    time_var : f64,
    hz_samples : Vec<f64>,
    hz_mean : f64,
    hz_var : f64,
    samples : f64,
    max_samples : usize,
    i : usize,
}

impl Timing {
    pub fn new(max_samples : usize) -> Self {
        Self {
            current_sample : Instant::now(),
            time_samples : vec![],
            time_mean : 0.0,
            time_var : 0.0,
            hz_samples : vec![],
            hz_mean : 0.0,
            hz_var : 0.0,
            samples : 0.0,
            max_samples,
            i : 0,
        }
    }

    pub fn start_sample(&mut self) {
        self.current_sample = Instant::now();
    }

    pub fn end_sample(&mut self) {
        let elapsed = self.current_sample.elapsed();
        let time = elapsed.as_secs_f64();
        let hz = 1.0 / time;

        self.time_mean += time;
        self.time_var += time * time;
        self.hz_mean += hz;
        self.hz_var += hz * hz;

        if self.samples < self.max_samples as f64 {
            self.time_samples.push(time);
            self.hz_samples.push(hz);
            self.samples += 1.0;
        } else {
            let ptime = self.time_samples[self.i];
            let phz = self.hz_samples[self.i];

            self.time_samples[self.i] = time;
            self.hz_samples[self.i] = hz;

            self.i = (self.i + 1) % self.max_samples;

            self.time_mean -= ptime;
            self.time_var -= ptime * ptime;
            self.hz_mean -= phz;
            self.hz_var -= phz * phz;
        }
    }

    pub fn stats(&self) -> (f64, f64, f64, f64) {
        let real_time_mean = self.time_mean / self.samples;
        let real_time_var = self.time_var / self.samples - real_time_mean * real_time_mean;

        let real_hz_mean = self.hz_mean / self.samples;
        let real_hz_var = self.hz_var / self.samples - real_hz_mean * real_hz_mean;

        (real_time_mean, real_time_var, real_hz_mean, real_hz_var)
    }
}