use std::time::{Instant, Duration};

pub struct Timing {
    current_sample : Instant,
    time_mean : f64,
    time_var : f64,
    hz_mean : f64,
    hz_var : f64,
    samples : f64,
}

impl Timing {
    pub fn new() -> Self {
        Self {
            current_sample : Instant::now(),
            time_mean : 0.0,
            time_var : 0.0,
            hz_mean : 0.0,
            hz_var : 0.0,
            samples : 0.0,
        }
    }

    pub fn start_sample(&mut self) {
        self.current_sample = Instant::now();
    }

    pub fn end_sample(&mut self) {
        let elapsed = self.current_sample.elapsed();
        let seconds = elapsed.as_secs_f64();
        let hz = 1.0 / seconds;

        self.time_mean += seconds;
        self.time_var += seconds * seconds;
        self.hz_mean += hz;
        self.hz_var += hz * hz;
        self.samples += 1.0;
    }

    pub fn stats(&self) -> (f64, f64, f64, f64) {
        let real_time_mean = self.time_mean / self.samples;
        let real_time_var = self.time_var / self.samples - real_time_mean * real_time_mean;

        let real_hz_mean = self.hz_mean / self.samples;
        let real_hz_var = self.hz_var / self.samples - real_hz_mean * real_hz_mean;

        (real_time_mean, real_time_var, real_hz_mean, real_hz_var)
    }
}