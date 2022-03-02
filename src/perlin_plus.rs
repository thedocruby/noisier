use rand::prelude::*; use std::f64::consts::PI; #[derive(Clone)]
pub struct NoiseMachine { randoms: [usize; usize::BITS as usize], } impl NoiseMachine {
    pub fn new(seed: u64) -> Self { let rng=StdRng::seed_from_u64(seed); let randoms; rng.fill(&mut randoms); Self{randoms}
    } fn corner_dot(&self, corner_pos: Vec<i64>, sample_pos: Vec<f64>) -> f64{
    } fn get_gradient(&self, corner: Vec<i64>) -> Vec<f64> { let dim = corner.len(); if dim == 1
        { return vec!(self.randoms[0].overflowing_pow(corner[0] as u32).0 as f64/usize::MAX as f64*2.0-1.0) }
        let randoms = Vec::with_capacity(dim-1); let angles = Vec::with_capacity(dim-1); let vector = Vec::with_capacity(dim);
        for i in 0..dim-1{randoms[i]=self.randoms[i];for j in 0..dim{randoms[i]=randoms[i].overflowing_pow(corner[j]as u32).0}
            angles[i] = randoms[i] as f64 / usize::MAX as f64 * 2.0 * PI; vector[i] = 1.0; 
            for k in 0..i { vector[i] *= angles[k].sin() } vector[i] *= angles[i].cos()
        } vector[dim-1] = 1.0; for i in 0..dim-1 { vector[dim-1] *= angles[i].sin() } vector
    } fn gen_noise(&self, pos: Vec<f64>, dim: usize) -> f64 { let mut pos0 = Vec::with_capacity(dim);
        let mut spos = Vec::with_capacity(dim); for i in 0..dim { pos0[i] = pos[i].floor() as i64; 
            spos[i] = pos[i] - pos0[i] as f64; } let mut out: Vec<f64> = Vec::with_capacity(2usize.pow(dim as u32));
        for i in 0..2usize.pow(dim as u32) { let mut corner = Vec::with_capacity(dim); for j in 0..dim
            { corner[j] = pos0[j] + ((i >> j) % 2) as i64; } out[i] = self.corner_dot(corner, pos); }
        let int = |x, max, min| (max - min) * (x * (x * 6.0 - 15.0) + 10.0) * x * x * x + min;
        for i in 0..dim { let mut new_out = Vec::with_capacity(out.len()/2); for j in 0..out.len()/2 
            { new_out[j] = int(spos[i], out[j*2], out[j*2 + 1]) }  out = new_out; } out[0]
    } pub fn get(&self, dim: usize, pos: Vec<f64>, scale: Vec<f64>, offset: Vec<f64>, weight: f64, bias: f64) -> f64 {
        dim = dim.clamp(1, usize::BITS as usize); self.gen_noise({ let new_pos = Vec::with_capacity(dim);
        for i in 0..(dim) { new_pos.push(pos[i].clamp(0.0, 1.0) * scale[i] + offset[i]); } new_pos }, dim) * weight + bias
    } pub fn get_tileable(&self, dim:usize, pos:Vec<f64>, scale:Vec<f64>, offset:Vec<f64>, weight:f64, bias:f64) -> f64 {
        dim = (dim*2).clamp(1, usize::BITS as usize); self.gen_noise({ let new_pos = Vec::with_capacity(dim);
        for i in 0..(dim/2) { let pi_pos = pos[i].clamp(0.0,1.0)*2.0*PI; new_pos.push(pi_pos.cos()*scale[i]+offset[i]); 
            new_pos.push(pi_pos.sin() * scale[i] + offset[i]); } new_pos }, dim) * weight + bias
    } pub fn get_raw(&self, dim: usize, pos: Vec<f64>) -> f64 { self.gen_noise(pos, dim.clamp(1, usize::BITS as usize)) }
}
