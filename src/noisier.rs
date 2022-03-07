extern crate sorting_rs as sort;
extern crate rand;
use self::rand::prelude::*;
use self::sort::*;
use std::{f64::consts::PI, fmt};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub trait NoiseMachine<'a> {
    fn new() -> Self;
    fn from_seed(seed: u64) -> Self;
    fn set_seed(&'a mut self, seed: u64);
    fn seed(&'a self) -> u64;
    fn sample_raw(&'a self, dim: usize, pos: &Vec<f64>) -> f64;
    fn sample(
        &'a self,
        dim: usize,
        pos: &Vec<f64>,
        scale: &Vec<f64>,
        offset: &Vec<f64>,
        weight: f64,
        bias: f64,
    ) -> f64 {
        let dim = dim.clamp(1, usize::BITS as usize);
        assert_eq!(pos.len(), dim);
        assert_eq!(scale.len(), dim);
        assert_eq!(offset.len(), dim);
        self.sample_raw(dim, &{
            let mut new_pos = Vec::with_capacity(dim);
            for i in 0..(dim) {
                new_pos.push(pos[i].clamp(0.0, 1.0) * scale[i].abs() + offset[i].abs());
            }
            new_pos
        }) * weight
            + bias
    }
    fn sample_tileable(
        &'a self,
        dim: usize,
        pos: &Vec<f64>,
        scale: &Vec<f64>,
        offset: &Vec<f64>,
        weight: f64,
        bias: f64,
    ) -> f64 {
        let dim2 = (dim * 2).clamp(1, usize::BITS as usize);
        let dim = dim2 / 2;
        assert_eq!(pos.len(), dim);
        assert_eq!(scale.len(), dim);
        assert_eq!(offset.len(), dim);
        self.sample_raw(dim2, &{
            let mut new_pos = Vec::with_capacity(dim2);
            for i in 0..(dim) {
                let pi_pos = pos[i].clamp(0.0, 1.0) * 2.0 * PI;
                new_pos.push((pi_pos.cos() + 1.0) / (2.0 * PI) * scale[i] + offset[i]);
            }
            for i in 0..(dim) {
                let pi_pos = pos[i].clamp(0.0, 1.0) * 2.0 * PI;
                new_pos.push((pi_pos.sin() + 1.0) / (2.0 * PI) * scale[i].abs() + offset[i].abs());
            }
            new_pos
        }) * weight
            + bias
    }
    fn gen_continuous(
        &'a self,
        dim: usize,
        scale: &Vec<f64>,
        offset: &Vec<f64>,
        weight: f64,
        bias: f64,
    ) -> ContinuousNoise<'a, Self> {
        ContinuousNoise::<'a, Self>::new(
            self,
            dim,
            scale.clone(),
            offset.clone(),
            weight,
            bias,
        )
    }
    fn gen_recurrent(
        &'a self,
        dim: usize,
        scale: &Vec<f64>,
        offset: &Vec<f64>,
        weight: f64,
        bias: f64,
    ) -> RecurrentNoise<'a, Self> {
        RecurrentNoise::<'a, Self>::new(
            self,
            dim,
            scale.clone(),
            offset.clone(),
            weight,
            bias,
        )
    }
}

#[derive(Clone)]
pub struct PerlinNoiseMachine where {
    seed: u64,
    randoms: [u64; usize::BITS as usize],
}
impl PerlinNoiseMachine {
    fn corner_dot(&self, corner: &Vec<u32>, sample: &Vec<f64>) -> f64 {
        let dim = sample.len();
        let grad = self.get_gradient(&corner);
        let mut dot_sum = 0.0;
        for i in 0..dim {
            dot_sum += (sample[i] - corner[i] as f64) * grad[i];
        }
        dot_sum
    }
    fn get_gradient(&self, corner: &Vec<u32>) -> Vec<f64> {
        let dim = corner.len();
        let mut angles: Vec<f64> = Vec::with_capacity(dim - 1);
        let mut vector = Vec::with_capacity(dim);
        for i in 0..dim - 1 {
            let mut s = DefaultHasher::new();
            self.randoms[i].hash(&mut s);
            corner.hash(&mut s);
            dim.hash(&mut s);
            let random = s.finish();
            angles.push((random as f64 / u64::MAX as f64) * 2.0 * PI);
            vector.push(1.0);
            for k in 0..i {
                vector[i] *= angles[k].sin();
            }
            vector[i] *= angles[i].cos();
        }
        vector.push(1.0);
        for i in 0..dim - 1 {
            vector[dim - 1] *= angles[i].sin();
        }
        vector
    }
}
impl <'a> NoiseMachine<'a> for PerlinNoiseMachine {
    fn new() -> Self {
        let seed = thread_rng().gen();
        let mut rng = StdRng::seed_from_u64(seed);
        let mut randoms = [0; usize::BITS as usize];
        rng.fill(&mut randoms);
        Self { seed, randoms }
    }
    fn from_seed(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut randoms = [0; usize::BITS as usize];
        rng.fill(&mut randoms);
        Self { seed, randoms }
    }
    fn set_seed(&'a mut self, seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut randoms = [0; usize::BITS as usize];
        rng.fill(&mut randoms);
        self.seed = seed;
        self.randoms = randoms;
    }
    fn seed(&'a self) -> u64 {
        self.seed
    }
    fn sample_raw(&'a self, dim: usize, pos: &Vec<f64>) -> f64 {
        assert_eq!(pos.len(), dim);
        let dim = dim.clamp(1, usize::BITS as usize);
        let pos: Vec<f64> = pos
            .iter()
            .map(|x| x.abs().clamp(0.0, (u32::MAX - usize::BITS) as f64))
            .collect();
        let mut pos0 = Vec::with_capacity(dim);
        let mut δ_pos = Vec::with_capacity(dim);
        for i in 0..dim {
            pos0.push(pos[i].floor() as u32);
            δ_pos.push(pos[i] - pos0[i] as f64);
        }
        let mut c = 2usize.pow(dim as u32);
        let mut out: Vec<f64> = Vec::with_capacity(c);
        for i in 0..2usize.pow(dim as u32) {
            let mut corner = Vec::with_capacity(dim);
            for j in 0..dim {
                corner.push(pos0[j] + ((i >> j) % 2) as u32);
            }
            out.push(self.corner_dot(&corner, &pos));
        }
        let int = |x, min, max| (max - min) * (x * (x * 6.0 - 15.0) + 10.0) * x * x * x + min;
        for i in 0..dim {
            c /= 2;
            let mut new_out = Vec::with_capacity(c);
            for j in 0..c {
                new_out.push(int(δ_pos[i], out[j * 2], out[j * 2 + 1]))
            }
            out.clone_from(&new_out);
        }
        (out[0] * 5.0 / 3.0).clamp(-1.0, 1.0)
    }
}
impl fmt::Debug for PerlinNoiseMachine {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "PerlinNoiseMachine{{seed: {}}}", self.seed)
    }
}

#[derive(Clone)]
pub struct SimplexNoiseMachine where {
    seed: u64,
    randoms: [u64; usize::BITS as usize],
}
impl SimplexNoiseMachine {
    fn get_gradient(&self, corner: &Vec<f64>) -> Vec<f64> {
        let dim = corner.len();
        let mut angles: Vec<f64> = Vec::with_capacity(dim - 1);
        let mut vector = Vec::with_capacity(dim);
        for i in 0..dim - 1 {
            let mut s = DefaultHasher::new();
            self.randoms[i].hash(&mut s);
            let corner: Vec<u32> = corner.iter().map(|float| float.clone() as u32).collect();
            corner.hash(&mut s);
            dim.hash(&mut s);
            let random = s.finish();
            angles.push((random as f64 / u64::MAX as f64) * 2.0 * PI);
            vector.push(1.0);
            for k in 0..i {
                vector[i] *= angles[k].sin();
            }
            vector[i] *= angles[i].cos();
        }
        vector.push(1.0);
        for i in 0..dim - 1 {
            vector[dim - 1] *= angles[i].sin();
        }
        vector
    }
}
impl <'a> NoiseMachine<'a> for SimplexNoiseMachine{
    fn new() -> Self {
        let seed = thread_rng().gen();
        let mut rng = StdRng::seed_from_u64(seed);
        let mut randoms = [0; usize::BITS as usize];
        rng.fill(&mut randoms);
        Self { seed, randoms }
    }
    fn from_seed(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut randoms = [0; usize::BITS as usize];
        rng.fill(&mut randoms);
        Self { seed, randoms }
    }
    fn set_seed(&'a mut self, seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut randoms = [0; usize::BITS as usize];
        rng.fill(&mut randoms);
        self.seed = seed;
        self.randoms = randoms;
    }
    fn seed(&'a self) -> u64 {
        self.seed
    }
    fn sample_raw(&'a self, dim: usize, pos: &Vec<f64>) -> f64 {
        let dim = dim.clamp(1, usize::BITS as usize);
        assert_eq!(pos.len(), dim);
        let mut pos: Vec<f64> = pos
            .iter()
            .map(|x| x.abs().clamp(0.0, (u32::MAX - usize::BITS) as f64))
            .collect();
        let in_pos = pos.clone();
        let mut pos0 = Vec::with_capacity(dim);
        let mut δ_pos = Vec::with_capacity(dim);
        {
            let f = ((dim as f64 + 1.0).sqrt() - 1.0) / dim as f64;
            let sum: f64 = pos.iter().sum();
            for i in 0..dim{
                pos[i] += sum * f;
                pos0.push(pos[i].floor() as u32);
                δ_pos.push((pos[i] - pos0[i] as f64, i));
            }
            heap_sort(&mut δ_pos);
        }
        let mut corners: Vec<Vec<f64>> = Vec::new();
        let mut grads: Vec<Vec<f64>> = Vec::new();
        {
            let mut corner: Vec<f64> = vec![0.0; dim];
            corners.push({
                let mut this_corner = corner.clone();
                for j in 0..dim {
                    this_corner[j] += pos0[j] as f64;
                }
                grads.push(self.get_gradient(&this_corner));
                this_corner
            });
            for i in (0..dim).rev() {
                corner[δ_pos[i].1] = 1.0;
                corners.push({
                    let mut this_corner = corner.clone();
                    for j in 0..dim {
                        this_corner[j] += pos0[j] as f64;
                    }
                    grads.push(self.get_gradient(&this_corner));
                    this_corner
                });
            }
            let g = (1.0 - 1.0/(dim as f64 + 1.0).sqrt()) / dim as f64;
            for corner in corners.iter_mut() {
                let sum: f64 = corner.iter().sum();
                for i in 0..dim {
                    corner[i] -= sum * g;
                }
            }
        }
        let mut d_squared = Vec::new();
        let mut dot_prod = Vec::new();
        {
            let mut δ_corners = Vec::new();
            for corner in corners.iter() {
                let mut δ_corner = Vec::new();
                for i in 0..dim {
                    δ_corner.push(in_pos[i] - corner[i]);
                }
                δ_corners.push(δ_corner);
            }
            for i in 0..(dim + 1) {
                let δ_corner = δ_corners[i].clone();
                let mut d_squared_sum: f64 = 0.0;
                let mut dot_prod_sum: f64 = 0.0;
                for j in 0..dim {
                    let coord = δ_corner[j];
                    d_squared_sum += coord * coord;
                    dot_prod_sum += coord * grads[i][j];
                }
                d_squared.push(d_squared_sum);
                dot_prod.push(dot_prod_sum);
            }
        }
        let mut output = Vec::new();
        {
            for i in 0..(dim + 1) {
                let c = 0.0f64.max(0.5 - d_squared[i]);
                output.push(c * c * c * c * dot_prod[i]);
            }
        }
        output.iter().sum::<f64>() * 100.0
    }
    fn sample_tileable(
        &'a self,
        dim: usize,
        pos: &Vec<f64>,
        scale: &Vec<f64>,
        offset: &Vec<f64>,
        weight: f64,
        bias: f64,
    ) -> f64 {
        let lerp = |x, a, b| (b - a) * x + a;
        let mut pow = 2usize.pow(dim as u32);
        let mut corners = Vec::with_capacity(pow);
        let mut vals = Vec::with_capacity(pow);
        for i in 0..pow {
            let mut corner = Vec::new();
            let mut this_pos = pos.clone();
            let this_scale: Vec<f64> = scale.iter().map(|x| 2.0 * x).collect();
            for j in 0..dim {
                corner.push((i >> j) % 2);
                this_pos[j] += corner[j] as f64;
                this_pos[j] /= 2.0;
            }
            corners.push(corner);
            vals.push(self.sample(dim, &this_pos, &this_scale, offset, weight, bias));
        }
        for i in 0..dim {
            pow /= 2;
            let mut new_vals = Vec::with_capacity(pow);
            for j in 0..pow {
                new_vals.push(lerp(pos[i], vals[j*2+1], vals[j*2]));
            }
            vals = new_vals;
        }
        vals[0]
    }
}
impl fmt::Debug for SimplexNoiseMachine {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SimplexNoiseMachine {{ seed: {} }}", self.seed)
    }
}

pub trait NoiseBuffer<'a, T> where T: NoiseMachine<'a> + ?Sized {
    fn sample(&'a self, pos: &Vec<f64>) -> f64;
    fn machine(&'a self) -> &'a T;
    fn dim(&'a self) -> usize;
    fn scale(&'a self) -> Vec<f64>;
    fn offset(&'a self) -> Vec<f64>;
    fn weight(&'a self) -> f64;
    fn bias(&'a self) -> f64;
    fn new(
        machine: &'a T,
        dim: usize,
        scale: Vec<f64>,
        offset: Vec<f64>,
        weight: f64,
        bias: f64,
    ) -> Self;
}
#[derive(Clone, Debug)]
pub struct ContinuousNoise<'a, T> where T: NoiseMachine<'a> + ?Sized {
    machine: &'a T,
    dim: usize,
    scale: Vec<f64>,
    offset: Vec<f64>,
    weight: f64,
    bias: f64,
}

impl<'a, T> NoiseBuffer<'a, T> for ContinuousNoise<'a, T> where T: NoiseMachine<'a> + ?Sized {
    fn sample(&'a self, pos: &Vec<f64>) -> f64 {
        self.machine.sample(
            self.dim,
            pos,
            &self.scale,
            &self.offset,
            self.weight,
            self.bias,
        )
    }
    fn machine(&'a self) -> &'a T {
        self.machine
    }
    fn dim(&'a self) -> usize {
        self.dim.clone()
    }
    fn scale(&'a self) -> Vec<f64> {
        self.scale.clone()
    }
    fn offset(&'a self) -> Vec<f64> {
        self.offset.clone()
    }
    fn weight(&'a self) -> f64 {
        self.weight.clone()
    }
    fn bias(&'a self) -> f64 {
        self.bias.clone()
    }
    fn new(
        machine: &'a T,
        dim: usize,
        scale: Vec<f64>,
        offset: Vec<f64>,
        weight: f64,
        bias: f64,
    ) -> Self {
        let dim = dim.clamp(1, usize::BITS as usize);
        assert_eq!(scale.len(), dim);
        assert_eq!(offset.len(), dim);
        Self {
            machine,
            dim,
            scale,
            offset,
            weight,
            bias,
        }
    }
}
#[derive(Clone, Debug)]
pub struct RecurrentNoise<'a, T> where T: NoiseMachine<'a> + ?Sized {
    machine: &'a T,
    dim: usize,
    scale: Vec<f64>,
    offset: Vec<f64>,
    weight: f64,
    bias: f64,
}

impl<'a, T> NoiseBuffer<'a, T> for RecurrentNoise<'a, T> where T: NoiseMachine<'a> + ?Sized {
    fn sample(&self, pos: &Vec<f64>) -> f64 {
        self.machine.sample_tileable(
            self.dim,
            pos,
            &self.scale,
            &self.offset,
            self.weight,
            self.bias,
        )
    }
    fn machine(&self) -> &'a T {
        self.machine
    }
    fn dim(&self) -> usize {
        self.dim.clone()
    }
    fn scale(&self) -> Vec<f64> {
        self.scale.clone()
    }
    fn offset(&self) -> Vec<f64> {
        self.offset.clone()
    }
    fn weight(&self) -> f64 {
        self.weight.clone()
    }
    fn bias(&self) -> f64 {
        self.bias.clone()
    }
    fn new(
        machine: &'a T,
        dim: usize,
        scale: Vec<f64>,
        offset: Vec<f64>,
        weight: f64,
        bias: f64,
    ) -> Self {
        let dim = (dim * 2).clamp(1, usize::BITS as usize) / 2;
        assert_eq!(scale.len(), dim);
        assert_eq!(offset.len(), dim);
        Self {
            machine,
            dim,
            scale,
            offset,
            weight,
            bias,
        }
    }
}
