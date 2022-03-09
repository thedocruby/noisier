mod noisier;

#[cfg(test)]
mod tests {
    extern crate rand;
    extern crate sorting_rs as sort;
    extern crate image;
    use noisier::*;
    use std::{io, io::Write, f64::consts::PI};
    use self::rand::prelude::*;
    use self::image::{Rgb, RgbImage};

    #[test]
    fn image_test(){

        const WIDTH: u32 = 256;
        const HEIGHT: u32 = 256;

        let machine = PerlinNoiseMachine::<256>::new();
        let buffer = machine.gen_continuous(2, &vec![4.0; 2], &vec![0.0; 2], 1.0, 0.0);
        
        {
            let mut image: RgbImage = RgbImage::new(WIDTH, HEIGHT);
            for i in 0..WIDTH { for j in 0..HEIGHT {
                image.put_pixel(
                    i, j,
                    {
                        //let val = buffer.sample(&vec![((i + WIDTH/2) % WIDTH) as f64 / WIDTH as f64, ((j + HEIGHT/2) % HEIGHT) as f64 / HEIGHT as f64]);
                        let val = buffer.sample(&vec![i as f64 / WIDTH as f64, j as f64 / HEIGHT as f64]);
                        Rgb::from([
                            ((val + 1.0) * 127.5).round().max(0.0) as u8,
                            ((val + 1.0) * 127.5).round().max(0.0) as u8,
                            ((val + 1.0) * 127.5).round().max(0.0) as u8,
                        ])
                    },
                );
            } }
            image.save("test.png").unwrap();
        }
    }

    #[test]
    fn video_test(){

    ///////////////////////////////////////////////////////
        
        let f = |t: f64| ((t).cos().clamp(0.0, 1.0) + (t * 2.0).cos().clamp(0.0, 1.0)) / 2.0;
        let h = |t: f64| ((t + 2.0 * PI / 3.0).sin().clamp(0.0, 1.0) - ((t + 2.0 * PI / 3.0) * 2.0).cos().clamp(0.0, 1.0)) / -2.0;

        const WIDTH: u32 = 256;
        const HEIGHT: u32 = 256;
        const LENGTH: f64 = 360.0;
        const FRAMERATE: f64 = 30.0;

        const DIM: usize = 3;

        const R_SEED: u64 = 949546605679;
        const R_SCALE: [f64; DIM] = [6.0, 6.0, 120.0];
        const R_OFFSET: [f64; DIM] = [0.0, 0.0, 0.0];
        const R_WEIGHT: f64 = 1.0;
        const R_BIAS: f64 = 0.0;
        let r_func = |r: f64, g: f64, b: f64, t: f64| (
            (r * f((t * PI * 5.0) % PI                 ) * 5.0 / 3.0).clamp(0.0, 1.0) +
            (r * f((t * PI * 5.0) % PI -       PI      ) * 5.0 / 3.0).clamp(0.0, 1.0) +
            (r * h((t * PI * 5.0) % PI - 2.0 * PI / 3.0) * 5.0 / 3.0).clamp(0.0, 1.0)
        ) * 255.0;

        /* const G_SEED: u64 = 949546605679;
        const G_SCALE: [f64; DIM] = [4.0, 4.0, 30.0];
        const G_OFFSET: [f64; DIM] = [0.0, 0.0, 0.0];
        const G_WEIGHT: f64 = 1.0;
        const G_BIAS: f64 = 0.0; */
        let g_func = |r: f64, g: f64, b: f64, t: f64| (
            (r * f((t * PI * 5.0) % PI -       PI / 3.0) * 5.0 / 3.0).clamp(0.0, 1.0) +
            (r * h((t * PI * 5.0) % PI                 ) * 5.0 / 3.0).clamp(0.0, 1.0) +
            (r * h((t * PI * 5.0) % PI -       PI      ) * 5.0 / 3.0).clamp(0.0, 1.0)
        ) * 255.0;

        /* const B_SEED: u64 = 949546605679;
        const B_SCALE: [f64; DIM] = [4.0, 4.0, 30.0];
        const B_OFFSET: [f64; DIM] = [0.0, 0.0, 0.0];
        const B_WEIGHT: f64 = 1.0;
        const B_BIAS: f64 = 0.0; */
        let b_func = |r: f64, g: f64, b: f64, t: f64| (
            (r * f((t * PI * 5.0) % PI - 2.0 * PI / 3.0) * 5.0 / 3.0).clamp(0.0, 1.0) +
            (r * h((t * PI * 5.0) % PI -       PI / 3.0) * 5.0 / 3.0).clamp(0.0, 1.0) +
            (r * h((t * PI * 5.0) % PI - 4.0 * PI / 3.0) * 5.0 / 3.0).clamp(0.0, 1.0)
        ) * 255.0;
        
        const PROGRESS_BAR_LEN: usize = 64;

    ///////////////////////////////////////////////////////

        let r_machine = PerlinNoiseMachine::<256>::from_seed(R_SEED);
        //let g_machine = PerlinNoiseMachine::<256>::from_seed(G_SEED);
        //let b_machine = PerlinNoiseMachine::<256>::from_seed(B_SEED);

        let r_buffer = r_machine.gen_recurrent(DIM, &Vec::from(R_SCALE), &Vec::from(R_OFFSET), R_WEIGHT, R_BIAS);
        //let g_buffer = g_machine.gen_recurrent(DIM, &Vec::from(G_SCALE), &Vec::from(G_OFFSET), G_WEIGHT, G_BIAS);
        //let b_buffer = b_machine.gen_recurrent(DIM, &Vec::from(B_SCALE), &Vec::from(B_OFFSET), B_WEIGHT, B_BIAS);
        
        let frames = (LENGTH * FRAMERATE).floor() as usize;
        let digits = ((frames as f64).log10().floor() + 1.0) as usize;
        let mut update: Box<dyn Fn(usize)> = Box::new(|frame| ());

        match digits {
            1 => update = Box::new(|frame| print!("\nRendering frame {:0>1} of {:0>1}... ", frame, frames)),
            2 => update = Box::new(|frame| print!("\nRendering frame {:0>2} of {:0>2}... ", frame, frames)),
            3 => update = Box::new(|frame| print!("\nRendering frame {:0>3} of {:0>3}... ", frame, frames)),
            4 => update = Box::new(|frame| print!("\nRendering frame {:0>4} of {:0>4}... ", frame, frames)),
            5 => update = Box::new(|frame| print!("\nRendering frame {:0>5} of {:0>5}... ", frame, frames)),
            6 => update = Box::new(|frame| print!("\nRendering frame {:0>6} of {:0>6}... ", frame, frames)),
            7 => update = Box::new(|frame| print!("\nRendering frame {:0>7} of {:0>7}... ", frame, frames)),
            8 => update = Box::new(|frame| print!("\nRendering frame {:0>8} of {:0>8}... ", frame, frames)),
            _ => update = Box::new(|frame| print!("\nRendering frame {} of {}... ", frame, frames)),
        }

        println!("Starting now...");
        for frame in 0..frames {
            let mut image: RgbImage = RgbImage::new(WIDTH, HEIGHT);
            let mut progress: usize = 0;
            update(frame + 1);
            io::stdout().flush().unwrap();
            let mut r = 0.0;
            let mut g = 0.0;
            let mut b = 0.0;
            for i in 0..WIDTH { for j in 0..HEIGHT {
                if (progress as f64 + 1.0) < (((j+i*HEIGHT) as f64)/((WIDTH*HEIGHT) as f64) * (PROGRESS_BAR_LEN as f64)) {
                    progress += 1;
                    print!("█");
                    io::stdout().flush().unwrap();
                }
                let t = frame as f64 / frames as f64;
                image.put_pixel(
                    i, j,
                    {
                        r = r_buffer.sample(&vec![i as f64 / WIDTH as f64, j as f64 / HEIGHT as f64, frame as f64 / frames as f64]);
                        // g = g_buffer.sample(&vec![i as f64 / WIDTH as f64, j as f64 / HEIGHT as f64, frame as f64 / frames as f64]);
                        // b = b_buffer.sample(&vec![i as f64 / WIDTH as f64, j as f64 / HEIGHT as f64, frame as f64 / frames as f64]);

                        Rgb::from([
                            r_func(r, g, b, t).round() as u8,
                            g_func(r, g, b, t).round() as u8,
                            b_func(r, g, b, t).round() as u8,
                        ])
                    },
                );
            } }
            image.save(format!("video/test{:0>8}.png", frame)).unwrap();
        }
    }
}