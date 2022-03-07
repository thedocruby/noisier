mod noisier;

#[cfg(test)]
mod tests {
    extern crate rand;
    extern crate sorting_rs as sort;
    extern crate image;
    use noisier::*;
    use self::rand::prelude::*;
    use self::image::{Rgb, RgbImage};
    
    #[test]
    fn test_sort(){
        let mut vec = Vec::new();
        for i in 0..8 {
            vec.push((thread_rng().gen::<f32>(), i))
        }
        sort::heap_sort(&mut vec);
        for i in (0..8).rev() {println!("{:?}", vec[i])}
    }

    #[test]
    fn image_test(){

        const WIDTH: u32 = 256;
        const HEIGHT: u32 = 256;

        let machine = SimplexNoiseMachine::new();
        let buffer = machine.gen_recurrent(2, &vec![4.0; 2], &vec![0.0; 2], 1.0, 0.0);
        
        {
            let mut image: RgbImage = RgbImage::new(WIDTH, HEIGHT);
            let mut minimum: f64 = 1.0;
            let mut maximum: f64 = 0.0;
            for i in 0..WIDTH { for j in 0..HEIGHT {
                image.put_pixel(
                    i, j,
                    {
                        //let val = buffer.sample(&vec![((i + WIDTH/2) % WIDTH) as f64 / WIDTH as f64, ((j + HEIGHT/2) % HEIGHT) as f64 / HEIGHT as f64]);
                        let val = buffer.sample(&vec![i as f64 / WIDTH as f64, j as f64 / HEIGHT as f64]);
                        minimum = minimum.min(val as f64);
                        maximum = maximum.max(val as f64);
                        Rgb::from([
                            ((val + 1.0) * 127.5).round().max(0.0) as u8,
                            ((val + 1.0) * 127.5).round().max(0.0) as u8,
                            ((val + 1.0) * 127.5).round().max(0.0) as u8,
                        ])
                    },
                );
            } }
            println!("{:?}; {:?}; ", minimum, maximum);
            image.save("test.png").unwrap();
        }
    }
}