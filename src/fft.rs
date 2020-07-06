// fftw wrapper
#![allow(unused)]

use ndarray::Array;
use fftw::types::*;
use fftw::plan::*;

// Cannot compile here
// pub fn fftn<D>(input: &Array<c64, D>) -> &Array<c64, D> {
//     let mut input = input.clone();
//     let mut output = input.clone();
//     C2CPlan64::aligned(input.shape(), Sign::Forward, Flag::MEASURE)
//         .unwrap()
//         .c2c(input.as_slice_mut().unwrap(), output.as_slice_mut().unwrap())
//         .unwrap();
//     output
// }

#[macro_export]
macro_rules! fft {
    ($x:expr) => {
        {
            use fftw::plan::*;
            use fftw::types::{c32, Sign, Flag};
            use fftw::types::c64;

            let mut out = $x.clone();
            C2CPlan64::aligned($x.shape(), Sign::Forward, Flag::MEASURE)
                .unwrap()
                .c2c($x.as_slice_mut().unwrap(), out.as_slice_mut().unwrap())
                .unwrap();
            out
        }
    }
}

#[macro_export]
macro_rules! ifft {
    ($x:expr) => {
        {
            use fftw::plan::*;
            use fftw::types::{c32, Sign, Flag};
            use fftw::types::c64;

            let mut out = $x.clone();
            let norm_fact = c64::new(out.len() as f64, 0.0);
            C2CPlan64::aligned($x.shape(), Sign::Backward, Flag::MEASURE)
                .unwrap()
                .c2c($x.as_slice_mut().unwrap(), out.as_slice_mut().unwrap())
                .unwrap();
            out /= norm_fact;
            out
        }
    }
}


#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{arr1, arr2, arr3};
    use ndarray::{Array1,Array2,Array3};
    use fftw::types::c64;

    #[test]
    fn test_fft_macro_1d() {
        let input = (1..=9)
            .map(|x| c64::new(x as f64, 0.0))
            .collect::<Array1::<c64>>();
        let mut output: Array1<c64> = fft!(input.clone());
        let expected = &[
            c64 { re: 45.0, im:   0.0, },
            c64 { re: -4.5, im:  12.363648387545801, },
            c64 { re: -4.5, im:   5.362891166673945, },
            c64 { re: -4.5, im:   2.598076211353316, },
            c64 { re: -4.5, im:   0.7934714131880916, },
            c64 { re: -4.5, im:  -0.793471413188092, },
            c64 { re: -4.5, im:  -2.598076211353316, },
            c64 { re: -4.5, im:  -5.362891166673945, },
            c64 { re: -4.5, im: -12.363648387545801, }];
        output.iter().zip(expected.iter())
            .for_each(|(x, y)|
                assert!((x.norm() - y.norm()).abs() < 1e-15));

        let ifft_output: Array1<c64> = ifft!(output);
        ifft_output.iter().zip(input.iter())
            .for_each(|(x, y)|
                assert!((x.norm() - y.norm()).abs() < 1e-15));
    }

    #[test]
    fn test_fft_macro_2d() {
        let input = (1..=9)
            .map(|x| c64::new(x as f64, 0.0))
            .collect::<Vec<_>>();
        let input = Array2::from_shape_vec((3, 3), input).unwrap();
        let mut output: Array2<c64> = fft!(input.clone());
        let expected = arr2(&[
            [c64 { re: 45.0, im: 0.0, },
                c64 { re: -4.5, im: 2.598076211353316, },
                c64 { re: -4.5, im: -2.598076211353316, }],
            [c64 { re: -13.5, im: 7.794228634059947, },
                c64 { re: 0.0, im: 0.0, },
                c64 { re: 0.0, im: 0.0, }],
            [c64 { re: -13.5, im: -7.794228634059947, },
                c64 { re: 0.0, im: 0.0, },
                c64 { re: 0.0, im: 0.0, }]
        ]);
        output.iter().zip(expected.iter())
            .for_each(|(x, y)|
                assert!((x.norm() - y.norm()).abs() < 1e-15));

        let ifft_output: Array2<c64> = ifft!(output);
        ifft_output.iter().zip(input.iter())
            .for_each(|(x, y)|
                assert!((x.norm() - y.norm()).abs() < 1e-15));
    }

    #[test]
    fn test_fft_macro_3d() {
        let input = (1..=27)
            .map(|x| c64::new(x as f64, 0.0))
            .collect::<Vec<_>>();
        let input = Array3::from_shape_vec((3, 3, 3), input).unwrap();
        let mut output: Array3<c64> = fft!(input.clone());
        let expected = arr3(
            &[[[c64 { re: 378.0, im: 0.0, }, c64 { re: -13.5, im: 7.794228634059948, }, c64 { re: -13.5, im: -7.794228634059948, }],
                [c64 { re: -40.5, im: 23.38268590217984, }, c64 { re: 0.0, im: 0.0, }, c64 { re: 0.0, im: 0.0, }],
                [c64 { re: -40.5, im: -23.38268590217984, }, c64 { re: 0.0, im: 0.0, }, c64 { re: 0.0, im: 0.0, }]],

                [[c64 { re: -121.5, im: 70.14805770653953, }, c64 { re: 0.0, im: 0.0, }, c64 { re: 0.0, im: 0.0, }],
                    [c64 { re: 0.0, im: 0.0, }, c64 { re: 0.0, im: 0.0, }, c64 { re: 0.0, im: 0.0, }],
                    [c64 { re: 0.0, im: 0.0, }, c64 { re: 0.0, im: 0.0, }, c64 { re: 0.0, im: 0.0, }]],

                [[c64 { re: -121.5, im: -70.14805770653953, }, c64 { re: 0.0, im: 0.0, }, c64 { re: 0.0, im: 0.0, }],
                    [c64 { re: 0.0, im: 0.0, }, c64 { re: 0.0, im: 0.0, }, c64 { re: 0.0, im: 0.0, }],
                    [c64 { re: 0.0, im: 0.0, }, c64 { re: 0.0, im: 0.0, }, c64 { re: 0.0, im: 0.0, }]]]
        );
        output.iter().zip(expected.iter())
            .for_each(|(x, y)|
                assert!((x.norm() - y.norm()).abs() < 1e-15));

        let ifft_output: Array3<c64> = ifft!(output);
        ifft_output.iter().zip(input.iter())
            .for_each(|(x, y)|
                assert!((x.norm() - y.norm()).abs() < 1e-15));
    }
}
