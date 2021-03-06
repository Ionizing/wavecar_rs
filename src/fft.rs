use fftw::plan::*;
use fftw::types::{Flag, Sign};
use ndarray::{Array3, Array1};
use num::complex::Complex64;

// #[macro_export]
// macro_rules! fft {
//     ($x:expr) => {{
//         use fftw::plan::*;
//         use fftw::types::{Flag, Sign};
//
//         let mut out = $x.clone();
//         C2CPlan64::aligned($x.shape(), Sign::Forward, Flag::ESTIMATE)
//             .unwrap()
//             .c2c($x.as_slice_mut().unwrap(), out.as_slice_mut().unwrap())
//             .unwrap();
//         out
//     }};
// }

pub(crate) fn ifft3(arr: Array3<Complex64>) -> Array3<Complex64> {
    let mut _in = arr.as_standard_layout();
    let mut out = Array3::zeros(_in.dim());
    let norm_fact = out.len() as f64;

    C2CPlan64::aligned(_in.shape(), Sign::Backward, Flag::ESTIMATE).unwrap()
        .c2c(_in.as_slice_mut().unwrap(), out.as_slice_mut().unwrap())
        .unwrap();
    out.par_mapv_inplace(|v| v.unscale(norm_fact));
    out
}

pub(crate) fn irfft3(arr: Array3<Complex64>, grid: (usize, usize, usize))  -> Array3<Complex64> {
    let mut _in = arr.as_standard_layout();
    let mut out = Array3::<f64>::zeros(grid);
    let norm_fact = out.len() as f64;

    C2RPlan64::aligned(out.shape(), Flag::ESTIMATE).unwrap()
        .c2r(_in.as_slice_mut().unwrap(), out.as_slice_mut().unwrap())
        .unwrap();
    out.into_iter().map(|v| Complex64::new(*v / norm_fact, 0.0))
        .collect::<Array1<Complex64>>()
        .into_shape(grid)
        .unwrap()
}


// #[macro_export]
// macro_rules! ifft {
//     ($x:expr) => {{
//         use fftw::plan::*;
//         use fftw::types::{Flag, Sign};
//
//         let mut _in = $x.as_standard_layout();
//         let mut out = $x.as_standard_layout().into_owned();
//         let norm_fact = out.len() as f64;
//         C2CPlan64::aligned(_in.shape(), Sign::Backward, Flag::ESTIMATE).unwrap()
//             .c2c(_in.as_slice_mut().unwrap(), out.as_slice_mut().unwrap())
//             .unwrap();
//         out.par_mapv_inplace(|v| v.unscale(norm_fact));
//         out
//     }};
// }

// #[macro_export]
// macro_rules! irfft {
//     ($x:expr, $s:expr) => {{
//         use fftw::plan::*;
//         use fftw::types::{Flag, Sign, c64};
//         use ndarray::{ArrayD, IxDyn};
//
//         let mut _in = $x.as_standard_layout();
//         let mut out = Array3::<f64>::zeros($s);
//         let norm_fact = out.len() as f64;
//         C2RPlan64::aligned(out.shape(), Flag::ESTIMATE).unwrap()
//             .c2r(_in.as_slice_mut().unwrap(), out.as_slice_mut().unwrap())
//             .unwrap();
//
//         let mut complex_out = Array3::<c64>::zeros($s);
//         complex_out.iter_mut().zip(out.iter())
//             .for_each(|(y, x)| y.re = *x / norm_fact);
//         complex_out.into_owned()
//     }}
// }

// #[cfg(test)]
// #[ignore]
// mod test {
//     use fftw::types::*;
//     use ndarray::{arr2, arr3};
//     use ndarray::{Array1, Array2, Array3};
//
//     #[test]
//     fn test_fft_macro_1d() {
//         let input = (1..=9)
//             .map(|x| c64::new(x as f64, 0.0))
//             .collect::<Array1<c64>>();
//         let output: Array1<c64> = fft!(input.clone());
//         let expected = &[
//             c64 { re: 45.0, im:   0.0, },
//             c64 { re: -4.5, im:  12.363648387545801, },
//             c64 { re: -4.5, im:   5.362891166673945, },
//             c64 { re: -4.5, im:   2.598076211353316, },
//             c64 { re: -4.5, im:   0.7934714131880916, },
//             c64 { re: -4.5, im:  -0.793471413188092, },
//             c64 { re: -4.5, im:  -2.598076211353316, },
//             c64 { re: -4.5, im:  -5.362891166673945, },
//             c64 { re: -4.5, im: -12.363648387545801, }];
//         output.iter().zip(expected.iter())
//             .for_each(|(x, y)|
//                 assert!((x.norm() - y.norm()).abs() < 1e-15));
//
//         let ifft_output: Array1<c64> = ifft!(output);
//         ifft_output.iter().zip(input.iter())
//             .for_each(|(x, y)|
//                 assert!((x.norm() - y.norm()).abs() < 1e-15));
//     }
//
//     #[test]
//     fn test_fft_macro_2d() {
//         let input = (1..=9).map(|x| c64::new(x as f64, 0.0)).collect::<Vec<_>>();
//         let input = Array2::from_shape_vec((3, 3), input).unwrap();
//         let output: Array2<c64> = fft!(input.clone());
//         let expected = arr2(&[
//             [c64 { re: 45.0, im: 0.0, },
//                 c64 { re: -4.5, im: 2.598076211353316, },
//                 c64 { re: -4.5, im: -2.598076211353316, }],
//             [c64 { re: -13.5, im: 7.794228634059947, },
//                 c64 { re: 0.0, im: 0.0, },
//                 c64 { re: 0.0, im: 0.0, }],
//             [c64 { re: -13.5, im: -7.794228634059947, },
//                 c64 { re: 0.0, im: 0.0, },
//                 c64 { re: 0.0, im: 0.0, }]
//         ]);
//         output.iter().zip(expected.iter())
//             .for_each(|(x, y)|
//                 assert!((x.norm() - y.norm()).abs() < 1e-15));
//
//         let ifft_output: Array2<c64> = ifft!(output);
//         ifft_output.iter().zip(input.iter())
//             .for_each(|(x, y)|
//                 assert!((x.norm() - y.norm()).abs() < 1e-15));
//     }
//
//     #[test]
//     fn test_fft_macro_3d() {
//         let input = (1..=27)
//             .map(|x| c64::new(x as f64, 0.0))
//             .collect::<Vec<_>>();
//         let input = Array3::from_shape_vec((3, 3, 3), input).unwrap();
//         let output: Array3<c64> = fft!(input.clone());
//         let expected = arr3(
//             &[[[c64 { re: 378.0, im: 0.0, }, c64 { re: -13.5, im: 7.794228634059948, }, c64 { re: -13.5, im: -7.794228634059948, }],
//                 [c64 { re: -40.5, im: 23.38268590217984, }, c64 { re: 0.0, im: 0.0, }, c64 { re: 0.0, im: 0.0, }],
//                 [c64 { re: -40.5, im: -23.38268590217984, }, c64 { re: 0.0, im: 0.0, }, c64 { re: 0.0, im: 0.0, }]],
//
//                 [[c64 { re: -121.5, im: 70.14805770653953, }, c64 { re: 0.0, im: 0.0, }, c64 { re: 0.0, im: 0.0, }],
//                     [c64 { re: 0.0, im: 0.0, }, c64 { re: 0.0, im: 0.0, }, c64 { re: 0.0, im: 0.0, }],
//                     [c64 { re: 0.0, im: 0.0, }, c64 { re: 0.0, im: 0.0, }, c64 { re: 0.0, im: 0.0, }]],
//
//                 [[c64 { re: -121.5, im: -70.14805770653953, }, c64 { re: 0.0, im: 0.0, }, c64 { re: 0.0, im: 0.0, }],
//                     [c64 { re: 0.0, im: 0.0, }, c64 { re: 0.0, im: 0.0, }, c64 { re: 0.0, im: 0.0, }],
//                     [c64 { re: 0.0, im: 0.0, }, c64 { re: 0.0, im: 0.0, }, c64 { re: 0.0, im: 0.0, }]]]
//         );
//         output.iter().zip(expected.iter())
//             .for_each(|(x, y)|
//                 assert!((x.norm() - y.norm()).abs() < 1e-15));
//
//         let ifft_output: Array3<c64> = ifft!(output);
//         ifft_output.iter().zip(input.iter())
//             .for_each(|(x, y)|
//                 assert!((x.norm() - y.norm()).abs() < 1e-15));
//     }
// }
