#![feature(test)]
extern crate test;

use test::Bencher;
use ndarray_linalg::generate::random;
use ndarray_linalg::Norm;
use ndarray::Array3;
use ndarray::Array2;
use num::complex::Complex64;

#[bench]
fn bench_norm_sqr_serial(bench: &mut Bencher) {
    let a: Array3<Complex64> = random((300, 300, 300));

    bench.iter(|| {
        let _ = a.mapv(|v: Complex64| { v.norm_sqr() });
    });
}

#[bench]
fn bench_mul_complex(bench: &mut Bencher) {
    let a: Array3<Complex64> = random((300, 300, 300));
    let b: Array3<Complex64> = random((300, 300, 300));

    bench.iter(|| {
        let _ = &a * &b;
    });
}

#[bench]
fn bench_ndarray_norm(bench: &mut Bencher) {
    let a: Array3<Complex64> = random((300, 300, 300));

    bench.iter(|| {
        a.norm();
    });
}

#[bench]
fn bench_ndarray_dot_product(bench: &mut Bencher) {
    let a: Array2<f64> = random((1000, 1000));
    let b: Array2<f64> = random((1000, 1000));

    bench.iter(|| {
        a.dot(&b);
    });
}