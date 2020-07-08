use ndarray::Array3;
use num::complex::Complex64;

use crate::wavecar::VaspType;
use crate::wavecar::GammaHalfDirection;

pub struct Wavefunction {
    vasp_type: VaspType,
    ispin: u64,
    ikpoint: u64,
    iband: u64,

    data: Array3<f64>,
}