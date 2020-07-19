use ndarray::Array3;
use num::complex::Complex64;

use crate::wavecar::GammaHalfDirection;
use crate::wavecar::VaspType;

pub struct Wavefunction {
    vasp_type: VaspType,
    ispin: u64,
    ikpoint: u64,
    iband: u64,

    kvec: [f64; 3],
    data: Array3<f64>,
}

// Desired function:
// Wavecar::from_file("path/to/wavecar/")
//   .get_wavefunction_in_realspace(ispin, ikpoint, iband, ngrid)
//   .apply_phase(r0)
//   .normalize()
//   .save_as_chgcar()

// or
// Wavecar::from_file("path/to/wavecar/")
//   .get_wavefunction_in_realspace(ispin, ikpoint, iband, ngrid)
//   .apply_phase(r0)
//   .normalize()
//   .into_charge_density()
