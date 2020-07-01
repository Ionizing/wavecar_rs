#![allow(unused)]

use ndarray::{Array3};
use num::complex::Complex64;
use rayon::prelude::*;

use crate::wavecar::*;
use crate::error::WavecarError;


impl Wavecar {
    pub fn _get_wavefunction_in_realspace_std(&mut self,
                                              ispin: u64,
                                              ikpoint: u64,
                                              iband: u64,
                                              ngrid: Vec<u64>) -> Result<Array3::<Complex64>, WavecarError> {
        todo!();
    }

    pub fn get_wavefunction_in_kspace(&mut self,
                                      ispin: u64,
                                      ikpoint: u64,
                                      iband: u64,
                                      spinor: u64) -> Result<Array3::<Complex64>, WavecarError> {
        // let ngrid = self.ngrid.to_owned();
        // let kvec = self.get_k_vecs().row(ikpoint as usize).to_owned();
        // let reci_cell = self.get_reci_cell();
        // let en_cutoff = self.get_en_cutoff();
        // let vasp_type = self.get_vasp_type();
        //
        // // Trick for SOC type
        // let mut ispin = ispin;
        // let mut spinor = 0u64;
        // if self.get_vasp_type() == VaspType::SpinOrbitCoupling {
        //     spinor = ispin;
        //     ispin = 0;
        // }
        //
        // let ngx = ngrid[0] as usize;
        // let ngy = ngrid[1] as usize;
        // let ngz = ngrid[2] as usize;
        // let gvecs: Vec<Vec<usize>> = Self::_generate_fft_grid_specific(
        //     ngrid, kvec, reci_cell, en_cutoff, vasp_type)
        //     .into_par_iter()
        //     .map(|v: Vec<i64>| -> Vec<usize> {
        //         let gx = if v[0] < 0 { v[0] + ngx as i64 } else { v[0] };
        //         let gy = if v[1] < 0 { v[1] + ngx as i64 } else { v[1] };
        //         let gz = if v[2] < 0 { v[2] + ngx as i64 } else { v[2] };
        //         vec![gx as usize, gy as usize, gz as usize]
        //     })
        //     .collect();
        //
        // let coeffs = self.read_wavefunction_coeffs(ispin, ikpoint, iband)?;
        // let mut wavefunc_in_kspace = Array3::<Complex64>::zeros((ngx, ngy, ngz));
        // gvecs.iter().zip(coeffs.into_iter())
        //     .for_each(|(idx, v)| wavefunc_in_kspace[[idx[0], idx[1], idx[2]]] = *v);
        // Ok(wavefunc_in_kspace)
        todo!();
    }

}
