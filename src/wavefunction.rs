#![allow(unused)]

use ndarray::Array3;
use ndarray::Array2;
use num::complex::Complex64;
use ndarray_linalg::norm::Norm;
use rayon::prelude::*;

use vasp_poscar::Poscar;
use vaspchg_rs::{
    ChgType,
    ChgBase,
};

// use crate::utils::*;
use crate::wavecar::*;
use crate::constants::*;

#[derive(Clone, Debug)]
pub struct Wavefunction {
    wavecar_type: WavecarType,
    ispin: u64,
    ikpoint: u64,
    iband: u64,

    real_cell: Array2<f64>,
    eigen_val: f64,

    kvec: Vec<f64>,
    ngrid: Vec<u64>,
    data: Array3<Complex64>,
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

impl Wavefunction {
    pub(crate) fn new(wavecar_type: WavecarType,
                      ispin: u64,
                      ikpoint: u64,
                      iband: u64,
                      real_cell: Array2<f64>,
                      eigen_val: f64,
                      kvec: Vec<f64>,
                      ngrid: Vec<u64>,
                      data: Array3<Complex64>) -> Self {
        Self {
            wavecar_type,
            ispin,
            ikpoint,
            iband,
            real_cell,
            eigen_val,
            kvec,
            ngrid,
            data,
        }
    }

    pub fn apply_phase(mut self, r0: &[f64; 3]) -> Self {
        let ngx = self.data.shape()[0];
        let ngy = self.data.shape()[1];
        let ngz = self.data.shape()[2];

        let kx = self.kvec[0];
        let ky = self.kvec[1];
        let kz = self.kvec[2];

        let fx: Vec<usize> = (0..ngx).collect();
        let fy: Vec<usize> = (0..ngy).collect();
        let fz: Vec<usize> = (0..ngz).collect();

        let phases_vec = fx
            .iter()
            .flat_map(|&x| {
                let fz = &fz;
                fy.iter().flat_map(move |&y| {
                    let fz = &fz;
                    fz.iter().map(move |&z| [x, y, z])
                })
            })
            // lines in the below equal to
            // for x in fx {
            //   for y in fy {
            //     for z in fz {
            //       ...
            // }}}
            .map(|[x, y, z]| {
                (x as f64 + r0[0]) * kx + (y as f64 + r0[1]) * ky + (z as f64 + r0[2]) * kz
            })
            .map(|v| (Complex64::new(0.0, 1.0) * PI * 2.0 * v).exp())
            // Exp[2 * pi * (k_i + r0) * g]
            .collect::<Vec<Complex64>>();

        let phases_vec: Array3<Complex64> =
            Array3::from_shape_vec((ngx, ngy, ngz), phases_vec).unwrap();
        self.data *= &phases_vec;
        self
    }

    pub fn normalize(mut self) -> Self {
        let norm = self.data.norm();
        self.data.par_mapv_inplace(|v| v.unscale(norm));
        self
    }

    pub fn get_charge_density(&self) -> Array3<f64> {
        let shape = self.data.shape();
        let vec = self.data.par_iter()
            .map(|v: &Complex64| v.norm_sqr())
            .collect::<Vec<f64>>();
        Array3::from_shape_vec((shape[0], shape[1], shape[2]), vec)
            .unwrap()
    }

    pub fn get_wavefun_realgrid(&self) -> Array3<f64> {
        let shape = self.data.shape();
        let vec = self.data.par_iter()
            .map(|v: &Complex64| v.re)
            .collect::<Vec<f64>>();
        Array3::from_shape_vec((shape[0], shape[1], shape[2]), vec)
            .unwrap()
    }

    pub fn get_wavefun_imagegrid(&self) -> Array3<f64> {
        let shape = self.data.shape();
        let vec = self.data.par_iter()
            .map(|v: &Complex64| v.im)
            .collect::<Vec<f64>>();
        Array3::from_shape_vec((shape[0], shape[1], shape[2]), vec)
            .unwrap()
    }

    pub fn into_parchg_obj(self, poscar: &Poscar) -> ChgBase {
        let chg = self.get_charge_density();
        ChgBase::from_builder(chg, vec![], poscar.clone())
    }

    pub fn into_vesta_obj(self, poscar: &Poscar) -> ChgBase {
        let chg = self.get_wavefun_realgrid();
        ChgBase::from_builder(chg, vec![], poscar.clone())
    }
}
