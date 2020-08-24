#![allow(unused)]

use ndarray::parallel::prelude::*;
use ndarray::Array3;
use ndarray::s;
use num::complex::Complex64;

use crate::constants::*;
use crate::error::WavecarError;
// use crate::ifft; //  ifft!
// use crate::irfft;
use crate::fft::{ifft3, irfft3};
use crate::wavecar::*;
use crate::wavefunction::Wavefunction;

impl Wavecar {
    fn _get_wavefunction_in_realspace_std(&mut self,
                                          ispin: u64,
                                          ikpoint: u64,
                                          iband: u64,
                                          ngrid: Vec<u64>) -> Array3<Complex64> {
        let ngx = ngrid[0] as usize;
        let ngy = ngrid[1] as usize;
        let ngz = ngrid[2] as usize;

        let gvecs: Vec<[usize; 3]> = self.generate_fft_grid(ikpoint)
            .into_par_iter()
            .map(|v: Vec<i64>| -> [usize; 3] {
                let gx = if v[0] < 0 { v[0] + ngx as i64 } else { v[0] };
                let gy = if v[1] < 0 { v[1] + ngx as i64 } else { v[1] };
                let gz = if v[2] < 0 { v[2] + ngx as i64 } else { v[2] };
                [gx as usize, gy as usize, gz as usize]
            })
            .collect();
        let coeffs = self.read_wavefunction_coeffs(ispin, ikpoint, iband).unwrap();
        let mut wavefun_in_kspace = Array3::<Complex64>::zeros((ngx, ngy, ngz));
        gvecs.into_iter().zip(coeffs.into_iter())
            .for_each(|(idx, v)| wavefun_in_kspace[idx] = *v);
        ifft3(wavefun_in_kspace)
    }

    fn _get_wavefunction_in_realspace_soc(&mut self,
                                          ispinor: u64,
                                          ikpoint: u64,
                                          iband: u64,
                                          ngrid: Vec<u64>) -> Array3<Complex64> {
        use ndarray::s; // used for creating slices

        let ngx = ngrid[0] as usize;
        let ngy = ngrid[1] as usize;
        let ngz = ngrid[2] as usize;

        let gvecs: Vec<[usize; 3]> = self.generate_fft_grid(ikpoint)
            .into_par_iter()
            .map(|v: Vec<i64>| -> [usize; 3] {
                let gx = if v[0] < 0 { v[0] + ngx as i64 } else { v[0] };
                let gy = if v[1] < 0 { v[1] + ngx as i64 } else { v[1] };
                let gz = if v[2] < 0 { v[2] + ngx as i64 } else { v[2] };
                [gx as usize, gy as usize, gz as usize]
            })
            .collect();

        // NSPIN = 1 for non-collinear version
        let coeffs = self.read_wavefunction_coeffs(0, ikpoint, iband).unwrap();
        let num_plws = coeffs.len() / 2;

        let coeffs = match ispinor {
            0 => coeffs.slice_move(s![0 .. num_plws]),
            1 => coeffs.slice_move(s![num_plws ..]),
            _ => unreachable!(),
        };

        let mut wavefun_in_kspace = Array3::<Complex64>::zeros((ngx, ngy, ngz));
        gvecs.into_iter().zip(coeffs.into_iter())
            .for_each(|(idx, v)| wavefun_in_kspace[idx] = *v);
        ifft3(wavefun_in_kspace)
    }

    fn _get_wavefunction_in_realspace_gam_x(&mut self,
                                            ispin: u64,
                                            ikpoint: u64,
                                            iband: u64,
                                            ngrid: Vec<u64>) -> Array3<Complex64> {
        let ngx = ngrid[0] as usize;
        let ngy = ngrid[1] as usize;
        let ngz = ngrid[2] as usize;

        let gvecs: Vec<[usize; 3]> = self
            .generate_fft_grid(ikpoint)
            .into_par_iter()
            .map(|v: Vec<i64>| -> [usize; 3] {
                let gx = if v[0] < 0 { v[0] + ngx as i64 } else { v[0] };
                let gy = if v[1] < 0 { v[1] + ngy as i64 } else { v[1] };
                let gz = if v[2] < 0 { v[2] + ngz as i64 } else { v[2] };
                [gx as usize, gy as usize, gz as usize]
            })
            .collect();

        let coeffs = self.read_wavefunction_coeffs(ispin, ikpoint, iband).unwrap();
        let mut wavefun_in_kspace = Array3::<Complex64>::zeros((ngx/2 + 1, ngy, ngz));
        gvecs.into_iter().zip(coeffs.into_iter())
            .for_each(|(idx, v)| wavefun_in_kspace[idx] = *v);

        {
            let ngy_ = ngy as i64;
            let ngz_ = ngz as i64;
            for j in 0 .. ngy_ {
                for k in 0 .. ngz_ {
                    let fy = if j < ngy_/2 + 1 { j } else  { j - ngy_ };
                    let fz = if k < ngz_/2 + 1 { k } else  { k - ngz_ };
                    if fy > 0 || (fy == 0 && fz >= 0) { continue; }
                    wavefun_in_kspace[[0, j as usize, k as usize]] =
                        wavefun_in_kspace[[0, (-j).rem_euclid(ngy_) as usize, (-k).rem_euclid(ngy_) as usize]].conj();
                }
            }
        }

        wavefun_in_kspace.par_mapv_inplace(|v| v.unscale(f64::sqrt(2.0)) );
        wavefun_in_kspace[[0, 0, 0]] *= Complex64::new(f64::sqrt(2.0), 0.0);
        wavefun_in_kspace.swap_axes(0, 2);
        let mut wavefun_in_rspace = irfft3(wavefun_in_kspace, (ngx, ngy, ngz));
        wavefun_in_rspace.swap_axes(0, 2);
        wavefun_in_rspace.as_standard_layout().into_owned()
    }

    fn _get_wavefunction_in_realspace_gam_z(&mut self,
                                            ispin: u64,
                                            ikpoint: u64,
                                            iband: u64,
                                            ngrid: Vec<u64>) -> Array3<Complex64> {
        let ngx = ngrid[0] as usize;
        let ngy = ngrid[1] as usize;
        let ngz = ngrid[2] as usize;

        let gvecs: Vec<[usize; 3]> = self
            .generate_fft_grid(ikpoint)
            .into_par_iter()
            .map(|v: Vec<i64>| -> [usize; 3] {
                let gx = if v[0] < 0 { v[0] + ngx as i64 } else { v[0] };
                let gy = if v[1] < 0 { v[1] + ngy as i64 } else { v[1] };
                let gz = if v[2] < 0 { v[2] + ngz as i64 } else { v[2] };
                [gx as usize, gy as usize, gz as usize]
            })
            .collect();

        let coeffs = self .read_wavefunction_coeffs(ispin, ikpoint, iband) .unwrap();
        let mut wavefun_in_kspace = Array3::<Complex64>::zeros((ngx, ngy, ngz/2 + 1));
        gvecs.into_iter().zip(coeffs.into_iter())
            .for_each(|(idx, v)| wavefun_in_kspace[idx] = *v);

        {
            let ngx_ = ngx as i64;
            let ngy_ = ngy as i64;
            for i in 0 .. ngx_ {
                for j in 0 .. ngy_ {
                    let fx = if i < ngx_/2 + 1 { i } else { i - ngx_ };
                    let fy = if j < ngy_/2 + 1 { j } else { j - ngy_ };
                    if (fy > 0) || (fy == 0 && fx >= 0) { continue; }
                    wavefun_in_kspace[[i as usize, j as usize, 0]] =
                        wavefun_in_kspace[[(-i).rem_euclid(ngx_) as usize, (-j).rem_euclid(ngy_) as usize, 0]].conj();
                }
            }
        }

        wavefun_in_kspace.par_mapv_inplace(|v| v.unscale(f64::sqrt(2.0)));
        wavefun_in_kspace[[0, 0, 0]] *= Complex64::new(f64::sqrt(2.0), 0.0);
        irfft3(wavefun_in_kspace, (ngx, ngy, ngz))
    }

    /// Transform the k-space wavefunction into real-space wavefunction.
    ///
    /// A grid size is needed
    pub fn get_wavefunction_in_realspace(&mut self,
                                         ispin: u64,
                                         ikpoint: u64,
                                         iband: u64,
                                         ngrid: Vec<u64>)
                                         -> Result<Wavefunction, WavecarError> {
        if self.get_wavecar_type() == WavecarType::SpinOrbitCoupling {
            self.check_indices(0, ikpoint, iband)?;
        } else {
            self.check_indices(ispin, ikpoint, iband)?;
        }

        let data = match self.get_wavecar_type() {
            WavecarType::Standard => {
                self._get_wavefunction_in_realspace_std(ispin, ikpoint, iband, ngrid.clone())
            }
            WavecarType::SpinOrbitCoupling => {
                self._get_wavefunction_in_realspace_soc(ispin, ikpoint, iband, ngrid.clone())
            }
            WavecarType::GammaHalf(GammaHalfDirection::X) => {
                self._get_wavefunction_in_realspace_gam_x(ispin, ikpoint, iband, ngrid.clone())
            }
            WavecarType::GammaHalf(GammaHalfDirection::Z) => {
                self._get_wavefunction_in_realspace_gam_z(ispin, ikpoint, iband, ngrid.clone())
            }
        };

        let wavecar_type = self.get_wavecar_type();
        let real_cell = self.get_real_cell();
        let eigen_val = self.get_band_eigs()[[ispin as usize, ikpoint as usize, iband as usize]];
        let kvec = self.get_k_vecs().row(ikpoint as usize).to_vec();

        Ok(
            Wavefunction::new(
                wavecar_type,
                ispin,
                ikpoint,
                iband,

                real_cell,
                eigen_val,

                kvec,
                ngrid,
                data
            )
        )
    }

    /// An alias of `get_wavefunction_in_realspace` but with default 2x grid size supplied.
    pub fn get_wavefunction_in_realspace_default_ngrid(&mut self,
                                                       ispin: u64,
                                                       ikpoint: u64,
                                                       iband: u64) -> Result<Wavefunction, WavecarError> {
        let ngrid = self.ngrid.iter().map(|x| x * 2).collect::<Vec<_>>();
        self.get_wavefunction_in_realspace(ispin, ikpoint, iband, ngrid)
    }

}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::arr3;

    #[test]
    fn test_fft_grid_complement() {
        let _ngx = 5i64;
        let ngy = 6i64;
        let ngz = 7i64;
        // x direction
        let mut gvecs_complement = Vec::<[i64; 3]>::new();
        for ify in (0..ngy as i64) {
            for ifz in (0..ngz as i64) {
                let fy = if ify < ngy as i64 / 2 + 1 {
                    ify
                } else {
                    ify - ngy as i64
                };
                let fz = if ifz < ngz as i64 / 2 + 1 {
                    ifz
                } else {
                    ifz - ngz as i64
                };
                if fy > 0 || (0 == fy && fz >= 0) {
                    continue;
                }
                // fy > 0   <==>   ify < ngy/2 + 1
                // fy == 0   <==>   ify == 0
                // fz >=0    <==>   ifz <= ngz/2 + 1
                gvecs_complement.push([0, fy, fz]);
            }
        }
        assert_eq!(17, gvecs_complement.len());
    }

    #[test]
    fn gvecs_complement_2() {
        let _ngx = 5i64;
        let ngy = 6i64;
        let ngz = 7i64;

        let mut gvecs_complement = Vec::<[i64; 3]>::new();
        for ify in (0..ngy) {
            for ifz in (0..ngz) {
                let fy = if ify < ngy / 2 + 1 { ify } else { ify - ngy };
                let fz = if ifz < ngz / 2 + 1 { ifz } else { ifz - ngz };
                if fy > 0 || (0 == fy && fz >= 0) {
                    continue;
                }
                gvecs_complement.push([0, fy, fz]);
            }
        }

        assert_eq!(17, gvecs_complement.len());

        let fy = (0..ngy).collect::<Vec<_>>();
        let fz = (0..ngz).collect::<Vec<_>>();
        let gvecs_complement_2 = fy
            .iter()
            .flat_map(|&y| {
                let fz = &fz;
                fz.into_iter().map(move |&z| [0, y, z])
            })
            .filter(|[_, y, z]| {
                let ify = if *y < ngy / 2 + 1 { *y } else { y - ngy };
                let ifz = if *z < ngz / 2 + 1 { *z } else { z - ngz };
                !(ify > 0 || (0 == ify && ifz >= 0))
            })
            .collect::<Vec<[i64; 3]>>();
        assert_eq!(17, gvecs_complement_2.len());
    }

    #[test]
    fn test_mul_assign_of_two_matrix() {
        let mut mat_a = Array3::from_shape_vec((3, 3, 3), (0..27).collect::<Vec<u32>>()).unwrap();
        let mat_b = Array3::from_shape_vec((3, 3, 3), (27..54).collect::<Vec<u32>>()).unwrap();
        mat_a += &mat_b;
        assert_eq!(
            mat_a,
            arr3(&[
                [[27u32, 29, 31], [33, 35, 37], [39, 41, 43]],
                [[45, 47, 49], [51, 53, 55], [57, 59, 61]],
                [[63, 65, 67], [69, 71, 73], [75, 77, 79]]
            ])
        );
    }
}
