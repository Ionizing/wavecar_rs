#![allow(unused)]

use ndarray::{Array3};
use num::complex::Complex64;
use rayon::prelude::*;

use crate::wavecar::*;
use crate::error::WavecarError;
#[macro_use]
use crate::fft;
#[macro_use]
use crate::ifft;
use crate::wavecar::WFPrecisionType::Complex32;

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
                let gy = if v[1] < 1 { v[1] + ngx as i64 } else { v[1] };
                let gz = if v[2] < 2 { v[2] + ngx as i64 } else { v[2] };
                [gx as usize, gy as usize, gz as usize]
            })
            .collect();
        let coeffs = self.read_wavefunction_coeffs(ispin, ikpoint, iband).unwrap();
        let mut wavefun_in_kspace = Array3::<Complex64>::zeros((ngx, ngy, ngz));
        gvecs.into_iter().zip(coeffs.into_iter())
            .for_each(|(idx, v)| wavefun_in_kspace[idx] = *v);
        ifft!(wavefun_in_kspace)
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
                let gx = if v[0] < 0 {v[0] + ngx as i64} else { v[0] };
                let gy = if v[1] < 1 {v[0] + ngx as i64} else { v[1] };
                let gz = if v[2] < 2 {v[2] + ngx as i64} else { v[2] };
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
        ifft!(wavefun_in_kspace)
    }

    fn _get_wavefunction_in_realspace_gam_x(&mut self,
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
                let gx= if v[0] < 0 { v[0] + ngx as i64 } else { v[0] };
                let gy= if v[1] < 1 { v[1] + ngy as i64 } else { v[1] };
                let gz= if v[2] < 2 { v[2] + ngz as i64 } else { v[2] };
                [gx as usize, gy as usize, gz as usize]
            })
            .collect();

        let coeffs = self.read_wavefunction_coeffs(ispin, ikpoint, iband).unwrap();
        let mut wavefun_in_kspace = Array3::<Complex64>::zeros((ngx/2 + 1, ngy, ngz));
        gvecs.into_iter().zip(coeffs.into_iter())
            .for_each(|(idx, v)| wavefun_in_kspace[idx] = *v);

        // complement of the grid
        let gvecs_complement = {
            let ngx = ngx as i64;
            let ngy = ngy as i64;
            let ngz = ngz as i64;
            let fyv = (0 .. ngy).collect::<Vec<_>>();
            let fzv = (0 .. ngz).collect::<Vec<_>>();
            fyv.iter().flat_map(|&y| {
                let fz = &fzv;
                fz.into_iter().map(move |&z| [0, y, z])
            })
                .filter(|[_, y, z]| {
                    let ify = if *y < ngy/2 + 1 { *y } else { y - ngy };
                    let ifz = if *z < ngz/2 + 1 { *z } else { z - ngz };
                    !(ify > 0 || (0 == ify && ifz >= 0))
                })
                .map(|[x, y, z]| [x as usize, y as usize, z as usize])
                .map(|[x, y, z]| ([x, y, z],
                                  [x, ngy as usize - y, ngz as usize - z]))
                .collect::<Vec::<([usize; 3], [usize; 3])>>()
        };

        gvecs_complement.into_iter()
            .for_each(|(a, b)| wavefun_in_kspace[a] = wavefun_in_kspace[b].conj() );
        wavefun_in_kspace /= Complex64::new(f64::sqrt(2.0), 0.0);
        wavefun_in_kspace[[0, 0, 0]] *= Complex64::new(f64::sqrt(2.0), 0.0);
        wavefun_in_kspace.swap_axes(0, 2);
        let mut wavefun_in_rspace = ifft!(wavefun_in_kspace);
        wavefun_in_rspace.swap_axes(0, 2);
        wavefun_in_rspace
    }

    fn _get_wavefunction_in_realspace_gam_z(&mut self,
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
                let gx= if v[0] < 0 { v[0] + ngx as i64 } else { v[0] };
                let gy= if v[1] < 1 { v[1] + ngy as i64 } else { v[1] };
                let gz= if v[2] < 2 { v[2] + ngz as i64 } else { v[2] };
                [gx as usize, gy as usize, gz as usize]
            })
            .collect();

        let coeffs = self.read_wavefunction_coeffs(ispin, ikpoint, iband).unwrap();
        let mut wavefun_in_kspace = Array3::<Complex64>::zeros((ngx/2 + 1, ngy, ngz));
        gvecs.into_iter().zip(coeffs.into_iter())
            .for_each(|(idx, v)| wavefun_in_kspace[idx] = *v);

        let gvecs_complement = {
            let ngx = ngx as i64;
            let ngy = ngy as i64;
            let ngz = ngz as i64;

            let fxv = (0 .. ngx).collect::<Vec<_>>();
            let fyv = (0 .. ngy).collect::<Vec<_>>();

            fxv.iter().flat_map(|&x| {
                let fy = &fyv;
                fy.into_iter().map(move |&y| [x, y, 0])
            })
                .filter(|[x, y, _]|{
                    let ifx = if *x < ngx/2 + 1 { *x } else { x - ngx };
                    let ify = if *y < ngy/2 + 1 { *y } else { y - ngy };
                    !(ify > 0 || (ify == 0 && ifx >= 0))
                })
                .map(|[x, y, z]| [x as usize, y as usize, z as usize])
                .map(|[x, y, z]| ([x, y, z],
                                  [ngx as usize - x, ngy as usize - y, z]))
                .collect::<Vec::<([usize; 3], [usize; 3])>>()
        };

        gvecs_complement.into_iter()
            .for_each(|(a, b)| wavefun_in_kspace[a] = wavefun_in_kspace[b].conj() );
        wavefun_in_kspace /= Complex64::new(f64::sqrt(2.0), 0.0);
        wavefun_in_kspace[[0, 0, 0]] *= Complex64::new(f64::sqrt(2.0), 0.0);
        ifft!(wavefun_in_kspace)
    }

    fn _get_wavefunction_in_realspace_gam(&mut self,
                                          ispin: u64,
                                          ikpoint: u64,
                                          iband: u64,
                                          ngrid: Vec<u64>) -> Array3<Complex64> {
        match self.get_vasp_type() {
            VaspType::GammaHalf(GammaHalfDirection::X) =>
                self._get_wavefunction_in_realspace_gam_x(ispin, ikpoint, iband, ngrid),
            VaspType::GammaHalf(GammaHalfDirection::Z) =>
                self._get_wavefunction_in_realspace_gam_z(ispin, ikpoint, iband, ngrid),
            _ => unreachable!(),
        }
    }

    pub fn get_wavefunction_in_realspace(&mut self,
                                         ispin: u64,
                                         ikpoint: u64,
                                         iband: u64,
                                         ngrid: Vec<u64>) -> Result<Array3<Complex64>, WavecarError> {
        if self.get_vasp_type() == VaspType::SpinOrbitCoupling {
            self.check_indices(1, ikpoint, iband)?;
        } else {
            self.check_indices(ispin, ikpoint, iband)?;
        }

        Ok(
            match self.get_vasp_type() {
                VaspType::Standard =>
                    self._get_wavefunction_in_realspace_std(ispin, ikpoint, iband, ngrid),
                VaspType::SpinOrbitCoupling =>
                    self._get_wavefunction_in_realspace_soc(ispin, ikpoint, iband, ngrid),
                VaspType::GammaHalf(GammaHalfDirection::X) =>
                    self._get_wavefunction_in_realspace_gam_x(ispin, ikpoint, iband, ngrid),
                VaspType::GammaHalf(GammaHalfDirection::Z) =>
                    self._get_wavefunction_in_realspace_gam_z(ispin, ikpoint, iband, ngrid),
            }
        )
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn test_fft_grid_complement() {
        let ngx = 5i64;
        let ngy = 6i64;
        let ngz = 7i64;
        // x direction
        let mut gvecs_complement = Vec::<[i64; 3]>::new();
        for ify in (0 .. ngy as i64) {
            for ifz in (0 .. ngz as i64) {
                let fy = if ify < ngy as i64 / 2 + 1 { ify } else { ify - ngy as i64};
                let fz = if ifz < ngz as i64 / 2 + 1 { ifz } else { ifz - ngz as i64};
                if fy > 0 || (0 == fy && fz >= 0) { continue; }
                // fy > 0   <==>   ify < ngy/2 + 1
                // fy == 0   <==>   ify == 0
                // fz >=0    <==>   ifz <= ngz/2 + 1
                gvecs_complement.push(
                    [0, fy, fz]);
            }
        }
        println!("{:?}", gvecs_complement);
    }

    #[test]
    fn gvecs_complement_2() {
        let ngx = 5i64;
        let ngy = 6i64;
        let ngz = 7i64;

        let mut gvecs_complement = Vec::<[i64; 3]>::new();
        for ify in (0 .. ngy) {
            for ifz in (0 .. ngz) {
                let fy = if ify < ngy/2 + 1 { ify } else { ify - ngy };
                let fz = if ifz < ngz/2 + 1 { ifz } else { ifz - ngz };
                if fy > 0 || (0 == fy && fz >= 0) { continue; }
                gvecs_complement.push(
                    [0, fy, fz] );
            }
        }

        println!("{:?}, size = {}", &gvecs_complement, gvecs_complement.len());

        let fy = (0 .. ngy).collect::<Vec<_>>();
        let fz = (0 .. ngz).collect::<Vec<_>>();
        let gvecs_complement_2 = fy.iter().flat_map(|&y| {
            let fz = &fz;
            fz.into_iter().map(move |&z| [0, y, z]) })
            .filter(|[_, y, z]| {
                let ify = if *y < ngy/2 + 1 { *y } else { y - ngy };
                let ifz = if *z < ngz/2 + 1 { *z } else { z - ngz };
                !(ify > 0 || (0 == ify && ifz >= 0))
            })
            .collect::<Vec::<[i64; 3]>>();
        println!("{:?}", gvecs_complement_2);
    }
}
