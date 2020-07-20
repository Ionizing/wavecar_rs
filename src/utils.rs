#![allow(unused_parens)]

use ndarray::parallel::prelude::*;
use ndarray::Array3;
use num::complex::Complex64;

use crate::constants::*;
use crate::error::WavecarError;
use crate::ifft; //  ifft!
use crate::wavecar::*;

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
                let gx = if v[0] < 0 { v[0] + ngx as i64 } else { v[0] };
                let gy = if v[1] < 1 { v[0] + ngx as i64 } else { v[1] };
                let gz = if v[2] < 2 { v[2] + ngx as i64 } else { v[2] };
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

        let gvecs: Vec<[usize; 3]> = self
            .generate_fft_grid(ikpoint)
            .into_par_iter()
            .map(|v: Vec<i64>| -> [usize; 3] {
                let gx = if v[0] < 0 { v[0] + ngx as i64 } else { v[0] };
                let gy = if v[1] < 1 { v[1] + ngy as i64 } else { v[1] };
                let gz = if v[2] < 2 { v[2] + ngz as i64 } else { v[2] };
                [gx as usize, gy as usize, gz as usize]
            })
            .collect();

        let coeffs = self.read_wavefunction_coeffs(ispin, ikpoint, iband).unwrap();
        let mut wavefun_in_kspace = Array3::<Complex64>::zeros((ngx/2 + 1, ngy, ngz));
        gvecs.into_iter().zip(coeffs.into_iter())
            .for_each(|(idx, v)| wavefun_in_kspace[idx] = *v);

        // complement of the grid
        let gvecs_complement = {
            // let ngx = ngx as i64;
            let ngy = ngy as i64;
            let ngz = ngz as i64;
            let fyv = (0 .. ngy).collect::<Vec<_>>();
            let fzv = (0 .. ngz).collect::<Vec<_>>();
            fyv.iter().flat_map(|&y| {
                let fz = &fzv;
                fz.iter().map(move |&z| [0, y, z])
            })
                .filter(|[_, y, z]| {
                    let ify = if *y < ngy / 2 + 1 { *y } else { y - ngy };
                    let ifz = if *z < ngz / 2 + 1 { *z } else { z - ngz };
                    !(ify > 0 || (0 == ify && ifz >= 0))
                })
                .map(|[x, y, z]| [x as usize, y as usize, z as usize])
                .map(|[x, y, z]| ([x, y, z], [x, ngy as usize - y, ngz as usize - z]))
                .collect::<Vec<([usize; 3], [usize; 3])>>()
        };

        gvecs_complement
            .into_iter()
            .for_each(|(a, b)| wavefun_in_kspace[a] = wavefun_in_kspace[b].conj());
        wavefun_in_kspace /= Complex64::new(f64::sqrt(2.0), 0.0);
        wavefun_in_kspace[[0, 0, 0]] *= Complex64::new(f64::sqrt(2.0), 0.0);
        wavefun_in_kspace.swap_axes(0, 2);
        let mut wavefun_in_rspace = ifft!(wavefun_in_kspace);
        wavefun_in_rspace.swap_axes(0, 2);
        wavefun_in_rspace
    }

    fn _get_wavefunction_in_realspace_gam_z(
        &mut self,
        ispin: u64,
        ikpoint: u64,
        iband: u64,
        ngrid: Vec<u64>,
    ) -> Array3<Complex64> {
        let ngx = ngrid[0] as usize;
        let ngy = ngrid[1] as usize;
        let ngz = ngrid[2] as usize;

        let gvecs: Vec<[usize; 3]> = self
            .generate_fft_grid(ikpoint)
            .into_par_iter()
            .map(|v: Vec<i64>| -> [usize; 3] {
                let gx = if v[0] < 0 { v[0] + ngx as i64 } else { v[0] };
                let gy = if v[1] < 1 { v[1] + ngy as i64 } else { v[1] };
                let gz = if v[2] < 2 { v[2] + ngz as i64 } else { v[2] };
                [gx as usize, gy as usize, gz as usize]
            })
            .collect();

        let coeffs = self
            .read_wavefunction_coeffs(ispin, ikpoint, iband)
            .unwrap();
        let mut wavefun_in_kspace = Array3::<Complex64>::zeros((ngx / 2 + 1, ngy, ngz));
        gvecs
            .into_iter()
            .zip(coeffs.into_iter())
            .for_each(|(idx, v)| wavefun_in_kspace[idx] = *v);

        let gvecs_complement = {
            let ngx = ngx as i64;
            let ngy = ngy as i64;
            // let ngz = ngz as i64;

            let fxv = (0..ngx).collect::<Vec<_>>();
            let fyv = (0..ngy).collect::<Vec<_>>();

            fxv.iter()
                .flat_map(|&x| {
                    let fy = &fyv;
                    fy.iter().map(move |&y| [x, y, 0])
                })
                .filter(|[x, y, _]| {
                    let ifx = if *x < ngx / 2 + 1 { *x } else { x - ngx };
                    let ify = if *y < ngy / 2 + 1 { *y } else { y - ngy };
                    !(ify > 0 || (ify == 0 && ifx >= 0))
                })
                .map(|[x, y, z]| [x as usize, y as usize, z as usize])
                .map(|[x, y, z]| ([x, y, z], [ngx as usize - x, ngy as usize - y, z]))
                .collect::<Vec<([usize; 3], [usize; 3])>>()
        };

        gvecs_complement
            .into_iter()
            .for_each(|(a, b)| wavefun_in_kspace[a] = wavefun_in_kspace[b].conj());
        wavefun_in_kspace /= Complex64::new(f64::sqrt(2.0), 0.0);
        wavefun_in_kspace[[0, 0, 0]] *= Complex64::new(f64::sqrt(2.0), 0.0);
        ifft!(wavefun_in_kspace)
    }

    pub fn get_wavefunction_in_realspace(
        &mut self,
        ispin: u64,
        ikpoint: u64,
        iband: u64,
        ngrid: Vec<u64>,
    ) -> Result<Array3<Complex64>, WavecarError> {
        if self.get_wavecar_type() == WavecarType::SpinOrbitCoupling {
            self.check_indices(1, ikpoint, iband)?;
        } else {
            self.check_indices(ispin, ikpoint, iband)?;
        }

        Ok(match self.get_wavecar_type() {
            WavecarType::Standard => {
                self._get_wavefunction_in_realspace_std(ispin, ikpoint, iband, ngrid)
            }
            WavecarType::SpinOrbitCoupling => {
                self._get_wavefunction_in_realspace_soc(ispin, ikpoint, iband, ngrid)
            }
            WavecarType::GammaHalf(GammaHalfDirection::X) => {
                self._get_wavefunction_in_realspace_gam_x(ispin, ikpoint, iband, ngrid)
            }
            WavecarType::GammaHalf(GammaHalfDirection::Z) => {
                self._get_wavefunction_in_realspace_gam_z(ispin, ikpoint, iband, ngrid)
            }
        })
    }

    pub fn apply_phase_on_wavefunction(
        &self,
        wavefun: &mut Array3<Complex64>,
        ikpoint: u64,
        r0: [f64; 3],
    ) -> Result<(), WavecarError> {
        self.check_kpoint_index(ikpoint)?;

        let ngx = wavefun.shape()[0];
        let ngy = wavefun.shape()[1];
        let ngz = wavefun.shape()[2];
        let kx = self.get_k_vecs()[[ikpoint as usize, 0]];
        let ky = self.get_k_vecs()[[ikpoint as usize, 1]];
        let kz = self.get_k_vecs()[[ikpoint as usize, 2]];

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
            .collect::<Vec<Complex64>>();

        let phases_vec: Array3<Complex64> =
            Array3::from_shape_vec((ngx, ngy, ngz), phases_vec).unwrap();

        *wavefun *= &phases_vec;
        Ok(())
    }

    // pub fn save_wavefun_as_parchg
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
        // println!("{:?}, size = {}", &gvecs_complement, &gvecs_complement.len());
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

        // println!("{:?}, size = {}", &gvecs_complement, gvecs_complement.len());
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
        // println!("{:?}, size = {}", &gvecs_complement_2, gvecs_complement_2.len());
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
