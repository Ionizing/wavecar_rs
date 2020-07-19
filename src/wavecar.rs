#![allow(unused_parens)]

use std::fmt;
use std::fs::File;
use std::io::{self, ErrorKind as IoErrorKind, Seek, SeekFrom};
use std::path::Path;

use byteorder::{LittleEndian, ReadBytesExt};
use ndarray::parallel::prelude::*;
use ndarray::{Array1, Array2, Array3};
use ndarray_linalg::{Determinant, Inverse, Norm};
use num::complex::Complex64;

use crate::binary_io::ReadArray;
use crate::constants::*;
use crate::error::{ErrorKind as WavecarErrorKind, WavecarError};

#[derive(PartialEq, Debug, Copy, Clone)]
pub enum WFPrecisionType {
    Complex32,
    Complex64,
}

#[derive(PartialOrd, PartialEq, Debug, Copy, Clone)]
pub enum GammaHalfDirection {
    X,
    Z,
}

#[derive(PartialOrd, PartialEq, Debug, Copy, Clone)]
pub enum VaspType {
    Standard,
    GammaHalf(GammaHalfDirection),
    SpinOrbitCoupling,
}

impl fmt::Display for VaspType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let description = match self {
            VaspType::Standard => "Standard",
            VaspType::SpinOrbitCoupling => "SpinOrbitCouplint",
            VaspType::GammaHalf(GammaHalfDirection::X) => "GammaX",
            VaspType::GammaHalf(GammaHalfDirection::Z) => "GammaZ",
        };
        write!(f, "{}", description)
    }
}

#[derive(Debug)]
pub struct Wavecar {
    // f32, f64; gamma only, std, soc
    file:               File,

    file_len:           u64,
    rec_len:            u64,
    prec_type:          WFPrecisionType,
    vasp_type:          VaspType,

    num_spin:           u64,
    num_kpoints:        u64,
    num_bands:          u64,

    en_cutoff:          f64,
    en_fermi:           f64,

    real_cell:          Array2<f64>,
    reci_cell:          Array2<f64>,

    real_cell_volume:   f64,
    pub (crate)
        ngrid:              Vec<u64>,

    num_plws:           Vec<u64>,
    k_vecs:             Array2<f64>,
    band_eigs:          Array3<f64>,
    band_fweight:       Array3<f64>,
}

// getters
impl Wavecar {
    pub fn get_wavecar_size(&self) -> u64               { self.file_len }
    pub fn get_record_len(&self) -> u64                 { self.rec_len }
    pub fn get_precision_type(&self) -> WFPrecisionType { self.prec_type }
    pub fn get_vasp_type(&self) -> VaspType             { self.vasp_type }
    pub fn get_num_spin(&self) -> u64                   { self.num_spin }
    pub fn get_num_kpoints(&self) -> u64                { self.num_kpoints }
    pub fn get_num_bands(&self) -> u64                  { self.num_bands }
    pub fn get_en_cutoff(&self) -> f64                  { self.en_cutoff }
    pub fn get_en_fermi(&self) -> f64                   { self.en_fermi }
    pub fn get_real_cell(&self) -> Array2<f64>          { self.real_cell.to_owned() } // 3x3, copying cost is low
    pub fn get_reci_cell(&self) -> Array2<f64>          { self.reci_cell.to_owned() } // same as below
    pub fn get_real_cell_volume(&self) -> f64           { self.real_cell_volume }
    pub fn get_num_plane_waves(&self) -> &Vec<u64>      { &self.num_plws }
    pub fn get_k_vecs(&self) -> &Array2<f64>            { &self.k_vecs }
    pub fn get_band_eigs(&self) -> &Array3<f64>         { &self.band_eigs }
    pub fn get_band_fweights(&self) -> &Array3<f64>     { &self.band_fweight }
}

    pub fn set_vasp_type(&mut self, t: VaspType) { self.vasp_type = t }
}

impl Wavecar {
    pub fn from_file(path: &Path) -> io::Result<Self> {
        let mut file = File::open(path)?;
        let file_len = file.metadata()?.len();
        file.seek(SeekFrom::Start(0))?;
        let mut dump = vec![0f64; 3];
        file.read_f64_into::<LittleEndian>(&mut dump)?;
        let rec_len = dump[0] as u64;
        let num_spin = dump[1] as u64;
        let prec_tag = dump[2] as u64;

        let prec_type = match prec_tag {
            45200 => WFPrecisionType::Complex32,
            45210 => WFPrecisionType::Complex64,
            53300 => {
                return Err(io::Error::new(
                    IoErrorKind::Other, "Unsupported WAVECAR format: VASP5 with f32"));
            }
            53310 => {
                return Err(io::Error::new(
                    IoErrorKind::Other, "Unsupported WAVECAR format: VASP5 with f64"));
            }
            _ => {
                return Err(io::Error::new(
                    IoErrorKind::Other, "Invalid WAVECAR format: Unknown VASP version"));
            }
        };

        file.seek(SeekFrom::Start(rec_len))?;
        let mut dump = vec![0f64; 3];
        file.read_f64_into::<LittleEndian>(&mut dump)?;
        let num_kpoints = dump[0] as u64;
        let num_bands = dump[1] as u64;
        let en_cutoff = dump[2];

        let real_cell = file.read_array_2d_f64(3, 3)?;
        let reci_cell = real_cell.inv().unwrap().t().to_owned();
        let real_cell_volume = real_cell.det().unwrap();
        let en_fermi = file.read_f64::<LittleEndian>()?;

        // generate nplws_maxs, maximum num of planewaves on each kpoint
        let ngrid = real_cell
            .outer_iter()
            .map(|row| row.norm() as f64)
            .map(|vec_len| {
                (
                    (en_cutoff / RY_TO_EV).sqrt()
                        /
                        (PIx2 / (vec_len / AU_TO_A))
                ).ceil() as u64
            })
            .map(|x| 2 * x + 1)
            .collect::<Vec<u64>>();

        let (num_plws, k_vecs, band_eigs, band_fweight) =
            Self::_read_band_info(&mut file, num_spin, num_kpoints, num_bands, rec_len)?;

        let vasp_type = Self::_determine_vasp_type(ngrid.clone(),
                                                   k_vecs.row(0).to_owned(),
                                                   reci_cell.clone(),
                                                   en_cutoff,
                                                   num_plws[0]).unwrap();

        Ok(Self {
            file,

            file_len,
            rec_len,
            prec_type,
            vasp_type,

            num_spin,
            num_kpoints,
            num_bands,

            en_cutoff,
            en_fermi,
            real_cell,
            reci_cell,

            real_cell_volume,
            ngrid,

            num_plws,
            k_vecs,
            band_eigs,
            band_fweight,
        })
    }

    fn _read_band_info(file: &mut File,
                       num_spin: u64,
                       num_kpoints: u64,
                       num_bands: u64,
                       rec_len: u64) -> io::Result<(Vec<u64>,       // num_plws, nkpts
                                                    Array2<f64>,    // k_vecs, nkpts * 3
                                                    Array3<f64>,    // band_eigs, nspin * nkpts * nbands
                                                    Array3<f64>)> {  // band_fweight, nspin * nkpts * nbands
        let mut num_plws = Vec::<u64>::new();
        let mut k_vecs = Vec::<f64>::new();
        let mut band_eigs = Vec::<f64>::new();
        let mut band_fweight = Vec::<f64>::new();

        for ispin in 0..num_spin {
            for ikpt in 0..num_kpoints {
                let rec_idx = Self::_calc_record_index(ispin, ikpt, 0, num_kpoints, num_bands);
                let rec_loc = SeekFrom::Start((rec_idx - 1) * rec_len);

                let mut dump = vec![0f64; (4 + 3 * num_bands as usize)];
                file.seek(rec_loc)?;
                file.read_f64_into::<LittleEndian>(&mut dump)?;

                if 0 == ispin {
                    num_plws.push(dump[0] as u64);
                    k_vecs.extend_from_slice(&dump[1..4]);
                }

                let dump = dump[4..].to_vec(); // now dump should be (nbands * 3) long;
                band_eigs.extend(dump.iter().step_by(3));
                band_fweight.extend(dump[2..].iter().step_by(3));
            }
        }

        let k_vecs = Array2::from_shape_vec((num_kpoints as usize, 3usize), k_vecs).unwrap();
        let band_eigs = Array3::from_shape_vec(
            (num_spin as usize, num_kpoints as usize, num_bands as usize), band_eigs).unwrap();
        let band_fweight = Array3::from_shape_vec(
            (num_spin as usize, num_kpoints as usize, num_bands as usize), band_fweight).unwrap();
        Ok((num_plws, k_vecs, band_eigs, band_fweight))
    }

    #[inline]
    fn _calc_record_index(ispin: u64, ikpoint: u64, iband: u64,
                          num_kpoints: u64, num_bands: u64) -> u64 {
        (2 + ispin * num_kpoints * (num_bands + 1) +
            ikpoint * (num_bands + 1) +
            iband + 1
        )
    }

    #[inline]
    fn _calc_record_location(ispin: u64, ikpoint: u64, iband: u64,
                             num_kpoints: u64, num_bands: u64, rec_len: u64) -> SeekFrom {
        SeekFrom::Start(
            Self::_calc_record_index(ispin, ikpoint, iband, num_kpoints, num_bands) * rec_len
        )
    }

    #[inline]
    fn calc_record_location(&self, ispin: u64, ikpoint: u64, iband:u64) -> Result<SeekFrom, WavecarError> {
        self.check_indices(ispin, ikpoint, iband)?;
        Ok(
            Self::_calc_record_location(ispin, ikpoint, iband,
                                        self.num_bands, self.num_bands, self.rec_len)
        )
    }

    #[inline]
    pub fn check_indices(&self, ispin: u64, ikpoint: u64, iband: u64)
                         -> Result<(), WavecarError> {
        if ispin >= self.num_spin {
            return Err(
                WavecarError::from_kind(
                    WavecarErrorKind::SpinIndexOutbound));
        }

        if ikpoint >= self.num_kpoints {
            return Err(
                WavecarError::from_kind(
                    WavecarErrorKind::KPointIndexOutbound));
        }

        if iband >= self.num_bands {
            return Err(
                WavecarError::from_kind(
                    WavecarErrorKind::BandIndexOutbound));
        }
        Ok(())
    }

    #[inline]
    pub fn check_spin_index(&self, ispin: u64) -> Result<(), WavecarError> {
        self.check_indices(ispin, 0, 0)
    }

    #[inline]
    pub fn check_kpoint_index(&self, ikpoint: u64) -> Result<(), WavecarError> {
        self.check_indices(0, ikpoint, 0)
    }

    #[inline]
    pub fn check_band_index(&self, iband: u64) -> Result<(), WavecarError> {
        self.check_indices(0, 0, iband)
    }

    fn _generate_fft_freq(ngrid: u64) -> Vec<i64> {
        // ret = [0 ..= ngrid/2] ++ [(1+ngrid/2-ngrid) ..= -1];
        // eg: ngrid = 11, ret = vec![0, 1, 2, 3, 4, 5, -5, -4, -3, -2, -1];
        let mut ret = Vec::<i64>::new();
        let ngrid = ngrid as i64;
        ret.extend(
            (0 .. (ngrid/2+1)).chain((1+ngrid/2-ngrid) .. 0)
        );
        ret
    }

    fn _generate_fft_grid_general(ngrid: Vec<u64>,
                                  kvec: Array1<f64>,
                                  reci_cell: Array2<f64>,
                                  en_cutoff: f64) -> Vec<Vec<i64>> {
        let reci_cell_t = reci_cell.t();
        let fx = Self::_generate_fft_freq(ngrid[0]);
        let fy = Self::_generate_fft_freq(ngrid[1]);
        let fz = Self::_generate_fft_freq(ngrid[2]);

        fz.iter().flat_map(|&z| {
            let fx = &fx;
            fy.iter().flat_map(move |&y| {
                let fx = &fx;
                fx.into_iter().map(move |&x| vec![x, y, z])
            }) })
            // those lines below equal to
            // for z in fz:
            //   for y in fy:
            //     for x in fx:
            //       gvecs.push_back(vec![x, y, z])
            // k_energy = (G + k)^2 / 2
            // G = gvecs[i], k = kvec in this kpoint
            .filter(|v| {
                // G + k, in fractional coordinate
                let gpk = Array1::from_shape_vec(3, v.to_vec())
                    .unwrap().mapv(|e| e as f64) + &kvec;
                (reci_cell_t.dot(&gpk)).norm().powf(2.0) * PIx2.powf(2.0) * HBAR2D2ME
                    <
                    en_cutoff
            }).collect::<Vec::<Vec::<i64>>>()
    }

    fn _generate_fft_grid_specific(ngrid: Vec<u64>,
                                   kvec: Array1<f64>,
                                   reci_cell: Array2<f64>,
                                   en_cutoff: f64,
                                   vasp_type: VaspType) -> Vec<Vec<i64>> {
        let gvecs = Self::_generate_fft_grid_general(ngrid, kvec, reci_cell, en_cutoff);
        match vasp_type {
            VaspType::Standard |
            VaspType::SpinOrbitCoupling => gvecs,

            VaspType::GammaHalf(GammaHalfDirection::X) => {
                gvecs.into_par_iter().filter(|v|
                    (v[0] > 0) ||
                        (v[0] == 0 && v[1] >  0) ||
                        (v[0] == 0 && v[1] == 0 && v[2] >= 0)
                ).collect()
            },
            VaspType::GammaHalf(GammaHalfDirection::Z) => {
                gvecs.into_par_iter().filter(|v|
                    (v[2] > 0) ||
                        (v[2] == 0 && v[1] >  0) ||
                        (v[2] == 0 && v[1] == 0 && v[0] >= 0)
                ).collect()
            }
        }
    }

    pub(crate) fn generate_fft_grid(&self, ikpoint: u64) -> Vec<Vec<i64>> {
        Self::_generate_fft_grid_specific(
            self.ngrid.clone(),
            self.k_vecs.row(ikpoint as usize).to_owned(),
            self.reci_cell.clone(),
            self.en_cutoff,
            self.vasp_type
        )
    }

    fn _determine_vasp_type(ngrid: Vec<u64>, kvec: Array1<f64>, reci_cell: Array2<f64>,
                            en_cutoff: f64, nplw: u64) -> Result<VaspType, WavecarError> {
        let gvecs = Self::_generate_fft_grid_general(ngrid, kvec, reci_cell, en_cutoff);
        let nplw = nplw as usize;

        if nplw as usize == gvecs.len() {
            Ok(VaspType::Standard)
        } else if nplw == gvecs.len() * 2 {
            Ok(VaspType::SpinOrbitCoupling)
        } else {
            if nplw ==
                // try gamma half x direction, used in vasp 5.4 and higher
                gvecs.par_iter().filter(|v|
                    (v[0] > 0) ||
                        (v[0] == 0 && v[1] >  0) ||
                        (v[0] == 0 && v[1] == 0 && v[2] >= 0)
                ).count() {
                Ok(VaspType::GammaHalf(GammaHalfDirection::X))
                // Sometimes there is no difference in nplws between vasp5.4 and vasp5.3 or lower
                // treat as bug, still have no idea about how to solve it.

            } else if nplw ==
                // try gamma half z direction, used in vasp 5.3 and lower
                gvecs.par_iter().filter(|v|
                    (v[2] > 0) ||
                        (v[2] == 0 && v[1] >  0) ||
                        (v[2] == 0 && v[1] == 0 && v[0] >= 0)
                ).count() {
                Ok(VaspType::GammaHalf(GammaHalfDirection::Z))
            } else {
                Err(
                    WavecarError::from_kind(
                        WavecarErrorKind::UnknownWaverType))
            }
        }
    }

    pub fn read_wavefunction_coeffs(&mut self,
                                    ispin: u64,
                                    ikpoint: u64,
                                    iband: u64) -> Result<Array1<Complex64>, WavecarError> {
        let seek_pos = self.calc_record_location(ispin, ikpoint, iband)?;
        self.file.seek(seek_pos).unwrap();

        let num_plws = self.num_plws[ikpoint as usize] as usize;
        let dump = match self.prec_type {
            WFPrecisionType::Complex32 => {
                let mut ret = vec![0f32; num_plws];
                self.file.read_f32_into::<LittleEndian>(&mut ret).unwrap();
                ret.into_par_iter()
                    .map(|x| x as f64)
                    .collect::<Vec<_>>()
            },
            WFPrecisionType::Complex64 => {
                let mut ret = vec![0f64; num_plws];
                self.file.read_f64_into::<LittleEndian>(&mut ret).unwrap();
                ret
            }
        };

        Ok(
            dump.chunks_exact(2)
                .map(|v| Complex64::new(v[0], v[1]))
                .collect::<Array1::<Complex64>>()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Ordering;
    use std::convert::TryInto;
    use ndarray::{arr1, arr2};

    fn generate_fft_freq_ref(ngrid: u64) -> Vec<i64> {
        let ngrid: i64 = ngrid.try_into().unwrap();
        (0 .. ngrid)
            .map(|x| match x.cmp(&(ngrid/2 + 1)) {
                Ordering::Less => x,
                _ => x - ngrid,
            })
            .collect()
    }

    #[test]
    fn test_generate_fft_freq() {
        for i in 2 ..= 50 {
            assert_eq!(Wavecar::_generate_fft_freq(i as u64), generate_fft_freq_ref(i as u64))
        }
    }

    #[test]
    fn test_generate_fft_grid_general() {
        let kvec = arr1(&[1.0/3.0, 1.0/3.0, 0.0]);
        let ngrid = vec![11u64, 11, 105];
        let reci_cell = arr2(
            &[[0.313971743, 0.181271670, 0.000000000],
                [0.000000000, 0.362543340, 0.000000000],
                [0.000000000, 0.000000000, 0.028571429]]);
        let en_cutoff = 323.36125000000004; // little difference from it in OUTCAR: 323.4

        let res = Wavecar::_generate_fft_grid_general(ngrid, kvec, reci_cell, en_cutoff);
        assert_eq!(res.len(), 3981);
    }

    #[test]
    fn test_determine_vasp_type() {
        let kvec = arr1(&[1.0/3.0, 1.0/3.0, 0.0]);
        let ngrid = vec![11u64, 11, 105];
        let reci_cell = arr2(
            &[[0.313971743, 0.181271670, 0.000000000],
                [0.000000000, 0.362543340, 0.000000000],
                [0.000000000, 0.000000000, 0.028571429]]);
        let en_cutoff = 323.36125000000004; // little difference from it in OUTCAR: 323.4

        let vasp_type = Wavecar::_determine_vasp_type(ngrid.clone(), kvec.clone(),
                                                      reci_cell.clone(), en_cutoff,
                                                      3981).unwrap();
        assert_eq!(VaspType::Standard, vasp_type);

        let vasp_type = Wavecar::_determine_vasp_type(ngrid, kvec, reci_cell, en_cutoff,
                                                      3981 * 2).unwrap();
        assert_eq!(VaspType::SpinOrbitCoupling, vasp_type);
    }
}