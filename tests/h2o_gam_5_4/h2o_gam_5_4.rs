use std::io::{self};
use std::path::PathBuf;

use num::complex::Complex64 as c64;

use wavecar_rs::{Wavecar, WavecarType, GammaHalfDirection};
use crate::get_fpath_in_current_dir;

#[cfg(test)]
mod test {
    use super::*;
    use vasp_poscar::Poscar;
    use vaspchg_rs::ChgType;
    // use ndarray::s;
    // use core::panicking::panic_fmt;

    #[test]
    fn test_basic_properties() -> io::Result<()> {
        let mut wavecar = Wavecar::from_file(&get_fpath_in_current_dir!("WAVECAR"))?;
        wavecar.set_wavecar_type(WavecarType::GammaHalf(GammaHalfDirection::X));
        let fft_grid = wavecar.generate_fft_grid(0);
        assert_eq!(fft_grid.len(), 4658);
        assert_eq!(fft_grid[2], &[2, 0, 0]);
        assert_eq!(fft_grid[4656], &[11, -1, -1]);

        let coeffs = wavecar.read_wavefunction_coeffs(0, 0, 0).unwrap();
        assert_eq!(coeffs[0], c64::new(0.14767155051231384, 0.0));
        assert_eq!(coeffs[1], c64::new(0.17830640077590942, -0.03066352568566799));
        assert_eq!(coeffs[2], c64::new(0.11648973077535629, -0.04047871753573418));
        assert_eq!(coeffs.shape()[0], 4658);

        let _wfcr = wavecar.get_wavefunction_in_realspace_default_ngrid(0, 0, 0).unwrap();
        // dbg!(wfcr.get_wavefun_realgrid().slice(s![0, 0, ..]));
        Ok(())
    }

    #[test]
    fn test_wavefun_in_realspace() -> io::Result<()> {
        let mut wavecar = Wavecar::from_file(&get_fpath_in_current_dir!("WAVECAR"))?;
        wavecar.set_wavecar_type(WavecarType::GammaHalf(GammaHalfDirection::X));
        let poscar = Poscar::from_path(&get_fpath_in_current_dir!("POSCAR")).unwrap();
        let wr = wavecar.get_wavefunction_in_realspace_default_ngrid(0, 0, 5)
            .unwrap();
        dbg!(wr.get_wavefun_realgrid().sum());
            wr
            .into_parchg_obj(&poscar)
            .write_file(&get_fpath_in_current_dir!("wfc.vasp"), ChgType::Parchg)
            .unwrap();
        Ok(())
    }

    #[test]
    #[should_panic]
    fn test_set_wavecar_type1() {
        let path = get_fpath_in_current_dir!("WAVECAR");
        let mut wavecar = Wavecar::from_file(&path).unwrap();
        wavecar.set_wavecar_type(WavecarType::SpinOrbitCoupling);
    }

    #[test]
    #[should_panic]
    fn test_set_wavecar_type2() {
        let path = get_fpath_in_current_dir!("WAVECAR");
        let mut wavecar = Wavecar::from_file(&path).unwrap();
        wavecar.set_wavecar_type(WavecarType::Standard);
    }

    #[test]
    // #[should_panic]
    fn test_set_wavecar_type3() {
        let path = get_fpath_in_current_dir!("WAVECAR");
        let mut wavecar = Wavecar::from_file(&path).unwrap();
        wavecar.set_wavecar_type(WavecarType::GammaHalf(GammaHalfDirection::X));
    }
}
