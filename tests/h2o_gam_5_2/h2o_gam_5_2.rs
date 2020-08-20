use std::io::{self};
use std::path::PathBuf;

use wavecar_rs::wavecar::Wavecar;
use crate::get_fpath_in_current_dir;

#[cfg(test)]
mod test {
    use super::*;
    use wavecar_rs::wavecar::{WavecarType, GammaHalfDirection};
    use vasp_poscar::Poscar;
    use vaspchg_rs::ChgType;
    use ndarray::s;
    use num::complex::Complex64 as c64;
    // use core::panicking::panic_fmt;

    #[test]
    fn test_basic_properties() -> io::Result<()> {
        let mut wavecar = Wavecar::from_file(&get_fpath_in_current_dir!("WAVECAR"))?;
        wavecar.set_wavecar_type(WavecarType::GammaHalf(GammaHalfDirection::Z));

        let coeffs = wavecar.read_wavefunction_coeffs(0, 0, 0).unwrap();
        assert_eq!(coeffs.len(), 4658);
        assert_eq!(coeffs[0], c64::new(-0.14767155051231384, 0.0));
        assert_eq!(coeffs[4657], c64::new(-0.0003599866759032011, -0.000024752824174356647));

        let fft_grid = wavecar.generate_fft_grid(0);
        assert_eq!(fft_grid.len(), 4658);
        assert_eq!(fft_grid[4657], &[0, -1, 13]);
        assert_eq!(fft_grid[2000], &[9, 7, 4]);

        let wfcr = wavecar.get_wavefunction_in_realspace_default_ngrid(0, 0, 0).unwrap();
        Ok(())
    }

    #[test]
    fn test_wavecar_in_realspace() -> io::Result<()> {
        let path = get_fpath_in_current_dir!("WAVECAR");
        let mut wavecar = Wavecar::from_file(&path)?;
        wavecar.set_wavecar_type(WavecarType::GammaHalf(GammaHalfDirection::X));

        let poscar = Poscar::from_path(&get_fpath_in_current_dir!("POSCAR")).unwrap();
        wavecar.get_wavefunction_in_realspace_default_ngrid(0, 0, 5)
            .unwrap()
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
