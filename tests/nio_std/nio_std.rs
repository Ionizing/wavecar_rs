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
    // use core::panicking::panic_fmt;

    #[test]
    fn test_read_wavecar() -> io::Result<()> {
        let path = get_fpath_in_current_dir!("WAVECAR");
        let mut wavecar = Wavecar::from_file(&path)?;
        let poscar = Poscar::from_path(&get_fpath_in_current_dir!("POSCAR")).unwrap();

        wavecar.get_wavefunction_in_realspace_default_ngrid(0, 0, 5)
            .unwrap()
            .into_vesta_obj(&poscar)
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
        wavecar.set_wavecar_type(WavecarType::GammaHalf(GammaHalfDirection::X));
    }

    #[test]
    #[should_panic]
    fn test_set_wavecar_type3() {
        let path = get_fpath_in_current_dir!("WAVECAR");
        let mut wavecar = Wavecar::from_file(&path).unwrap();
        wavecar.set_wavecar_type(WavecarType::GammaHalf(GammaHalfDirection::Z));
    }
}
