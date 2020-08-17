use std::io::{self};
use std::path::PathBuf;

use wavecar_rs::wavecar::Wavecar;
use crate::get_fpath_in_current_dir;

#[cfg(test)]
mod test {
    use super::*;
    use wavecar_rs::wavecar::{WavecarType, GammaHalfDirection};
    // use core::panicking::panic_fmt;

    #[test]
    fn test_read_wavecar() -> io::Result<()> {
        let path = get_fpath_in_current_dir!("WAVECAR");
        let wavecar = Wavecar::from_file(&path)?;
        println!("{:#?}", wavecar);
        Ok(())
    }

    #[test]
    #[should_panic]
    fn test_set_wavecar_type1() {
        let path = get_fpath_in_current_dir!("WAVECAR");
        let mut wavecar = Wavecar::from_file(&path).unwrap();
        wavecar.set_wavecar_type(WavecarType::Standard);
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
