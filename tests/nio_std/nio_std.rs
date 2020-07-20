use std::io::{self};
use std::path::PathBuf;

use wavecar_rs::wavecar::Wavecar;

#[cfg(test)]
mod test {
    use super::*;
    use wavecar_rs::wavecar::{WavecarType, GammaHalfDirection};
    // use core::panicking::panic_fmt;

    fn get_current_dir() -> PathBuf {
        let mut path = PathBuf::from(file!());
        path.pop();
        path
    }

    fn get_fpath_in_current_dir(fname: &str) -> PathBuf {
        let mut path = get_current_dir();
        path.push(fname);
        path
    }

    #[test]
    fn test_read_wavecar() -> io::Result<()> {
        let path = get_fpath_in_current_dir("WAVECAR");
        let wavecar = Wavecar::from_file(&path)?;
        println!("{:#?}", wavecar);
        Ok(())
    }

    #[test]
    #[should_panic]
    fn test_set_wavecar_type1() {
        let path = get_fpath_in_current_dir("WAVECAR");
        let mut wavecar = Wavecar::from_file(&path).unwrap();
        wavecar.set_wavecar_type(WavecarType::SpinOrbitCoupling);
    }

    #[test]
    #[should_panic]
    fn test_set_wavecar_type2() {
        let path = get_fpath_in_current_dir("WAVECAR");
        let mut wavecar = Wavecar::from_file(&path).unwrap();
        wavecar.set_wavecar_type(WavecarType::GammaHalf(GammaHalfDirection::X));
    }

    #[test]
    #[should_panic]
    fn test_set_wavecar_type3() {
        let path = get_fpath_in_current_dir("WAVECAR");
        let mut wavecar = Wavecar::from_file(&path).unwrap();
        wavecar.set_wavecar_type(WavecarType::GammaHalf(GammaHalfDirection::Z));
    }
}
