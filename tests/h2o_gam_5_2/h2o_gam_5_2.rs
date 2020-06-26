use std::io::{self};
use std::path::PathBuf;

use wavecar_rs::wavecar::Wavecar;

#[cfg(test)]
mod test {
    use super::*;
    // use core::panicking::panic_fmt;

    fn get_current_dir() -> PathBuf {
        let mut path = PathBuf::from(file!());
        path.pop();
        path
    }

    fn get_fpath_in_current_dit(fname: &str) -> PathBuf {
        let mut path = get_current_dir();
        path.push(fname);
        path
    }

    #[test]
    fn test_read_wavecar() -> io::Result<()> {
        let path = get_fpath_in_current_dit("WAVECAR");
        let wavecar = Wavecar::from_file(&path)?;
        println!("{:#?}", wavecar);
        Ok(())
    }
}