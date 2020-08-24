# wavecar_rs

![Rust](https://github.com/Ionizing/wavecar_rs/workflows/Rust/badge.svg)

A crate to manipulate wavefunction from VASP WAVECAR.

# Example

``` rust
use wavecar_rs::*;
use vaspchg_rs::ChgType;
use vasp_poscar::Poscar;

fn main() -> io::Result<()> {
    let mut wavecar = Wavecar::from_file("WAVECAR")?;
    // if your calculation is done via vasp5.2.x or lower in parallel
    // you need to set the wavecar type
    // wavecar.set_wavecar_type(WavecarType::GammaHalf(GammaHalfDirection::Z));

    let poscar = Poscar::from_path("POSCAR").unwrap();
    wavecar.get_wavefunction_in_realspace_default_grid(0, 0, 5)
        .unwrap()
        .apply_phase(&[0.5f64, 0.5, 0.5])
        .into_vesta_obj(&poscar)
        .write_file("wfc_xx.vasp", ChgType::Parchg)
        .unwrap();
    Ok(())
}
```


# Features

- [X] Get the meta information of WAVECAR;
- [X] Transform wavefunction from k-spake into real-space;
- [X] Save the wavefunction as CHGCAR format to visualize the spatial distribution;
- [X] Apply phase on the wavefunction to get full bloch waves;


# Acknowledgement

- [Qijing Zheng](https://github.com/QijingZheng/VaspBandUnfolding/blob/master/vaspwfc.py);
- [ExpHP](https://github.com/ExpHP/vasp-poscar);
- Other guys from [the group](https://t.me/rust_zh).
