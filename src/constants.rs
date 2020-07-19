#![allow(unused)]
#![allow(non_upper_case_globals)]

pub const AVOGADRO: f64 = 6.0221367E23; // Avogadro constant
pub const AU_TO_A: f64 = 0.529177249; // a.u. to angstrom
pub const BOHR_TO_ANG: f64 = 0.529177249; // Bohr radius to angstrom
pub const C_LIGHT: f64 = 137.037; // Light speed in a.u.
pub const CAL_TO_J: f64 = 4.1840; // Calorie in joule
pub const DEBYE: f64 = 3.336E-30; // Coulomb m
pub const EV_TO_J: f64 = 1.60217733E-19; // eV to J
pub const H_PLANCK: f64 = 6.6260755E-34; // Planck constant J s
pub const HATREE_TO_J: f64 = 4.3597482E18; // Hatree to joule
pub const K_B_EV: f64 = 8.6173857E-5; // Boltzmann constant in eV/K
pub const M_ELECT: f64 = 9.10938356E-31; // Mass of electron
pub const M_PROTON: f64 = 1.672621898E-27; // Mass of proton
pub const M_AU: f64 = 1.660539040E-27; // Unit mass in a.u.
pub const PI: f64 = std::f64::consts::PI; // Pi
pub const RY_TO_EV: f64 = 13.605693009; // Rydberg to eV

pub const EV_TO_KCAL: f64 = EV_TO_J * AVOGADRO / 1000.0 / CAL_TO_J;
pub const HBAR: f64 = H_PLANCK / PI / 2.0;
pub const HATREE_TO_KCAL: f64 = HATREE_TO_J * AVOGADRO / 100.0 / CAL_TO_J;
pub const K_B: f64 = K_B_EV * EV_TO_J; // Boltzmann constant in J/K
pub const PIx2: f64 = PI * 2.0; // Pi * 2
pub const HBAR2D2ME: f64 = RY_TO_EV * AU_TO_A * AU_TO_A;

pub use num::complex::Complex64;
pub const IMAGE_UNIT: Complex64 = Complex64::new(0.0, 1.0);
pub const PIx2_COMPLEX: Complex64 = Complex64::new(0.0, PIx2);
