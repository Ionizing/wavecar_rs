#![allow(unused)]
#![allow(non_upper_case_globals)]

//! Some physical constants used in VASP
//!
//! Note: A to B means _A / B = conversion ratio_

/// Avogadro constant
pub const AVOGADRO: f64 = 6.0221367E23;

/// Atom unit to angstrom
pub const AU_TO_A: f64 = 0.529177249;

/// Bohr radius to angstrom
pub const BOHR_TO_ANG: f64 = 0.529177249;

/// Light speed in a.u.
pub const C_LIGHT: f64 = 137.037;

/// Calorie to joule
pub const CAL_TO_J: f64 = 4.1840;

/// Debye unit, in Coulomb*meter
pub const DEBYE: f64 = 3.336E-30;

/// eV to joule
pub const EV_TO_J: f64 = 1.60217733E-19;

/// Planck constant, in J*s
pub const H_PLANCK: f64 = 6.6260755E-34;

/// Hatree to joule
pub const HATREE_TO_J: f64 = 4.3597482E18;

/// Boltzmann constant, in eV/K
pub const K_B_EV: f64 = 8.6173857E-5;

/// Mass of electron, in kilogram
pub const M_ELECT: f64 = 9.10938356E-31;

/// Mass of proton, in kilogram
pub const M_PROTON: f64 = 1.672621898E-27;

/// Atomic unit mass to kilogram
pub const M_AU: f64 = 1.660539040E-27;

/// Pi
pub const PI: f64 = std::f64::consts::PI;

/// Rydberg to eV
pub const RY_TO_EV: f64 = 13.605693009;


/// eV to kilo calories
pub const EV_TO_KCAL: f64 = EV_TO_J * AVOGADRO / 1000.0 / CAL_TO_J;

/// hbar = h / 2pi
pub const HBAR: f64 = H_PLANCK / PI / 2.0;

/// hatree to kilo calories
pub const HATREE_TO_KCAL: f64 = HATREE_TO_J * AVOGADRO / 100.0 / CAL_TO_J;

/// Boltzmann constant, in J/K
pub const K_B: f64 = K_B_EV * EV_TO_J;

/// Pi * 2
pub const PIx2: f64 = PI * 2.0;

/// hbar<sup>2</sup> / m<sub>e</sub><sup>2</sup>
pub const HBAR2D2ME: f64 = RY_TO_EV * AU_TO_A * AU_TO_A;

use num::complex::Complex64;

/// Image unit of complex number, = 1i
pub const IMAGE_UNIT: Complex64 = Complex64::new(0.0, 1.0);

/// 2Pi * _i_
pub const PIx2_COMPLEX: Complex64 = Complex64::new(0.0, PIx2);
