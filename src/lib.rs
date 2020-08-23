// #![feature(test)]

//! # Overview
//! Here is the [description](http://www.andrew.cmu.edu/user/feenstra/wavetrans/) of the WAVECAR
//! file structure.
//!
//! ```no
//! Record-length #spin components RTAG(a value specifying the precision)
//! #k-points #bands ENCUT(maximum energy for plane waves)
//! LatVec-A
//! LatVec-B
//! LatVec-C
//! Loop over spin
//!    Loop over k-points
//!       #plane waves, k vector
//!       Loop over bands
//!          band energy, band occupation
//!       End loop over bands
//!       Loop over bands
//!          Loop over plane waves
//!             Plane-wave coefficient
//!          End loop over plane waves
//!       End loop over bands
//!    End loop over k-points
//! End loop over spin
//! ```
//!
//! # Meta information
//! The meta information contains two records (see the record concept in fortran's
//! [unformatted io action](https://docs.oracle.com/cd/E19957-01/806-3593/2_io.html)).
//!
//! The first record only contains
//! ```no
//! RECL, NSPIN, RTAG
//! ```
//! corresponding to record-length, number of spin components, precision tag. Though these three
//! values are integer, they are stored in float64.
//!
//! Here is how we treat RTAG:
//! ```no_run
//! match prec_tag {
//!     45200 => Complex32,
//!     45210 => Complex64,
//!     53300 => {
//!         return Err(io::Error::new(
//!             IoErrorKind::Other, "Unsupported WAVECAR format: VASP5 with f32"));
//!     },
//!     53310 => {
//!         return Err(io::Error::new(
//!             IoErrorKind::Other, "Unsupported WAVECAR format: VASP5 with f64"));
//!     },
//!     _ => {
//!         return Err(io::Error::new(
//!             IoErrorKind::Other, "Invalid WAVECAR format: Unknown VASP version"));
//!     }
//! };
//! ```
//!
//!
//!
//! The second record contains some more information:
//! ```no
//! NKPTS, NBANDS, ENCUT, LATT, EFERMI
//! ```
//! corresponding to number k-points, number of bands in each k-point, energy cutoff, lattice
//! vectors in real space (3x3 matrix), fermi energy (in vasp 5 and higher). NKPTS and NBANDS are
//! casted from float64 to integer when reading. ENCUT, LATT and EFERMI are originally float64 value
//! or matrix.
//!
//! **Note: Meta information or header is stored in float64 for all the WAVECARs. The main wavefunction
//! coefficients are stored in either float32 or float64 determined by `RTAG` in meta information.**
//!
//! # Body
//!
//!
//! This crate only works on little endian machine. You can patch it if you have some requests on
//! big endian machine, that's not difficult (chang all the LittleEndian generic parameter to
//! BigEndian shall do the job).

pub use error::WavecarError;
pub use wavecar::Wavecar;
pub use wavecar::WavecarType;
pub use wavecar::WFPrecisionType;
pub use wavecar::GammaHalfDirection;
pub use wavefunction::Wavefunction;

mod error;
mod wavecar;
mod wavefunction;
mod constants;
mod binary_io;

mod fft;
mod utils;
