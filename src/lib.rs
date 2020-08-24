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
//! ```text
//! let prec_type = match RTAG {
//!     45200 => Complex32,
//!     45210 => Complex64,
//!     53300 => Err("Unsupported WAVECAR format: VASP5 with f32")
//!     53310 => Err("Unsupported WAVECAR format: VASP5 with f64")
//!         _ => Err("Invalid WAVECAR format: Unknown VASP version")
//! };
//! ```
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
//! Here LATT is stored in row-major, which means in memory, it should be:
//! ```text
//! LATT[0][0], LATT[0][1], LATT[0][2],
//! LATT[1][0], LATT[1][1], LATT[1][2],
//! LATT[2][0], LATT[2][1], LATT[2][2],
//! ```
//!
//! **Note: Meta information or header is stored in float64 for all the WAVECARs. The main wavefunction
//! coefficients are stored in either float32 or float64 determined by `RTAG` in meta information.**
//!
//! # Body
//!
//! The body content starts at the third record. In this part, band eigen value, fermi weight and
//! band coefficients are stored.
//!
//! Here is the structure, from the third record:
//! ```no_run
//! for ispin in 0..NSPIN {
//!     for ikpoint in 0..NKPTS {
//!         // One record here          1)
//!         for iband in 0..NBANDS {
//!             // One record here      2)
//!         }
//!     }
//! }
//! ```
//!
//! 1) There are _4 + 3*NBANDS_ values here:
//!     1) NPLWS -> number of plane waves (number of coefficients) in this k-point, need to be casted
//!         into integer;
//!     2) KVEC -> k-vector for current k-point, three float64 values;
//!     3) The next _3*NBANDS_ values makes up a _NBANDS * 3_ matrix, where first two rows mean the
//!         eigen values for each band and the last row means the fermi-weight (aka fermi occupation);
//!         ```text
//!         band[0].real, band[0].imaginary, fermi_weight[0];
//!         band[1].real, band[1].imaginary, fermi_weight[1];
//!         band[2].real, band[2].imaginary, fermi_weight[2];
//!         ...           ...                ...
//!         ```
//!
//! 2) This part **dominates the WAVECAR**. There are _NBANDS_ records here. Each records contains
//!     _NPLWS_ planewave coefficients, and the type of coefficients is either complexf32 or
//!     complex64, determined by _RTAG_;
//!     ```notrust
//!     coeff[0], coeff[1], coeff[2], ...       // NPLWS values in total
//!     ```
//!
//!
//! This crate only works on little endian machine. You can patch it if you have some feature
//! requests on big endian machine, that's not difficult (changing all the LittleEndian generic
//! parameter to BigEndian shall do the job).

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
