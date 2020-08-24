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
//! ```text
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
//! To sum up, the number of total records in WAVECAR should be `2 + NSPIN * NKPTS * (1 + NBANDS)`.
//!
//! However, for the calculations that enables spin orbits coupling correction, `NSPIN = 1`, but on
//! each band, two spinor components (equal length) are stored, where the upper and the lower is
//! are placed in order.
//! ```text
//! // on each band, spinor up
//! coeffs[0], coeffs[1], ...
//! //               spinor down
//! coeffs[0], coeffs[1], ...
//! ```
//!
//! **This crate only works on little endian machine. You can patch it if you have some feature
//! requests on big endian machine, that's not difficult (changing all the LittleEndian generic
//! parameter to BigEndian shall do the job).**
//!
//!
//! # Implementation of transformation of wavefunction in k-space into real-space
//! We've got the band coefficients, but how to use it? What should we do if we want to visualize it in
//! real space?  The relation between k-space and real space is the Fourier transformation. But in which
//! order is the coefficients placed?
//!
//! ## FFT grid generation for standard & SOC system
//! Wavefunction in k-space is a 3D grid, but we must start with 1D to illustrate how the grid is
//! generated:
//!
//! Suppose we have a 1D grid whose size is _ngrid_, the 1D fft grid should be
//! ```text
//! [0, 1, 2, ... ngrid/2] ++ [(1+ngrid/2-ngrid), (ngrid/2-ngrid), ... -1]
//! e.g. when ngrid = 11
//! ret = [0, 1, 2, 3, 4, 5, -5, -4, -3, -2, -1]
//! ```
//!
//! And for 3D  grid, three directions does the same job
//! ```text
//! fx = generate_grid(ngrid[0])
//! fy = generate_grid(ngrid[1])
//! fz = generate_grid(ngrid[2])
//! ```
//!
//! Then combine them (in Fortran, the inner index is the fastest one)
//! ```text
//! for ifz in fz {
//! for ify in fy {
//! for ifx in fx {
//!     fft_grid += [ifx, ify, ifz]
//! }}}
//! ```
//! Each coordinate `[ifx, ify, ifz]` in the above will be named with `G` in the following.
//!
//! Now we've a cube in k-space. However the valid wavefunction grid should be a sphere where the
//! radius is determined by the formula `(G + k)^2 / 2 < ENCUT`, where `k` is the k-vector of current
//! k-point.
//!
//! We've rubbed the k-space cube into a k-space sphere, then the coefficients from WAVECAR can be
//! placed into the grid in correspondence.
//!
//! Then the arrangement is done for standard and SOC systems. As for the gamma only system, the
//! arrangement is somewhat more complicate.
//!
//! ## FFT grid generation for gamma only system
//!
//! First we need to perform a standard FFT grid generation, then filter the G points, i.e. cut the
//! sphere and remove half of it. For gamma half of `x` direction:
//! ```text
//! fft_grid.iter()
//!     .filter(|[gx, gy, gz]| {
//!     (gx > 0) ||
//!     ((gx == 0) && (gy > 0)) ||
//!     ((gx == 0) && (gy == 0) && (gz >= 0)
//! })
//! ```
//! And for 'z' direction:
//! ```text
//! fft_grid.iter()
//!     .filter(|[gx, gy, gz]| {
//!     (gz > 0) ||
//!     ((gz == 0) && (gy > 0)) ||
//!     ((gz == 0) && (gy == 0) && (gx >= 0))
//! })
//! ```
//!
//! ## Reverse Fourier transformation
//!
//!

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
