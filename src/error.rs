use std::fmt;
use std::error::Error;

#[derive(Clone)]
pub struct WavecarError {
    repr: ErrorKind,
}

impl WavecarError {
    #[inline]
    pub fn kind(&self) -> ErrorKind {
        self.repr
    }

    pub fn from_kind(error: ErrorKind) -> Self {
        Self{repr: error}
    }
}

#[derive(Copy, Clone, Debug)]
pub enum ErrorKind {
    SpinIndexOutbound = 1,
    KPointIndexOutbound,
    BandIndexOutbound,
    UnkownWavecarType,
}

impl PartialEq for ErrorKind {
    #[inline(always)]
    fn eq(&self, rhs: &Self) -> bool {
        *self as u8 == *rhs as u8
    }
}

impl Error for WavecarError { }

impl fmt::Display for WavecarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let description = match self.kind() {
            ErrorKind::SpinIndexOutbound => "Spin index outbound",
            ErrorKind::KPointIndexOutbound => "K point index outbound",
            ErrorKind::BandIndexOutbound => "Band index outboud",
            ErrorKind::UnkownWavecarType => "Unknown WAVECAR type, which should be among std, gam or ncl"
        };
        write!(f, "WavecarIndexError/{:?}: {}", self.kind(), description)
    }
}

impl fmt::Debug for WavecarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}
