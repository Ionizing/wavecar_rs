use crate::wavecar::WavecarType;
use std::error::Error;
use std::fmt;

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
        Self { repr: error }
    }
}

#[derive(Copy, Clone, Debug, PartialOrd, PartialEq)]
pub enum ErrorKind {
    SpinIndexOutbound,
    KPointIndexOutbound,
    BandIndexOutbound,
    UnknownWavecarType,
    UnmatchedWavecarType(WavecarType, WavecarType),
}

impl Error for WavecarError {}

impl fmt::Display for WavecarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let description: String = match self.kind() {
            ErrorKind::SpinIndexOutbound => "Spin index outbound".into(),
            ErrorKind::KPointIndexOutbound => "K point index outbound".into(),
            ErrorKind::BandIndexOutbound => "Band index outboud".into(),
            ErrorKind::UnknownWavecarType => {
                "Unknown WAVECAR type, which should be among std, gam or ncl".into()
            }
            ErrorKind::UnmatchedWavecarType(t1, t2) => format!(
                "WAVECAR type <{}> differs from user's input <{}>",
                t1.to_string(),
                t2.to_string()
            ),
        };
        write!(f, "WavecarIndexError/{:?}: {}", self.kind(), description)
    }
}

impl fmt::Debug for WavecarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}
