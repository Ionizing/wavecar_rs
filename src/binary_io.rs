use byteorder::{LittleEndian, ReadBytesExt};
use ndarray::Array1;
use ndarray::Array2;
use ndarray::Array3;
use std::io::{self, Result};

pub trait ReadArray: io::Read {
    fn read_array_1d_f64(&mut self, len: usize) -> Result<Array1<f64>> {
        let mut ret = Array1::zeros(len);
        self.read_f64_into::<LittleEndian>(ret.as_slice_mut().unwrap())?;
        Ok(ret)
    }

    fn read_array_1d_f32(&mut self, len: usize) -> Result<Array1<f32>> {
        let mut ret = Array1::zeros(len);
        self.read_f32_into::<LittleEndian>(ret.as_slice_mut().unwrap())?;
        Ok(ret)
    }

    #[inline]
    fn read_array_1d_i64(&mut self, len: usize) -> Result<Array1<i64>> {
        let mut ret = Array1::zeros(len);
        self.read_i64_into::<LittleEndian>(ret.as_slice_mut().unwrap())?;
        Ok(ret)
    }

    fn read_array_1d_i32(&mut self, len: usize) -> Result<Array1<i32>> {
        let mut ret = Array1::zeros(len);
        self.read_i32_into::<LittleEndian>(ret.as_slice_mut().unwrap())?;
        Ok(ret)
    }

    fn read_array_2d_f64(&mut self, nrow: usize, ncol: usize) -> Result<Array2<f64>> {
        let mut ret = Array2::zeros((nrow, ncol));
        self.read_f64_into::<LittleEndian>(ret.as_slice_mut().unwrap())?;
        Ok(ret)
    }

    fn read_array_2d_f32(&mut self, nrow: usize, ncol: usize) -> Result<Array2<f32>> {
        let mut ret = Array2::zeros((nrow, ncol));
        self.read_f32_into::<LittleEndian>(ret.as_slice_mut().unwrap())?;
        Ok(ret)
    }

    fn read_array_2d_i64(&mut self, nrow: usize, ncol: usize) -> Result<Array2<i64>> {
        let mut ret = Array2::zeros((nrow, ncol));
        self.read_i64_into::<LittleEndian>(ret.as_slice_mut().unwrap())?;
        Ok(ret)
    }

    fn read_array_2d_i32(&mut self, nrow: usize, ncol: usize) -> Result<Array2<i32>> {
        let mut ret = Array2::zeros((nrow, ncol));
        self.read_i32_into::<LittleEndian>(ret.as_slice_mut().unwrap())?;
        Ok(ret)
    }

    fn read_array_3d_f64(&mut self, ni: usize, nj: usize, nk: usize) -> Result<Array3<f64>> {
        let mut ret = Array3::zeros((ni, nj, nk));
        self.read_f64_into::<LittleEndian>(ret.as_slice_mut().unwrap())?;
        Ok(ret)
    }

    fn read_array_3d_f32(&mut self, ni: usize, nj: usize, nk: usize) -> Result<Array3<f32>> {
        let mut ret = Array3::zeros((ni, nj, nk));
        self.read_f32_into::<LittleEndian>(ret.as_slice_mut().unwrap())?;
        Ok(ret)
    }

    fn read_array_3d_i64(&mut self, ni: usize, nj: usize, nk: usize) -> Result<Array3<i64>> {
        let mut ret = Array3::zeros((ni, nj, nk));
        self.read_i64_into::<LittleEndian>(ret.as_slice_mut().unwrap())?;
        Ok(ret)
    }

    fn read_array_3d_i32(&mut self, ni: usize, nj: usize, nk: usize) -> Result<Array3<i32>> {
        let mut ret = Array3::zeros((ni, nj, nk));
        self.read_i32_into::<LittleEndian>(ret.as_slice_mut().unwrap())?;
        Ok(ret)
    }
}

impl<R: io::Read + ?Sized> ReadArray for R {}
