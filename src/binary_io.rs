use std::io::{
    self,
    Result,
};
use ndarray::Array1;
use ndarray::Array2;
use ndarray::Array3;
use byteorder::{
    LittleEndian,
    ReadBytesExt,
};

pub trait ReadArray: io::Read {
    fn read_array_1d_f64(&mut self, len: usize)
                         -> Result<Array1<f64>> {
        let mut buf = vec![0f64; len];
        let _ = self.read_f64_into::<LittleEndian>(&mut buf).unwrap();
        let ret = Array1::from_shape_vec(len, buf).unwrap();
        Ok(ret)
    }

    fn read_array_1d_f32(&mut self, len: usize)
                         -> Result<Array1<f32>> {
        let mut buf = vec![0f32; len];
        let _ = self.read_f32_into::<LittleEndian>(&mut buf).unwrap();
        let ret = Array1::from_shape_vec(len, buf).unwrap();
        Ok(ret)
    }

    #[inline]
    fn read_array_1d_i64(&mut self, len: usize)
                         -> Result<Array1<i64>> {
        let mut buf = vec![0i64; len];
        let _ = self.read_i64_into::<LittleEndian>(&mut buf).unwrap();
        let ret = Array1::from_shape_vec(len, buf).unwrap();
        Ok(ret)
    }

    fn read_array_1d_i32(&mut self, len: usize)
                         -> Result<Array1<i32>> {
        let mut buf = vec![0i32; len];
        let _ = self.read_i32_into::<LittleEndian>(&mut buf).unwrap();
        let ret = Array1::from_shape_vec(len, buf).unwrap();
        Ok(ret)
    }



    fn read_array_2d_f64(&mut self, nrow: usize, ncol: usize)
            -> Result<Array2<f64>> {
        let mut buf = vec![0f64; nrow * ncol];
        let _ = self.read_f64_into::<LittleEndian>(&mut buf).unwrap();
        let ret = Array2::from_shape_vec((nrow, ncol), buf).unwrap();
        Ok(ret)
    }

    fn read_array_2d_f32(&mut self, nrow: usize, ncol: usize)
                         -> Result<Array2<f32>> {
        let mut buf = vec![0f32; nrow * ncol];
        let _ = self.read_f32_into::<LittleEndian>(&mut buf).unwrap();
        let ret = Array2::from_shape_vec((nrow, ncol), buf).unwrap();
        Ok(ret)
    }

    fn read_array_2d_i64(&mut self, nrow: usize, ncol: usize)
                         -> Result<Array2<i64>> {
        let mut buf = vec![0i64; nrow * ncol];
        let _ = self.read_i64_into::<LittleEndian>(&mut buf).unwrap();
        let ret = Array2::from_shape_vec((nrow, ncol), buf).unwrap();
        Ok(ret)
    }

    fn read_array_2d_i32(&mut self, nrow: usize, ncol: usize)
                         -> Result<Array2<i32>> {
        let mut buf = vec![0i32; nrow * ncol];
        let _ = self.read_i32_into::<LittleEndian>(&mut buf).unwrap();
        let ret = Array2::from_shape_vec((nrow, ncol), buf).unwrap();
        Ok(ret)
    }



    fn read_array_3d_f64(&mut self, ni: usize, nj: usize, nk: usize)
                         -> Result<Array3<f64>> {
        let mut buf = vec![0f64; ni * nj * nk];
        let _ = self.read_f64_into::<LittleEndian>(&mut buf).unwrap();
        let ret = Array3::from_shape_vec((ni, nj, nk), buf).unwrap();
        Ok(ret)
    }

    fn read_array_3d_f32(&mut self, ni: usize, nj: usize, nk: usize)
                         -> Result<Array3<f32>> {
        let mut buf = vec![0f32; ni * nj * nk];
        let _ = self.read_f32_into::<LittleEndian>(&mut buf).unwrap();
        let ret = Array3::from_shape_vec((ni, nj, nk), buf).unwrap();
        Ok(ret)
    }

    fn read_array_3d_i64(&mut self, ni: usize, nj: usize, nk: usize)
                         -> Result<Array3<i64>> {
        let mut buf = vec![0i64; ni * nj * nk];
        let _ = self.read_i64_into::<LittleEndian>(&mut buf).unwrap();
        let ret = Array3::from_shape_vec((ni, nj, nk), buf).unwrap();
        Ok(ret)
    }

    fn read_array_3d_i32(&mut self, ni: usize, nj: usize, nk: usize)
                         -> Result<Array3<i32>> {
        let mut buf = vec![0i32; ni * nj * nk];
        let _ = self.read_i32_into::<LittleEndian>(&mut buf).unwrap();
        let ret = Array3::from_shape_vec((ni, nj, nk), buf).unwrap();
        Ok(ret)
    }
}

impl <R: io::Read + ?Sized> ReadArray for R {}
