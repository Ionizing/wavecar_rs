mod nio_std { mod nio_std; }
mod nio_soc { mod nio_soc; }
mod h2o_gam_5_2 { mod h2o_gam_5_2; }
mod h2o_gam_5_4 { mod h2o_gam_5_4; }

// #[cfg(test)]
// mod test {
//     use rayon::prelude::*;
//     // use ndarray::Array2;
//
//     #[test]
//     fn test_rayon_preserve_order() {
//         let vec = (0..1 << 10).collect::<Vec<u64>>();
//
//         let result_ref = vec.iter().map(|x: &u64| x.pow(3)).collect::<Vec<u64>>();
//
//         let result_rayon = vec.par_iter().map(|x: &u64| x.pow(3)).collect::<Vec<u64>>();
//
//         assert_eq!(result_ref, result_rayon);
//     }
//
// }

#[macro_export]
macro_rules! get_fpath_in_current_dir {
    ($fname:expr) => {{
        let mut path = PathBuf::from(file!());
        path.pop();
        path.push($fname);
        path
    }}
}