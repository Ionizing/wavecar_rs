mod nio_std {
    mod nio_std;
}

mod nio_soc {
    mod nio_soc;
}

mod h2o_gam_5_2 {
    mod h2o_gam_5_2;
}

mod h2o_gam_5_4 {
    mod h2o_gam_5_4;
}

#[cfg(test)]
mod test {
    use rayon::prelude::*;
    use ndarray::Array2;

    #[test]
    fn test_rayon_preserve_order() {
        let vec = (0 .. 1<<10).collect::<Vec::<u64>>();

        let result_ref = vec
            .iter()
            .map(|x: &u64| x.pow(3))
            .collect::<Vec<u64>>();

        let result_rayon = vec
            .par_iter()
            .map(|x: &u64| x.pow(3))
            .collect::<Vec<u64>>();

        assert_eq!(result_ref, result_rayon);
    }


    // #[test]
    // fn test_ndarray_raw_data() {
    //     let vec = (0 .. 9).collect::<Vec::<i32>>();
    //     let arr = Array2::<i32>::from_shape_vec((3, 3), vec).unwrap();
    //     println!("{:#?}", &arr);
    //     dbg!(arr.as_slice().unwrap());
    // }
}