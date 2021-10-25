
extern crate rand;
use std::f64;

use crate::utils::{arr_from_vec2};
use crate::argminmax::ArgMinMax;
use crate::io::NormMode;
use crate::io::DispersionMode;
use crate::rank_vector::RankVector;
use crate::rank_vector::Node;
use crate::rank_vector::Stencil;
use crate::io::Parameters;

use ndarray::prelude::*;
use rayon::prelude::*;


#[derive(Debug,Clone,Serialize,Deserialize)]
// #[derive(Debug,Clone)]
pub struct RankMatrix {
    pub meta_vector: Vec<RankVector<Vec<Node>>>,
    pub dimensions: (usize,usize),
    // Features x Samples, not the usual Samples x Features
    dispersion_mode: DispersionMode,
    norm_mode: NormMode,
    split_fraction_regularization: f64,
    standardize: bool,
}


impl RankMatrix {

    pub fn new<'a> (counts: Vec<Vec<f64>>, parameters:&Parameters) -> RankMatrix {

        let meta_vector: Vec<RankVector<Vec<Node>>> = counts
            // .into_iter()
            .into_par_iter()
            .map(|loc_counts|{
                RankVector::<Vec<Node>>::link(&loc_counts)
        }).collect();

        let dim = (meta_vector.len(),meta_vector.get(0).map(|x| x.raw_len()).unwrap_or(0));

        let rm = RankMatrix {
            meta_vector:meta_vector,

            dimensions:dim,

            norm_mode: parameters.norm_mode,
            dispersion_mode: parameters.dispersion_mode,
            split_fraction_regularization: parameters.split_fraction_regularization as f64,
            standardize: parameters.standardize,
        };


        rm

    }

    pub fn from_array(counts: &Array2<f64>,parameters:&Parameters) -> RankMatrix {

        let meta_vector: Vec<RankVector<Vec<Node>>> = counts.axis_iter(Axis(0))
            // .into_iter()
            .into_par_iter()
            .map(|loc_counts|{
                RankVector::<Vec<Node>>::link_array(loc_counts)
        }).collect();

        let dim = (meta_vector.len(),meta_vector.get(0).map(|x| x.raw_len()).unwrap_or(0));

        let rm = RankMatrix {
            meta_vector:meta_vector,

            dimensions:dim,

            norm_mode: parameters.norm_mode,
            dispersion_mode: parameters.dispersion_mode,
            split_fraction_regularization: parameters.split_fraction_regularization as f64,
            standardize: parameters.standardize,
        };


        rm
    }

    pub fn to_array(&self) -> Array2<f64> {
        arr_from_vec2(self.full_values())
    }

    pub fn empty() -> RankMatrix {
        RankMatrix {
            meta_vector:vec![],
            dimensions:(0,0),
            // feature_dictionary: HashMap::with_capacity(0),

            norm_mode: NormMode::L1,
            dispersion_mode: DispersionMode::MAD,
            split_fraction_regularization: 1.,
            standardize: true,
        }

    }

    pub fn means(&self) -> Vec<f64> {
        self.meta_vector.iter().map(|x| x.mean()).collect()
    }

    pub fn medians(&self) -> Vec<f64> {
        self.meta_vector.iter().map(|x| x.median()).collect()
    }

    pub fn fetch_mean(&self,i:usize) -> f64 {
        self.meta_vector[i].mean()
    }

    pub fn dispersions(&self) -> Vec<f64> {
        (0..self.meta_vector.len()).map(|i| self.vec_dispersions(i)).collect()
    }

    pub fn vec_dispersions(&self,index:usize) -> f64 {

        match self.dispersion_mode {
            DispersionMode::Variance => self.meta_vector[index].var(),
            DispersionMode::SSE => self.meta_vector[index].sse(),
            DispersionMode::MAD => self.meta_vector[index].mad(),
            DispersionMode::SSME => self.meta_vector[index].ssme(),
            DispersionMode::SME => self.meta_vector[index].sme(),
            DispersionMode::Entropy => self.meta_vector[index].entropy(),
            DispersionMode::Mixed => panic!("Mixed mode isn't a valid setting for dispersion calculation in individual trees")
        }
    }

    pub fn covs(&self) -> Vec<f64> {
        self.dispersions().into_iter().zip(self.dispersions().into_iter()).map(|x| x.0/x.1).map(|y| if y.is_nan() {0.} else {y}).collect()
    }

    // This method returns the draw order of this input table using local indices, for
    // A certain feature. This is the ranking of each sample in this table by that feature
    // least to greatest

    pub fn sort_by_feature(&self, feature:usize) -> Vec<usize> {
        self.meta_vector[feature].draw_order()
    }

    pub fn rv_fetch(&self,index:usize) -> &RankVector<Vec<Node>> {
        &self.meta_vector[index]
    }

    pub fn feature_fetch(&self, feature: usize, sample: usize) -> f64 {
        self.meta_vector[feature].fetch(sample)
    }

    pub fn full_values(&self) -> Vec<Vec<f64>> {
        let mut values = Vec::new();
        for feature in &self.meta_vector {
            values.push(feature.full_values().cloned().collect::<Vec<f64>>());
        }
        values
    }

    pub fn full_feature_values(&self,index:usize) -> Vec<f64> {
        self.meta_vector[index].full_values().cloned().collect::<Vec<f64>>()
    }

    pub fn full_ordered_values(&self) -> Vec<Vec<f64>> {
        self.meta_vector.iter().map(|x| x.ordered_values()).collect()
    }

    pub fn dispersion_mode(&self) -> DispersionMode {
        self.dispersion_mode
    }

    pub fn set_dispersion_mode(&mut self, dispersion_mode: DispersionMode) {
        self.dispersion_mode = dispersion_mode;
    }

    pub fn derive(&self, samples:&[usize]) -> RankMatrix {

        let dummy_features: Vec<usize> = (0..self.meta_vector.len()).collect();

        self.derive_specified(&dummy_features[..], samples)

    }

    pub fn derive_specified(&self, features:&[usize],samples:&[usize]) -> RankMatrix {

        // Stencil derivation allows to quickly derive vectors with exactly repeating values. See rank vector for full logic.

        let sample_stencil = Stencil::from_slice(samples);

        let new_meta_vector: Vec<RankVector<Vec<Node>>> = features.iter().map(|i| self.meta_vector[*i].derive_stencil(&sample_stencil)).collect();

        let dimensions = (new_meta_vector.len(),new_meta_vector.get(0).map(|x| x.raw_len()).unwrap_or(0));

        RankMatrix {

            meta_vector: new_meta_vector,
            dimensions: dimensions,
            norm_mode: self.norm_mode,
            dispersion_mode: self.dispersion_mode,
            split_fraction_regularization: self.split_fraction_regularization,
            standardize: self.standardize,
        }

    }



    pub fn order_dispersions(&self,draw_order:&[usize]) -> Array1<f64> {

        let mut dispersions: Array1<f64> = Array1::zeros(draw_order.len()+1);

        let mut worker_vec = RankVector::empty_sv();

        for v in self.meta_vector.iter() {
            worker_vec.clone_from_prototype(v);

            let standardization = if self.standardize {
                let raw = worker_vec.dispersion(self.dispersion_mode);
                if raw.abs() > 0.0000000001 {
                    raw
                }
                else {1.0}
            }
            else {1.0};

            match self.norm_mode {
                // We don't want to check our norm at each interation so we choose one of two loops here, even though most of the code is redundant.

                NormMode::L1 => {
                    for (i,draw) in draw_order.iter().enumerate() {
                        let regularization = (worker_vec.len() as f64 / draw_order.len() as f64).powf(self.split_fraction_regularization);
                        dispersions[i] += worker_vec.dispersion(self.dispersion_mode) * regularization / standardization;
                        worker_vec.pop(*draw);
                    }
                }
                NormMode::L2 => {
                    for (i,draw) in draw_order.iter().enumerate() {
                        let regularization = (worker_vec.len() as f64 / draw_order.len() as f64).powf(self.split_fraction_regularization);
                        dispersions[i] += (worker_vec.dispersion(self.dispersion_mode) * regularization / standardization).powi(2);
                        worker_vec.pop(*draw);
                    }
                }
            }
        }

        // We operate over the same features but in reverse sample order

        for v in self.meta_vector.iter() {
            worker_vec.clone_from_prototype(v);

            let standardization = if self.standardize {
                let raw = worker_vec.dispersion(self.dispersion_mode);
                if raw.abs() > 0.0000000001 {
                    raw
                }
                else {1.0}
            }
            else {1.0};


            match self.norm_mode {
                NormMode::L1 => {
                    for (i,draw) in draw_order.iter().enumerate().rev() {
                        let regularization = (worker_vec.len() as f64 / draw_order.len() as f64).powf(self.split_fraction_regularization);
                        // i+1 is important here because the first and last values are those where no sample was drawn or all samples were drawn, thus we need to offset the forward and reverse.
                        dispersions[i+1] += worker_vec.dispersion(self.dispersion_mode) * regularization / standardization;
                        worker_vec.pop(*draw);
                    }
                }
                NormMode::L2 => {
                    for (i,draw) in draw_order.iter().enumerate().rev() {
                        let regularization = (worker_vec.len() as f64 / draw_order.len() as f64).powf(self.split_fraction_regularization);
                        dispersions[i+1] += (worker_vec.dispersion(self.dispersion_mode) * regularization / standardization).powi(2);
                        worker_vec.pop(*draw);
                    }
                }
            }
        }

        dispersions

    }




    pub fn split_candidates(input_matrix:RankMatrix,output_matrix:RankMatrix) -> Vec<(usize,usize,f64)> {


        let draw_orders: Vec<Vec<usize>> = input_matrix.meta_vector.iter().map(|mv| mv.draw_order()).collect();

        let mut minima: Vec<(usize,usize,f64)> =
            draw_orders
                // .into_iter()
                .into_par_iter()
                .enumerate()
                .flat_map(|(i,draw_order)| {
                    let ordered_dispersions = output_matrix.order_dispersions(&draw_order);
                    let (local_index,dispersion) = ArgMinMax::argmin_v(ordered_dispersions.iter().skip(1))?;
                    Some((i,draw_order[local_index],*dispersion))
                })
                .collect();



        minima.sort_by(|&a,&b| (a.2).partial_cmp(&b.2).unwrap());

        for triplet in minima.iter_mut() {
            triplet.2 = input_matrix.feature_fetch(triplet.0,triplet.1);
        }

        minima
    }


}






#[cfg(test)]
mod rank_matrix_tests {

    use super::*;
    use crate::utils::test_utils::{iris};
    use crate::utils::slow_mad;

    fn blank_parameter() -> Parameters {
        let parameters = Parameters::empty();
        parameters
    }

    #[test]
    fn rank_matrix_general_test() {
        let parameters = blank_parameter();
        let table = RankMatrix::new(vec![vec![1.,2.,3.],vec![4.,5.,6.],vec![7.,8.,9.]],&parameters);
        assert_eq!(table.medians(),vec![2.,5.,8.]);

        // This assertion tests a default dispersion of SSME
        assert_eq!(table.dispersions(),vec![2.,2.,2.]);

        // This assertion tests a default dispersion of Variance
        // assert_eq!(table.dispersions(),vec![1.,1.,1.]);
    }

    #[test]
    fn rank_matrix_trivial_test() {
        let parameters = Parameters::empty();
        let mtx = RankMatrix::new(Vec::new(),&parameters);
        let empty: Vec<f64> = Vec::new();
        assert_eq!(mtx.medians(),empty);
        assert_eq!(mtx.dispersions(),empty);
    }

    #[test]
    pub fn rank_matrix_simple_test() {
        let parameters = blank_parameter();
        let mtx = RankMatrix::new(vec![vec![10.,-3.,0.,5.,-2.,-1.,15.,20.]],&parameters);
        let draw_order = mtx.sort_by_feature(0);
        println!("{:?}",draw_order);
        let mad_order = mtx.meta_vector[0].clone().ordered_meds_mads(&draw_order);
        assert_eq!(mad_order, vec![(2.5,5.),(5.0,6.0),(7.5,7.5),(10.,5.),(12.5,5.),(15.,5.),(17.5,2.5),(20.,0.),(0.0,0.0)]);
    }

    #[test]
    pub fn rank_matrix_test_ordered_ssme() {
        let mut parameters = blank_parameter();
        parameters.dispersion_mode = DispersionMode::SSME;
        parameters.norm_mode = NormMode::L1;
        parameters.split_fraction_regularization = 0.;
        let mtx = RankMatrix::new(vec![vec![10.,-3.,0.,5.,-2.,-1.,15.,20.]],&parameters);
        let draw_order = mtx.sort_by_feature(0);

        let order_dispersions = mtx.order_dispersions(&draw_order);
        assert_eq!(order_dispersions,array![594.,460.0, 354.0, 252.0, 130.0, 92.0, 162.0, 364.0,594.]);
    }

// ,-3.,-2.,-1.,0.,5.,10.,15.,20.

    #[test]
    pub fn rank_matrix_test_ordered_mad() {
        let mut parameters = blank_parameter();
        parameters.dispersion_mode = DispersionMode::MAD;
        parameters.norm_mode = NormMode::L1;
        parameters.split_fraction_regularization = 1.;
        let simple = vec![vec![10.,-3.,0.,5.,-2.,-1.,15.,20.]];
        let mtx = RankMatrix::new(simple.clone(),&parameters);
        let draw_order = mtx.sort_by_feature(0);

        let order_dispersions = mtx.order_dispersions(&draw_order);

        let correct = vec![5., 5.25, 5.75, 3.5, 3., 2.5,2.125, 2.625, 5.];

        assert_eq!(order_dispersions.to_vec(),correct);
    }


    #[test]
    pub fn rank_matrix_derive_test() {
        let parameters = blank_parameter();
        let mtx = RankMatrix::new(vec![vec![10.,-3.,0.,5.,-2.,-1.,15.,20.]],&parameters);
        let kid1 = mtx.derive(&vec![0,2,4,6]);
        let kid2 = mtx.derive(&vec![1,3,5,7]);
        println!("{:?}",kid1);
        println!("{:?}",kid2);
        assert_eq!(kid1.medians(),vec![5.]);
        assert_eq!(kid2.medians(),vec![2.]);

        // These assertions test ssme as a dispersion
        assert_eq!(kid1.dispersions(),vec![199.]);
        assert_eq!(kid2.dispersions(),vec![367.]);

        // // These assertions test variance as a dispersion
        // assert_eq!(kid1.dispersions(),vec![5.]);
        // assert_eq!(kid2.dispersions(),vec![4.]);
    }

    #[test]
    pub fn rank_matrix_derive_feature_twice() {
        let parameters = blank_parameter();
        let mtx = RankMatrix::new(vec![vec![10.,-3.,0.,5.,-2.,-1.,15.,20.]],&parameters);
        let kid = mtx.derive_specified(&vec![0,0],&vec![0,2,4,6]);
        println!("{:?}",kid);
        assert_eq!(kid.medians(),vec![5.,5.]);

        // These assertions test ssme as a dispersion
        assert_eq!(kid.dispersions(),vec![199.,199.]);

        // // These assertions test variance as a dispersion
        // assert_eq!(kid1.dispersions(),vec![5.]);
        // assert_eq!(kid2.dispersions(),vec![4.]);
    }

    #[test]
    pub fn rank_matrix_derive_sample_twice() {
        let parameters = blank_parameter();
        let mtx = RankMatrix::new(vec![vec![10.,-3.,0.,5.,-2.,-1.,15.,20.]],&parameters);
        let kid = mtx.derive_specified(&vec![0],&vec![0,2,4,6,6]);
        println!("{:?}",kid);
        assert_eq!(kid.medians(),vec![10.]);

        // These assertions test ssme as a dispersion
        assert_eq!(kid.dispersions(),vec![294.]);

        // // These assertions test variance as a dispersion
        // assert_eq!(kid1.dispersions(),vec![5.]);
        // assert_eq!(kid2.dispersions(),vec![4.]);
    }

    #[test]
    pub fn rank_matrix_derive_empty_test() {
        let parameters = blank_parameter();
        let mtx = RankMatrix::new(vec![vec![10.,-3.,0.,5.,-2.,-1.,15.,20.],vec![0.,1.,0.,1.,0.,1.,0.,1.]],&parameters);
        let _kid1 = mtx.derive(&vec![0,2,4,6]);
        let _kid2 = mtx.derive(&vec![1,3,5,7]);

    }

    #[test]
    pub fn rank_matrix_full_ssme() {
        let mut parameters = Parameters::empty();
        parameters.dispersion_mode = DispersionMode::SSME;
        parameters.standardize = false;
        parameters.split_fraction_regularization = 0.;
        let iris_matrix = RankMatrix::from_array(&iris().t().to_owned(),&parameters);
        let ordered = iris_matrix.order_dispersions(&(0..150).collect::<Vec<usize>>());
        for (i,f) in ordered.axis_iter(Axis(0)).enumerate() {
            eprintln!("{:?}",i);
            eprintln!("{:?}",f);
        }
        // panic!();
    }

}
