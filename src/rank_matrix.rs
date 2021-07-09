
use std::collections::HashSet;
use std::collections::HashMap;
use std::cmp::Ordering;
use std::sync::Arc;
use std::sync::mpsc;
extern crate rand;
use std::f64;

use crate::Feature;
use crate::Sample;
use crate::utils::{l1_sum,l2_sum,vec2_from_arr,arr_from_vec2,ArgMinMax,ArgMinMaxII,ArgSort,ArgSortII};
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

        let mut meta_vector: Vec<RankVector<Vec<Node>>> = counts
            .into_par_iter()
            .enumerate()
            .map(|(i,loc_counts)|{
                // if i%200 == 0 {
                //     // println!("Initializing: {}",i);
                // }
                RankVector::<Vec<Node>>::link(loc_counts)
        }).collect();

        let dim = (meta_vector.len(),meta_vector.get(0).map(|x| x.raw_len()).unwrap_or(0));

        // println!("Made rank table with {} features, {} samples:", dim.0,dim.1);

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
        RankMatrix::new(vec2_from_arr(counts),parameters)
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

        let mut new_meta_vector: Vec<Arc<RankVector<Vec<Node>>>> = Vec::with_capacity(features.len());

        let sample_stencil = Stencil::from_slice(samples);

        let mut new_meta_vector: Vec<RankVector<Vec<Node>>> = features.iter().map(|i| self.meta_vector[*i].derive_stencil(&sample_stencil)).collect();

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

    pub fn order_dispersions(&self,draw_order:&[usize],feature_weights:&Array1<f64>) -> Array1<f64> {
        let full_dispersion = self.full_dispersion(draw_order);

        let normed = match self.norm_mode {
            NormMode::L1 => { l1_sum(&full_dispersion,feature_weights)},
            NormMode::L2 => { l2_sum(&full_dispersion,feature_weights)},
        };

        normed
    }

    pub fn full_dispersion(&self,draw_order:&[usize]) -> Array2<f64> {


        let mut forward_dispersions: Array2<f64> = Array2::zeros((draw_order.len()+1,self.dimensions.0));
        let mut reverse_dispersions: Array2<f64> = Array2::zeros((draw_order.len()+1,self.dimensions.0));

        let mut worker_vec = RankVector::empty_sv();

        for (i,v) in self.meta_vector.iter().enumerate() {
            worker_vec.clone_from_prototype(v);
            let fd = worker_vec.ordered_dispersion(draw_order,self.dispersion_mode);
            for (j,fr) in fd.into_iter().enumerate() {
                forward_dispersions[(j,i)] = fr;
            }
        }

        let mut reverse_draw_order:Vec<usize> = draw_order.to_owned();
        reverse_draw_order.reverse();

        for (i,v) in self.meta_vector.iter().enumerate() {
            worker_vec.clone_from_prototype(v);
            let mut rd = worker_vec.ordered_dispersion(&reverse_draw_order,self.dispersion_mode);
            for (j,rr) in rd.into_iter().enumerate() {
                reverse_dispersions[(reverse_draw_order.len() - j,i)] = rr;
            }
        }
        // assert_eq!(draw_order.len(),7);
        // assert_eq!(forward_dispersions.len(),8);
        // assert_eq!(reverse_dispersions.len(),8);

        let len = forward_dispersions.dim().0;

        let reverse_regularization = (Array1::<f64>::range(0.,len as f64 ,1.) / len as f64).mapv(|x| x.powf(self.split_fraction_regularization));
        let forward_regularization = (Array1::<f64>::range(len as f64 ,0.,-1.) / len as f64).mapv(|x| x.powf(self.split_fraction_regularization));

        for mut feature in forward_dispersions.axis_iter_mut(Axis(1)) {
            feature *= &forward_regularization;
        }

        for mut feature in reverse_dispersions.axis_iter_mut(Axis(1)) {
            feature *= &reverse_regularization;
        }

        let mut dispersions = &forward_dispersions + &reverse_dispersions;

        if self.standardize {
            for (i,mut feature) in dispersions.axis_iter_mut(Axis(1)).enumerate() {
                if forward_dispersions[[0,i]] > 0. {
                    feature /= forward_dispersions[[0,i]];
                }
                else {
                    feature.fill(1.);
                };

            }
        }
        // println!("{:?}",dispersions);

        dispersions

    }

    pub fn split(input:&Array2<f64>,output:&Array2<f64>,parameters:&Parameters) -> Option<(usize,usize,f64)> {
        let input_matrix = RankMatrix::from_array(input, parameters);
        let output_matrix = RankMatrix::from_array(output, parameters);
        RankMatrix::split_input_output(input_matrix, output_matrix,parameters)
    }

    pub fn split_input_output(input_matrix:RankMatrix,output_matrix:RankMatrix,parameters:&Parameters) -> Option<(usize,usize,f64)> {


        let mut draw_orders: Vec<Vec<usize>> = input_matrix.meta_vector.iter().map(|mv| mv.draw_order()).collect();

        // let lsc = parameters.leaf_size_cutoff;

        // for draw_order in draw_orders.iter_mut() {
        //     draw_order.rotate_left(lsc);
        //     draw_order.truncate((draw_order.len() - lsc*2).max(0));
        // }


        let feature_weights = Array1::<f64>::ones(output_matrix.dimensions.0);

        let minima: Vec<Option<(usize,usize,f64)>> =
            draw_orders
                // .into_iter()
                .into_par_iter()
                .enumerate()
                .map(|(i,draw_order)| {
                    let ordered_dispersions = output_matrix.order_dispersions(&draw_order,&feature_weights);
                    let (local_index,dispersion) = ArgMinMax::argmin_v(ordered_dispersions.iter().skip(1))?;
                    Some((i,draw_order[local_index],*dispersion))
                })
                .collect();



        // println!("Minima:{:?}",minima);

        let (feature,sample,_) =
            minima.iter()
            .flat_map(|m| m)
            .min_by(|&a,&b| (a.2).partial_cmp(&b.2).unwrap())?;

        let threshold = input_matrix.feature_fetch(*feature,*sample);

        Some((*feature,*sample,threshold))
    }
}






#[cfg(test)]
mod rank_matrix_tests {

    use super::*;
    use smallvec::SmallVec;

    fn blank_parameter() -> Parameters {
        let mut parameters = Parameters::empty();
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
        let mut parameters = Parameters::empty();
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
        let mut mtx = RankMatrix::new(vec![vec![10.,-3.,0.,5.,-2.,-1.,15.,20.]],&parameters);
        let draw_order = mtx.sort_by_feature(0);

        let order_dispersions = mtx.order_dispersions(&draw_order,&array![1.,]);
        assert_eq!(order_dispersions,array![594.0, 460.0, 354.0, 252.0, 130.0, 92.0, 162.0, 364.0, 594.0]);
    }

    #[test]
    pub fn rank_matrix_test_ordered_mad() {
        let mut parameters = blank_parameter();
        parameters.dispersion_mode = DispersionMode::MAD;
        parameters.norm_mode = NormMode::L1;
        parameters.split_fraction_regularization = 1.;
        let mut mtx = RankMatrix::new(vec![vec![10.,-3.,0.,5.,-2.,-1.,15.,20.]],&parameters);
        let draw_order = mtx.sort_by_feature(0);

        let order_dispersions = mtx.order_dispersions(&draw_order,&array![1.,]);
        assert_eq!(order_dispersions,array![594.0, 460.0, 354.0, 252.0, 130.0, 92.0, 162.0, 364.0, 594.0]);
    }

    #[test]
    pub fn rank_matrix_test_split() {
        let mut parameters = blank_parameter();
        parameters.dispersion_mode = DispersionMode::SSME;
        parameters.norm_mode = NormMode::L1;
        parameters.split_fraction_regularization = 0.;
        let mtx = array![[-3.,10.,0.,5.,-2.,-1.,15.,20.]];

        let out = RankMatrix::split(&mtx.clone(),&mtx.clone(), &parameters);
        assert_eq!(out,Some((0,1,10.)));
    }


    #[test]
    pub fn rank_matrix_derive_test() {
        let parameters = blank_parameter();
        let mut mtx = RankMatrix::new(vec![vec![10.,-3.,0.,5.,-2.,-1.,15.,20.]],&parameters);
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
        let mut mtx = RankMatrix::new(vec![vec![10.,-3.,0.,5.,-2.,-1.,15.,20.]],&parameters);
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
        let mut mtx = RankMatrix::new(vec![vec![10.,-3.,0.,5.,-2.,-1.,15.,20.]],&parameters);
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
        let kid1 = mtx.derive(&vec![0,2,4,6]);
        let kid2 = mtx.derive(&vec![1,3,5,7]);
    }


}
    //
    //
    // pub fn split(input:&Array2<f64>,output:&Array2<f64>,parameters:&Parameters) -> Option<(usize,usize,f64)> {
    //
    //     let mut output_vectors: Vec<RankVector<Vec<Node>>> =
    //         output.axis_iter(Axis(0))
    //         .into_par_iter()
    //         .map(|column| {
    //             RankVector::<Vec<Node>>::link(column.to_vec())
    //             // println!("O:{:?}",output_vectors.last().unwrap());
    //         })
    //         .collect();
    //
    //
    //     let mut draw_orders: Vec<Vec<usize>> =
    //         input.axis_iter(Axis(0))
    //         .into_par_iter()
    //         .map(|column|
    //         {
    //             column.argsort().into_iter().map(|(i,_)| i).collect()
    //         })
    //         .collect();
    //
    //     let mut output_matrix = RankMatrix::from_array(output, parameters);
    //
    //     RankMatrix::split_presorted(&draw_orders, output_matrix, parameters)
    // }
    //
    //
    // pub fn split_presorted(draw_orders:&Vec<Vec<usize>>,mut output_matrix:RankMatrix,parameters:&Parameters) -> Option<(usize,usize,f64)> {
    //
    //     let feature_weights = Array1::<f64>::ones(output_matrix.dimensions.0);
    //
    //     let minima: Vec<Option<(usize,usize,f64)>> =
    //         draw_orders
    //             // .into_iter()
    //             .into_par_iter()
    //             .enumerate()
    //             .map(|(i,draw_order)| {
    //                 let ordered_dispersions = output_matrix.order_dispersions(&draw_order,&feature_weights)?;
    //                 let (local_index,dispersion) = ordered_dispersions.argmin_v()?;
    //                 Some((i,draw_order[local_index],*dispersion))
    //             })
    //             .collect();
    //
    //     eprintln!("{:?}",minima);
    //
    //     let (feature,sample,dispersion) = minima.iter().flat_map(|m| m).min_by(|&a,&b| (a.2).partial_cmp(&b.2).unwrap())?;
    //
    //     // let feature_index = minima.iter().map(|m| m.map(|(f,s,d)| d).unwrap_or(std::f64::MAX)).argmin()?;
    //     // let (feature,sample,dispersion) = minima[feature_index]?;
    //     let mut initial_dispersion: f64 = 0.;
    //
    //     for rv in output_matrix.meta_vector.iter() {
    //         let dispersion: f64 = rv.dispersion(parameters.dispersion_mode);
    //         initial_dispersion += dispersion;
    //     }
    //
    //     // println!("Split successful");
    //     // println!("{}",output_vectors.len());
    //     let delta_dispersion = initial_dispersion - dispersion;
    //     Some((*feature,*sample,delta_dispersion))
    //
    // }
