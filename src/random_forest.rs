use std::fs::File;
use std::io::Write;
use std::io::Error;
use std::io::BufRead;
use std::io;
use std::collections::HashMap;
use ndarray::prelude::{Array,Axis,Array2,arr2};

use rayon::prelude::*;

use std::fs::OpenOptions;
use rand::seq::SliceRandom;
use rand::prelude::*;


extern crate rand;
use rand::seq;

use crate::fast_nipal_vector::{project};

use crate::io::Parameters;
use crate::Feature;
use crate::Sample;
use crate::node::Node;
use crate::utils::{project_vec,vec2_from_arr,arr_from_vec2};
use crate::rank_matrix::RankMatrix;

pub struct Forest {
    input_features: Vec<Feature>,
    output_features: Vec<Feature>,
    samples: Vec<Sample>,

    prototype: Prototype,

    roots: Vec<Node>,

    parameters: Parameters,
}

pub struct Prototype {
    pub input_array: Array2<f64>,
    pub output_array: Array2<f64>,
    pub input_ranks: RankMatrix,
    pub output_ranks: RankMatrix,
}

impl Prototype {
    fn from(input:Array2<f64>,output:Array2<f64>,parameters: &Parameters) -> Prototype {
        Prototype {
            input_ranks:RankMatrix::from_array(&input.t().to_owned(),parameters),
            output_ranks: RankMatrix::from_array(&output.t().to_owned(),parameters),
            input_array: input,
            output_array: output,
        }
    }
}

impl Forest {

    pub fn initialize_from(input_array: Array2<f64>, output_array: Array2<f64>,parameters: Parameters) -> Forest{

                let report_string = format!("{}.prototype",parameters.report_address).to_string();

                let samples = Sample::nvec(&parameters.sample_names().unwrap_or(
                    (0..input_array.dim().0).map(|i| format!("{:?}",i)).collect()
                ));

                let input_features = Feature::nvec(&parameters.input_feature_names().unwrap_or(
                    (0..input_array.dim().1).map(|i| format!("{:?}",i)).collect()
                ));
                let output_features = Feature::nvec(&parameters.output_feature_names().unwrap_or(
                    (0..output_array.dim().1).map(|i| format!("{:?}",i)).collect()
                ));

                let prototype = Prototype::from(input_array,output_array, &parameters);

                let tree_limit = parameters.tree_limit;
                let processor_limit = parameters.processor_limit;


                Forest {
                    input_features,
                    output_features,
                    samples,
                    prototype,
                    roots: Vec::with_capacity(tree_limit),
                    parameters: parameters,
                }
    }


    pub fn generate(&mut self) -> Result<(),Error> {

        let results: Vec<Result<(),Error>> = (0..self.parameters.tree_limit)
            .into_par_iter()
            .map(|i| {

                println!("Computing tree {}",i);

                let mut root = Node::prototype(
                            &self.input_features,
                            &self.output_features,
                            &self.samples,
                            &self.parameters,
                        );

                root.grow(&self.prototype,&self.parameters);

                let specific_address = format!("{}.tree_{}.compact",self.parameters.report_address,i);

                root.to_serial().dump(specific_address)
            }).collect();

        results.into_iter().try_fold((),|acc,x| x)

    //     for i in (0..self.parameters.tree_limit) {
    //
    //         println!("Computing tree {}",i);
    //
    //         let mut root = Node::prototype(
    //                     &self.input_features,
    //                     &self.output_features,
    //                     &self.samples,
    //                     &self.parameters,
    //                 );
    //
    //         root.grow(&self.prototype,&self.parameters);
    //
    //         let specific_address = format!("{}.tree_{}.compact",self.parameters.report_address,i);
    //
    //         root.to_serial().dump(specific_address)?;
    //     }
    //
    //     Ok(())
    }



}


#[cfg(test)]
mod random_forest_tests {

    use super::*;
    use super::super::io::{read_matrix,read_header};
    use std::fs::remove_file;
    use std::env;

}
