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

#[derive(Debug)]
pub struct Prototype {
    pub input_array: Array2<f64>,
    pub output_array: Array2<f64>,
    pub input_ranks: RankMatrix,
    pub output_ranks: RankMatrix,
}

impl Prototype {
    pub fn new(input:Array2<f64>,output:Array2<f64>,parameters: &Parameters) -> Prototype {
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

                let prototype = Prototype::new(input_array,output_array, &parameters);

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

                println!("Computing tree {}\r",i);

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

        println!("");

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
    use ndarray::prelude::*;
    use super::super::io::{read_matrix,read_header};
    use std::fs::remove_file;
    use std::env;


    fn iris() -> Array2<f64> {
        array![ [5.1,3.5,1.4,0.2],
          [4.9,3.0,1.4,0.2],
          [4.7,3.2,1.3,0.2],
          [4.6,3.1,1.5,0.2],
          [5.0,3.6,1.4,0.2],
          [5.4,3.9,1.7,0.4],
          [4.6,3.4,1.4,0.3],
          [5.0,3.4,1.5,0.2],
          [4.4,2.9,1.4,0.2],
          [4.9,3.1,1.5,0.1],
          [5.4,3.7,1.5,0.2],
          [4.8,3.4,1.6,0.2],
          [4.8,3.0,1.4,0.1],
          [4.3,3.0,1.1,0.1],
          [5.8,4.0,1.2,0.2],
          [5.7,4.4,1.5,0.4],
          [5.4,3.9,1.3,0.4],
          [5.1,3.5,1.4,0.3],
          [5.7,3.8,1.7,0.3],
          [5.1,3.8,1.5,0.3],
          [5.4,3.4,1.7,0.2],
          [5.1,3.7,1.5,0.4],
          [4.6,3.6,1.0,0.2],
          [5.1,3.3,1.7,0.5],
          [4.8,3.4,1.9,0.2],
          [5.0,3.0,1.6,0.2],
          [5.0,3.4,1.6,0.4],
          [5.2,3.5,1.5,0.2],
          [5.2,3.4,1.4,0.2],
          [4.7,3.2,1.6,0.2],
          [4.8,3.1,1.6,0.2],
          [5.4,3.4,1.5,0.4],
          [5.2,4.1,1.5,0.1],
          [5.5,4.2,1.4,0.2],
          [4.9,3.1,1.5,0.1],
          [5.0,3.2,1.2,0.2],
          [5.5,3.5,1.3,0.2],
          [4.9,3.1,1.5,0.1],
          [4.4,3.0,1.3,0.2],
          [5.1,3.4,1.5,0.2],
          [5.0,3.5,1.3,0.3],
          [4.5,2.3,1.3,0.3],
          [4.4,3.2,1.3,0.2],
          [5.0,3.5,1.6,0.6],
          [5.1,3.8,1.9,0.4],
          [4.8,3.0,1.4,0.3],
          [5.1,3.8,1.6,0.2],
          [4.6,3.2,1.4,0.2],
          [5.3,3.7,1.5,0.2],
          [5.0,3.3,1.4,0.2],
          [7.0,3.2,4.7,1.4],
          [6.4,3.2,4.5,1.5],
          [6.9,3.1,4.9,1.5],
          [5.5,2.3,4.0,1.3],
          [6.5,2.8,4.6,1.5],
          [5.7,2.8,4.5,1.3],
          [6.3,3.3,4.7,1.6],
          [4.9,2.4,3.3,1.0],
          [6.6,2.9,4.6,1.3],
          [5.2,2.7,3.9,1.4],
          [5.0,2.0,3.5,1.0],
          [5.9,3.0,4.2,1.5],
          [6.0,2.2,4.0,1.0],
          [6.1,2.9,4.7,1.4],
          [5.6,2.9,3.6,1.3],
          [6.7,3.1,4.4,1.4],
          [5.6,3.0,4.5,1.5],
          [5.8,2.7,4.1,1.0],
          [6.2,2.2,4.5,1.5],
          [5.6,2.5,3.9,1.1],
          [5.9,3.2,4.8,1.8],
          [6.1,2.8,4.0,1.3],
          [6.3,2.5,4.9,1.5],
          [6.1,2.8,4.7,1.2],
          [6.4,2.9,4.3,1.3],
          [6.6,3.0,4.4,1.4],
          [6.8,2.8,4.8,1.4],
          [6.7,3.0,5.0,1.7],
          [6.0,2.9,4.5,1.5],
          [5.7,2.6,3.5,1.0],
          [5.5,2.4,3.8,1.1],
          [5.5,2.4,3.7,1.0],
          [5.8,2.7,3.9,1.2],
          [6.0,2.7,5.1,1.6],
          [5.4,3.0,4.5,1.5],
          [6.0,3.4,4.5,1.6],
          [6.7,3.1,4.7,1.5],
          [6.3,2.3,4.4,1.3],
          [5.6,3.0,4.1,1.3],
          [5.5,2.5,4.0,1.3],
          [5.5,2.6,4.4,1.2],
          [6.1,3.0,4.6,1.4],
          [5.8,2.6,4.0,1.2],
          [5.0,2.3,3.3,1.0],
          [5.6,2.7,4.2,1.3],
          [5.7,3.0,4.2,1.2],
          [5.7,2.9,4.2,1.3],
          [6.2,2.9,4.3,1.3],
          [5.1,2.5,3.0,1.1],
          [5.7,2.8,4.1,1.3],
          [6.3,3.3,6.0,2.5],
          [5.8,2.7,5.1,1.9],
          [7.1,3.0,5.9,2.1],
          [6.3,2.9,5.6,1.8],
          [6.5,3.0,5.8,2.2],
          [7.6,3.0,6.6,2.1],
          [4.9,2.5,4.5,1.7],
          [7.3,2.9,6.3,1.8],
          [6.7,2.5,5.8,1.8],
          [7.2,3.6,6.1,2.5],
          [6.5,3.2,5.1,2.0],
          [6.4,2.7,5.3,1.9],
          [6.8,3.0,5.5,2.1],
          [5.7,2.5,5.0,2.0],
          [5.8,2.8,5.1,2.4],
          [6.4,3.2,5.3,2.3],
          [6.5,3.0,5.5,1.8],
          [7.7,3.8,6.7,2.2],
          [7.7,2.6,6.9,2.3],
          [6.0,2.2,5.0,1.5],
          [6.9,3.2,5.7,2.3],
          [5.6,2.8,4.9,2.0],
          [7.7,2.8,6.7,2.0],
          [6.3,2.7,4.9,1.8],
          [6.7,3.3,5.7,2.1],
          [7.2,3.2,6.0,1.8],
          [6.2,2.8,4.8,1.8],
          [6.1,3.0,4.9,1.8],
          [6.4,2.8,5.6,2.1],
          [7.2,3.0,5.8,1.6],
          [7.4,2.8,6.1,1.9],
          [7.9,3.8,6.4,2.0],
          [6.4,2.8,5.6,2.2],
          [6.3,2.8,5.1,1.5],
          [6.1,2.6,5.6,1.4],
          [7.7,3.0,6.1,2.3],
          [6.3,3.4,5.6,2.4],
          [6.4,3.1,5.5,1.8],
          [6.0,3.0,4.8,1.8],
          [6.9,3.1,5.4,2.1],
          [6.7,3.1,5.6,2.4],
          [6.9,3.1,5.1,2.3],
          [5.8,2.7,5.1,1.9],
          [6.8,3.2,5.9,2.3],
          [6.7,3.3,5.7,2.5],
          [6.7,3.0,5.2,2.3],
          [6.3,2.5,5.0,1.9],
          [6.5,3.0,5.2,2.0],
          [6.2,3.4,5.4,2.3],
          [5.9,3.0,5.1,1.8] ]
    }

    pub fn iris_prototype() -> Prototype {

        let mut parameters = Parameters::empty();

        Prototype::new(iris(), iris(), &parameters)
    }


    #[test]
    fn prototype_test() {
        let prototype = iris_prototype();
        eprintln!("{:?}",prototype);
        // panic!();
    }


}
