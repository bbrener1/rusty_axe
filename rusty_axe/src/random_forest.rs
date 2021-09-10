use std::io::Write;
use std::io::Error;
use std::io;
use ndarray::prelude::{Array2};

use rayon::prelude::*;


extern crate rand;

use crate::io::Parameters;
use crate::Feature;
use crate::Sample;
use crate::node::Node;
use crate::rank_matrix::RankMatrix;

pub struct Forest {
    input_features: Vec<Feature>,
    output_features: Vec<Feature>,
    samples: Vec<Sample>,

    prototype: Prototype,

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


                Forest {
                    input_features,
                    output_features,
                    samples,
                    prototype,
                    parameters: parameters,
                }
    }

    pub fn compute_tree(&self,index:usize) -> Result<(),Error> {

        // print!("Computing tree {}\r",index);
        print!("Computing tree {}",index);
        io::stdout().flush()?;

        let mut root = Node::prototype(
                    &self.input_features,
                    &self.output_features,
                    &self.samples,
                );

        root.grow(&self.prototype,&self.parameters);

        let specific_address = format!("{}.tree_{}.compact",self.parameters.report_address,index);

        root.to_serial().dump(specific_address)

    }

    pub fn generate(&mut self) -> Result<(),Error> {


        if self.parameters.parallel_trees {

            println!("Working on trees (not in order)");

            let results: Vec<Result<(),Error>> = (0..self.parameters.tree_limit)
                .into_par_iter()
                .map(|i| {
                    self.compute_tree(i)
                }).collect();

            print!("\n");

            results.into_iter().try_fold((),|_,x| x)
        }
        else {

            println!("Working on trees");

            let results: Vec<Result<(),Error>> = (0..self.parameters.tree_limit)
                .map(|i| {
                    self.compute_tree(i)
                }).collect();

            print!("\n");

            results.into_iter().try_fold((),|_,x| x)

        }

    }





}


#[cfg(test)]
mod random_forest_tests {

    use super::*;
    use crate::utils::test_utils::iris;


    pub fn iris_prototype() -> Prototype {

        let parameters = Parameters::empty();

        Prototype::new(iris(), iris(), &parameters)
    }


    #[test]
    fn prototype_test() {
        let prototype = iris_prototype();
        eprintln!("{:?}",prototype);
        // panic!();
    }


}
