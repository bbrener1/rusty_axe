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
            input_ranks:RankMatrix::from_array(&input,parameters),
            output_ranks: RankMatrix::from_array(&output,parameters),
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

        let roots: Vec<Node> = (0..self.parameters.tree_limit).map(|tree|
            {
                Node::prototype(
                    &self.input_features,
                    &self.output_features,
                    &self.samples,
                    &self.parameters,
                )
            }
        ).collect();


        self.roots = roots;

        for root in self.roots.iter_mut() {
            root.grow(&self.prototype,&self.parameters);
        }

        for (i,root) in self.roots.iter().enumerate() {
            let specific_address = format!("{}.tree_{}.zip",self.parameters.report_address,i);
            root.to_serial().dump(specific_address)?;
        }

        Ok(())
    }



}


#[cfg(test)]
mod random_forest_tests {

    use super::*;
    use super::super::io::{read_matrix,read_header};
    use std::fs::remove_file;
    use std::env;

    // #[test]
    // fn test_forest_initialization_trivial() {
    //     Forest::initialize_from(vec![],vec![],Parameters::empty() , "../testing/test_trees");
    // }

    #[test]
    fn test_forest_initialization_simple() {
        let counts = arr_from_vec2(vec![vec![10.,-3.,0.,5.,-2.,-1.,15.,20.]]);
        Forest::initialize_from(counts.clone(),counts.clone(),Parameters::empty());
    }

    #[test]
    fn test_forest_initialization_iris() {
        let iris_vec = read_matrix("./testing/iris.tsv");
        let iris = arr_from_vec2(iris_vec);
        // let features = read_header("../testing/iris.features");
        Forest::initialize_from(iris.clone(),iris.clone(),Parameters::empty());
    }
    //
    // #[test]
    // fn test_forest_bootstrap() {
    //     let iris_vec = read_matrix("./testing/iris.tsv");
    //     let iris = arr_from_vec2(iris_vec);
    //     // let features = read_header("../testing/iris.features");
    //     let mut forest = Forest::initialize_from(iris.clone(),iris.clone(),Parameters::empty(),"./testing/err");
    //
    //     forest.bootstrap(None);
    // }


    // #[test]
    // fn test_forest_reconstitution_simple() {
    //     let params = Parameters::empty();
    //     params.backup_vec = Some(vec!["./testing/precomputed_trees/simple.0.compact".to_string(), "./testing/precomputed_trees/simple.1.compact".to_string()]);
    //     let new_forest = Forest::compact_reconstitute(Arc::new(params),"./testing/").expect("Reconstitution test");
    //
    //     println!("Reconstitution successful");
    //
    //     let reconstituted_features: Vec<String> = new_forest.predictive_trees()[0].crawl_nodes().iter().map(|x| x.feature().clone()).filter(|x| x.is_some()).map(|x| x.unwrap()).collect();
    //     let correct_features: Vec<String> = vec!["0","0","0","0","0","0"].iter().map(|x| x.to_string()).collect();
    //     assert_eq!(reconstituted_features,correct_features);
    //
    //
    //     let correct_splits: Vec<f64> = vec![-1.,-2.,20.,10.,10.,5.];
    //     let reconstituted_splits: Vec<f64> = new_forest.predictive_trees()[0].crawl_nodes().iter().map(|x| x.split().clone()).filter(|x| x.is_some()).map(|x| x.unwrap()).collect();
    //     assert_eq!(reconstituted_splits,correct_splits);
    // }
    //
    //
    // #[test]
    // fn test_forest_reconstitution() {
    //     let new_forest = Forest::compact_reconstitute(TreeBackups::Vector(vec!["./testing/precomputed_trees/iris.0.compact".to_string(),"./testing/precomputed_trees/iris.1.compact".to_string()]), None, None, Some(1), "./testing/").expect("Reconstitution test");
    //
    //     println!("Reconstitution successful");
    //
    //     let reconstituted_features: Vec<String> = new_forest.predictive_trees()[0].crawl_nodes().iter().map(|x| x.feature().clone()).filter(|x| x.is_some()).map(|x| x.unwrap()).collect();
    //     let correct_features: Vec<String> = vec!["sepal_length","petal_length","sepal_width","sepal_width","sepal_length","sepal_width","sepal_width","sepal_width","sepal_width","sepal_width"].iter().map(|x| x.to_string()).collect();
    //     assert_eq!(reconstituted_features,correct_features);
    //
    //
    //     let correct_splits: Vec<f64> = vec![1.5,5.7,1.2,1.1,4.9,1.8,1.4,2.2,1.8,1.];
    //     let reconstituted_splits: Vec<f64> = new_forest.predictive_trees()[0].crawl_nodes().iter().map(|x| x.split().clone()).filter(|x| x.is_some()).map(|x| x.unwrap()).collect();
    // }

    // #[test]
    // fn test_forest_generation() {
    //
    //     let counts = read_counts("./testing/iris.drop");
    //     let features = read_header("./testing/iris.features");
    //
    //     let mut params = Parameters::empty();
    //
    //     params.leaf_size_cutoff = Some(10);
    //
    //
    //     params.input_features = Some(4);
    //     params.output_features = Some(4);
    //     params.sample_subsample = Some(150);
    //     params.processor_limit = Some(1);
    //     params.counts = Some(counts.clone());
    //     params.feature_names = Some(features);
    //     params.tree_limit = Some(1);
    //     params.auto();
    //
    //     let arc_params = Arc::new(params);
    //
    //     let mut new_forest = Forest::initialize(&counts,arc_params.clone(), "./testing/tmp_test");
    //     new_forest.generate(arc_params.clone(), true);
    //
    //
    //     let computed_features: Vec<String> = new_forest.predictive_trees()[0].crawl_nodes().iter().map(|x| x.feature().clone()).filter(|x| x.is_some()).map(|x| x.unwrap()).collect();
    //     let correct_features: Vec<String> = vec!["sepal_length","petal_length","sepal_width","sepal_width","sepal_length","sepal_width","sepal_width","sepal_width","sepal_width","sepal_width"].iter().map(|x| x.to_string()).collect();
    //     assert_eq!(computed_features,correct_features);
    //
    //
    //     let computed_splits: Vec<f64> = new_forest.predictive_trees()[0].crawl_nodes().iter().map(|x| x.split().clone()).filter(|x| x.is_some()).map(|x| x.unwrap()).collect();
    //     let correct_splits: Vec<f64> = vec![1.5,5.7,1.2,1.1,4.9,1.8,1.4,2.2,1.8,1.];
    //     assert_eq!(computed_splits,correct_splits);
    //
    //
    //     remove_file("./testing/tmp_test.0");
    //     remove_file("./testing/tmp_test.0.summary");
    //     remove_file("./testing/tmp_test.0.dump");
    //     remove_file("./testing/tmp_test.1");
    //     remove_file("./testing/tmp_test.1.summary");
    //     remove_file("./testing/tmp_test.1.dump");
    // }
}
