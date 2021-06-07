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

pub struct Forest {
    input_array: Array2<f64>,
    output_array: Array2<f64>,
    input_features: Vec<Feature>,
    output_features: Vec<Feature>,
    samples: Vec<Sample>,

    roots: Vec<Node>,

    parameters: Parameters,
}

impl Forest {

    fn initialize_from(input_array: Array2<f64>, output_array: Array2<f64>,parameters: Parameters, report_address: &str) -> Forest{

                let report_string = format!("{}.prototype",report_address).to_string();

                let samples = Sample::nvec(&parameters.sample_names().unwrap_or(
                    (0..input_array.dim().0).map(|i| format!("{:?}",i)).collect()
                ));

                let input_features = Feature::nvec(&parameters.input_feature_names().unwrap_or(
                    (0..input_array.dim().1).map(|i| format!("{:?}",i)).collect()
                ));
                let output_features = Feature::nvec(&parameters.output_feature_names().unwrap_or(
                    (0..output_array.dim().1).map(|i| format!("{:?}",i)).collect()
                ));

                println!("Starting prototype");

                let prototype = Node::prototype(&input_array,&output_array,&input_features,&output_features,&samples, &parameters, None);

                let tree_limit = parameters.tree_limit;
                let processor_limit = parameters.processor_limit;

                Forest {
                    input_array,
                    output_array,
                    input_features,
                    output_features,
                    samples,
                    roots: Vec::with_capacity(tree_limit),
                    parameters: parameters,
                }
    }

    fn bootstrap(&self,available_samples:Option<Vec<Sample>>) -> Node {

        let input_feature_indices: Vec<usize>;
        let output_feature_indices: Vec<usize>;

        if self.parameters.unsupervised {
            let input_feature_len = self.input_features.len();
            let mut all_indices = (0..input_feature_len).collect::<Vec<usize>>();
            let (ii,oi) = all_indices.partial_shuffle(&mut thread_rng(),input_feature_len/2);
            // Why can't you destructure into an initialized variable? It is a mystery.
            input_feature_indices = ii.to_vec();
            output_feature_indices = oi.to_vec();
        }
        else {
            input_feature_indices = (0..self.input_features.len()).collect();
            output_feature_indices = (0..self.output_features.len()).collect();
        }

        let mut rng = thread_rng();
        let input_index_bootstrap: Vec<usize> = (0..self.parameters.input_feature_subsample).map(|_| rng.gen_range(0..input_feature_indices.len())).collect();
        let output_index_bootstrap: Vec<usize> = (0..self.parameters.output_feature_subsample).map(|_| rng.gen_range(0..output_feature_indices.len())).collect();

        let input_feature_bootstrap: Vec<Feature> = input_index_bootstrap.iter().map(|&i| self.input_features[i].clone()).collect();
        let output_feature_bootstrap: Vec< Feature> = output_index_bootstrap.iter().map(|&i| self.output_features[i].clone()).collect();

        let samples = available_samples.unwrap_or(self.samples.clone());
        let sample_bootstrap: Vec<Sample> = (0..self.parameters.sample_subsample).map(|i| samples[i].clone()).collect();
        let sample_index_bootstrap: Vec<usize> = sample_bootstrap.iter().map(|s| s.index).collect();

        let mut input_array_bootstrap = self.input_array.select(Axis(0),&sample_index_bootstrap).select(Axis(1),&input_index_bootstrap);
        let mut output_array_bootstrap = self.output_array.select(Axis(0),&sample_index_bootstrap).select(Axis(1),&output_index_bootstrap);

        println!("{:?}",input_array_bootstrap);
        println!("{:?}",output_array_bootstrap);


        Node::prototype(
            &input_array_bootstrap,
            &output_array_bootstrap,
            &input_feature_bootstrap,
            &output_feature_bootstrap,
            &sample_bootstrap,
            &self.parameters,
            None
        )

    }

    pub fn generate(&mut self) -> Result<(),Error> {

        let roots: Vec<Node> = (0..self.parameters.tree_limit).map(|tree|
            {
                self.bootstrap(None)
            }
        ).collect();

        self.roots = roots;

        Ok(())
    }

    //
    // pub fn generate(&mut self, parameters:Arc<Parameters>, remember: bool) -> Result<(),Error> {
    //
    //     if let Some(ref prototype) = self.prototype_tree {
    //
    //         eprintln!("Constructing {} trees",self.size);
    //
    //         if parameters.big_mem {
    //             for chunk in
    //                 (1_usize..self.size+1)
    //                 .collect::<Vec<usize>>()
    //                 .chunks(10) {
    //                     let ct = chunk.iter()
    //                     .map(|i| {
    //                         eprintln!("Tree {}",i);
    //                         let mut new_tree = self.prototype_tree.as_ref().expect("No prototype tree").clone();
    //                         new_tree.report_address = format!("{}.{}.compact",parameters.report_address, i).to_string();
    //                         new_tree
    //                     })
    //                     .collect::<Vec<Tree>>()
    //                     .into_par_iter()
    //                     .map(|mut new_tree| {
    //                         new_tree.grow_branches(parameters.clone());
    //                         // new_tree.serialize_clone();
    //                         new_tree.serialize_nano();
    //                         new_tree
    //                     })
    //                     .collect::<Vec<Tree>>();
    //                     if remember {
    //                         self.trees.extend_from_slice(&ct);
    //                     }
    //             }
    //         }
    //         else {
    //             for tree in 1..self.size+1 {
    //
    //                 eprintln!("Tree {}",tree);
    //
    //                 let mut new_tree = self.prototype_tree.as_ref().expect("No prototype tree").clone();
    //                 new_tree.report_address = format!("{}.{}.compact",parameters.report_address, tree).to_string();
    //                 new_tree.grow_branches(parameters.clone());
    //                 if remember {
    //                     // new_tree.serialize_clone()?;
    //                     new_tree.serialize_nano()?;
    //                     self.trees.push(new_tree);
    //                 }
    //                 // else { new_tree.serialize()?; };
    //                 else {new_tree.serialize_nano()?;};
    //             }
    //
    //         };
    //
    //         eprintln!("Dumping headers:");
    //         eprintln!("{:?}",["./",&self.parameters.report_address.clone(),".ifh"].join(""));
    //         eprintln!("{:?}",["./",&self.parameters.report_address.clone(),".ofh"].join(""));
    //
    //         let mut output_header_dump = OpenOptions::new().write(true).create(true).truncate(true).open([&self.parameters.report_address.clone(),".ifh"].join(""))?;
    //         output_header_dump.write(self.prototype_tree.as_ref().unwrap().input_feature_names().join("\n").as_bytes())?;
    //         output_header_dump.write(b"\n")?;
    //
    //         let mut output_header_dump = OpenOptions::new().write(true).create(true).truncate(true).open([&self.parameters.report_address.clone(),".ofh"].join(""))?;
    //         output_header_dump.write(self.prototype_tree.as_ref().unwrap().output_feature_names().join("\n").as_bytes())?;
    //         output_header_dump.write(b"\n")?;
    //
    //         Ok(())
    //
    //     }
    //     else {
    //         panic!("Attempted to generate a forest without a prototype tree. Are you trying to do predictions after reloading from compact backups?")
    //     }
    //
    //     // self.set_leaf_weights();
    //
    // }


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
        Forest::initialize_from(counts.clone(),counts.clone(),Parameters::empty(), "./testing/test_trees");
    }

    #[test]
    fn test_forest_initialization_iris() {
        let iris_vec = read_matrix("./testing/iris.tsv");
        let iris = arr_from_vec2(iris_vec);
        // let features = read_header("../testing/iris.features");
        Forest::initialize_from(iris.clone(),iris.clone(),Parameters::empty(),"./testing/err");
    }

    #[test]
    fn test_forest_bootstrap() {
        let iris_vec = read_matrix("./testing/iris.tsv");
        let iris = arr_from_vec2(iris_vec);
        // let features = read_header("../testing/iris.features");
        let mut forest = Forest::initialize_from(iris.clone(),iris.clone(),Parameters::empty(),"./testing/err");

        forest.bootstrap(None);
    }


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
