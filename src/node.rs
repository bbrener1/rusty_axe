
use std::sync::Arc;
use std::cmp::PartialOrd;
use std::cmp::Ordering;
use std::sync::mpsc;
use std::f64;
use std::mem::replace;
use std::collections::{HashMap,HashSet};
use serde_json;

use ndarray::prelude::*;

extern crate rand;
use rand::Rng;

use rayon::prelude::*;

use crate::rank_table::RankMatrix;
use crate::Feature;
use crate::Sample;
use crate::io::Parameters;
use crate::io::DispersionMode;
use crate::Filter;
use crate::utils::{argmin,argsort,jaccard,arr_from_vec2};

use std::fs::File;
use std::io::Write;
use std::io::Read;
use std::error::Error;

use std::fs;
use std::path::Path;
use std::ffi::OsStr;
use std::env;

use crate::fast_nipal_vector::project;

use rayon::prelude::*;

// Prefer braid thickness to to be odd to make consensus braids work well
// const BRAID_THICKNESS: usize = 3;

#[derive(Clone,Serialize,Deserialize,Debug)]
pub struct PrototypeNode {

    pub prototype: bool,

    input_table: RankMatrix,
    output_table: RankMatrix,

    input_features: Vec<Feature>,
    output_features: Vec<Feature>,
    samples: Vec<Sample>,

    filter: Option<Filter>,

    pub parent_id: String,
    pub id: String,
    pub depth: usize,
    pub children: Vec<Node>,

    pub medians: Vec<f64>,
    pub feature_weights: Option<Vec<f64>>,
    pub dispersions: Vec<f64>,
    // pub additive: Vec<f64>,
    // pub local_gains: Option<Vec<f64>>,
    // pub absolute_gains: Option<Vec<f64>>

}


impl Node {

    pub fn prototype<'a>(
        input_counts:&Array2<f64>,
        output_counts:&Array2<f64>,
        input_features:&'a[Feature],
        output_features:&'a[Feature],
        samples:&'a[Sample],
        parameters: &'a Parameters,
        feature_weights: Option<Vec<f64>>
    ) -> Node {

        let input_table = RankMatrix::from_array(input_counts,parameters);
        let output_table = RankMatrix::from_array(output_counts,parameters);
        let medians = output_table.medians();
        let additive = vec![0.;medians.len()];
        let dispersions = output_table.dispersions();
        let local_gains = vec![0.;dispersions.len()];

        let new_node = Node {

            prototype: true,

            input_table: input_table,
            output_table: output_table,

            input_features: input_features.to_owned(),
            output_features: output_features.to_owned(),
            samples: samples.to_owned(),

            id: "RT".to_string(),
            parent_id: "RT".to_string(),
            depth: 0,
            children: Vec::new(),

            filter: None,

            medians: medians,
            feature_weights: feature_weights,
            dispersions: dispersions,

        };

        new_node

    }


    fn bootstrap(&self,available_samples:Option<Vec<Sample>>,parameters:&Parameters) -> Node {

        let input_feature_indices: Vec<usize>;
        let output_feature_indices: Vec<usize>;

        if parameters.unsupervised {
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
    // Used for testing

    pub fn blank_node() -> Node {
        let input_counts = arr_from_vec2(vec![]);
        let output_counts = arr_from_vec2(vec![]);
        let input_features = &vec![][..];
        let output_features = &vec![][..];
        let samples = &vec![][..];
        let parameters = Parameters::empty();
        let feature_weight_option = None;
        Node::prototype(&input_counts,&output_counts,input_features,output_features,samples,&parameters,feature_weight_option)
    }

    pub fn rayon_best_split(&self) -> (Feature,Sample) {
        (Feature::q(&0),Sample::q(&0))
    }

    pub fn best_feature_split(&self,feature:Feature) -> Sample {
        Sample::q(&0)
    }

    pub fn derive_specified(&self, samples: &[usize], input_features: &[usize], output_features: &[usize], new_id: &str) -> Node {

        // This is the base derivation of a node from self.
        // Other derivations produce a node specification, then call this method

        // Features and samples in the child node are to be specified using LOCAL INDEXING.

        // DESIGN DECISION: For the moment I am allowing derivation from non-prototype nodes.
        // I may get rid of this later. As a result the global indices of a feature may not match
        // the indices provided above.

        // Derived nodes are guaranteed have matching features/tables
        // Derived nodes are not guaranteed to be a prototype, set this elsewhere

        let mut new_input_table = self.input_table.derive_specified(&input_features,samples);
        let mut new_output_table = self.output_table.derive_specified(&output_features,samples);

        // eprintln!("spec_derivation debug, tables done");

        let new_input_features = input_features.iter().map(|i| self.input_features[*i].clone()).collect();
        let new_output_features = output_features.iter().map(|i| self.output_features[*i].clone()).collect();
        let new_samples: Vec<Sample> = samples.iter().map(|i| self.samples[*i].clone()).collect();


        let medians = new_output_table.medians();


        let dispersions = new_output_table.dispersions();
        let feature_weights = self.feature_weights.as_ref().map(|fw| output_features.iter().map(|y| fw[*y]).collect());

        // let additive = self.medians.iter().zip(medians.iter()).map(|(pm,cm)| cm - pm).collect();
        // let local_gains = Some(self.dispersions().iter().zip(dispersions.iter()).map(|(p,c)| (p/(self.samples().len() as f64)) - (c/((new_samples.len() + 1) as f64))).collect());

        let child = Node {

            prototype: false,

            input_table: new_input_table,
            output_table: new_output_table,

            input_features: new_input_features,
            output_features: new_output_features,
            samples: new_samples,

            parent_id: self.id.clone(),
            id: new_id.to_string(),
            depth: self.depth + 1,
            children: Vec::new(),

            filter: None,

            medians: medians,
            feature_weights: feature_weights,
            dispersions: dispersions,

        };

        child
    }
    //
    // pub fn local_split(&self,parameters:&Parameters) -> (usize,usize) {
    //
    //     let draw_orders: Vec<Vec<usize>>;
    //
    //     if parameters.reduce_input {
    //         let input_projection = project(self.input_table.to_array(),parameters.reduction);
    //         draw_orders = input_projection.loadings
    //             .axis_iter(Axis(0))
    //             .map(|l| argsort(&l.to_vec()))
    //             .map(|v| )
    //             .collect();
    //     }
    //     else {
    //         draw_orders = (0..self.input_table.dimensions.0).map(|i| self.input_table.sort_by_feature(i)).collect();
    //     }
    //
    //     for f_i in (0..self.input_table.dimensions.0) {
    //         let draw_order = self.input_table.sort_by_feature(f_i);
    //     }
    //
    //     (0,0)
    // }

    pub fn report(&self,verbose:bool) {
        println!("Node reporting:");
        println!("Output features:{}",self.output_features().len());
        if verbose {
            println!("{:?}",self.output_features());
            println!("{:?}",self.medians);
            println!("{:?}",self.dispersions);
            println!("{:?}",self.feature_weights);
        }
        println!("Samples: {}", self.samples().len());
        if verbose {
            println!("{:?}", self.samples());
            println!("Counts: {:?}", self.output_table.full_ordered_values());
            println!("Ordered counts: {:?}", self.output_table.full_values());
        }

    }


    pub fn summary(&self) -> String {
        let mut report_string = "".to_string();
        if self.children.len() > 1 {
            report_string.push_str(&format!("!ID:{}\n",self.id));
        }

        report_string
    }

    pub fn data_dump(&self) -> String {
        let mut report_string = String::new();
        report_string.push_str(&format!("!ID:{}\n",self.id));
        report_string.push_str(&format!("Children:"));
        for child in &self.children {
            report_string.push_str(&format!("!C:{}",child.id));
        }
        report_string.push_str("\n");
        report_string.push_str(&format!("ParentID:{}\n",self.parent_id));
        report_string.push_str(&format!("Output features:{:?}\n",self.output_features().len()));
        report_string.push_str(&format!("{:?}\n",self.output_features()));
        report_string.push_str(&format!("Medians:{:?}\n",self.medians));
        report_string.push_str(&format!("Dispersions:{:?}\n",self.dispersions));
        report_string.push_str(&format!("Feature weights:{:?}\n",self.feature_weights));
        report_string.push_str(&format!("Samples:{:?}\n",self.samples().len()));
        report_string.push_str(&format!("{:?}\n",self.samples()));
        // report_string.push_str(&format!("Local gains:{:?}\n",self.local_gains));
        // report_string.push_str(&format!("Absolute gains:{:?}\n",self.absolute_gains));
        report_string.push_str(&format!("Full:{:?}\n",self.output_table.full_ordered_values()));
        report_string
    }


    pub fn set_weights(&mut self, weights:Vec<f64>) {
        self.feature_weights = Some(weights);
    }

    pub fn set_dispersion_mode(&mut self, dispersion_mode : DispersionMode) {
        self.output_table.set_dispersion_mode(dispersion_mode);
    }

    pub fn dispersion_mode(&self) -> DispersionMode {
        self.output_table.dispersion_mode()
    }


    pub fn set_children(&mut self, children: Vec<Node>) {
        self.children = children;
    }

    pub fn output_rank_table(&self) -> &RankMatrix {
        &self.output_table
    }

    pub fn input_rank_table(&self) -> &RankMatrix {
        &self.input_table
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn samples(&self) -> &[Sample] {
        &self.samples
    }

    pub fn sample_names(&self) -> Vec<String> {
        self.samples().iter().map(|s| s.name().to_owned()).collect()
    }

    pub fn input_features(&self) -> &[Feature] {
        &self.input_features
    }

    pub fn input_feature_names(&self) -> Vec<String> {
        self.input_features().iter().map(|f| f.name().to_owned()).collect()
    }

    pub fn output_features(&self) -> &[Feature] {
        &self.output_features
    }

    pub fn output_feature_names(&self) -> Vec<String> {
        self.output_features().iter().map(|f| f.name().to_owned()).collect()
    }

    pub fn medians(&self) -> &Vec<f64> {
        &self.medians
    }

    pub fn dispersions(&self) -> &Vec<f64> {
        &self.dispersions
    }

    pub fn mads(&self) -> &Vec<f64> {
        &self.dispersions
    }

    pub fn dimensions(&self) -> (usize,usize) {
        self.output_table.dimensions
    }



    pub fn covs(&self) -> Vec<f64> {
        self.dispersions.iter().zip(self.mads().iter()).map(|(d,m)| d/m).map(|x| if x.is_normal() {x} else {0.}).collect()
    }

    pub fn crawl_children(&self) -> Vec<&Node> {
        let mut output = Vec::new();
        for child in &self.children {
            output.extend(child.crawl_children());
        }
        output.push(&self);
        output
    }

    pub fn crawl_leaves(&self) -> Vec<&Node> {
        let mut output = Vec::new();
        if self.children.len() < 1 {
            return vec![&self]
        }
        else {
            for child in &self.children {
                output.extend(child.crawl_leaves());
            }
        };
        output
    }

}

impl Node {
    pub fn to_string(self) -> Result<String,serde_json::Error> {
        serde_json::to_string(&self)
    }

    pub fn from_str(input:&str) -> Result<Node,serde_json::Error> {
        serde_json::from_str(input)
    }
}





#[cfg(test)]
mod node_testing {

    use super::*;
    // use ndarray_linalg;

    fn blank_parameter() -> Parameters {
        let mut parameters = Parameters::empty();
        parameters
    }

    fn blank_counts() -> (Array2<f64>,Array2<f64>) {
        (arr_from_vec2(vec![vec![]]),arr_from_vec2(vec![vec![]]))
    }

    fn blank_node() -> Node {
        let (input_counts,output_counts) = blank_counts();
        let input_features = &vec![][..];
        let output_features = &vec![][..];
        let samples = &vec![][..];
        let parameters = blank_parameter();
        let feature_weight_option = None;
        Node::prototype(&input_counts,&output_counts,input_features,output_features,samples,&parameters,feature_weight_option)
    }

    fn trivial_node() -> Node {
        let (input_counts,output_counts) = blank_counts();
        let input_features = &vec![Feature::q(&1)][..];
        let output_features = &vec![Feature::q(&2)][..];
        let samples = &vec![][..];
        let parameters = blank_parameter();
        let feature_weight_option = None;
        Node::prototype(&input_counts,&output_counts,input_features,output_features,samples,&parameters,feature_weight_option)
    }

    fn simple_node() -> Node {
        let input_counts = arr_from_vec2(vec![vec![10.,-3.,0.,5.,-2.,-1.,15.,20.]]);
        let output_counts = arr_from_vec2(vec![vec![10.,-3.,0.,5.,-2.,-1.,15.,20.]]);
        let input_features = &vec![Feature::q(&0)][..];
        let output_features = &vec![Feature::q(&0)][..];
        let samples = &Sample::vec(vec![0,1,2,3,4,5,6,7])[..];
        let parameters = blank_parameter();
        let feature_weight_option = None;
        Node::prototype(&input_counts,&output_counts,input_features,output_features,samples,&parameters,feature_weight_option)
    }

    #[test]
    fn node_test_blank() {
        let mut root = blank_node();
        root.mads();
        root.medians();
    }

    #[test]
    fn node_test_trivial() {
        let mut root = trivial_node();
        root.mads();
        root.medians();
    }

    //
    // #[test]
    // fn node_test_dispersions() {
    //
    //     let mut root = simple_node();
    //
    //     let split0 = root.feature_index_split(0).unwrap();
    //
    //     println!("{:?}",root.samples());
    //     println!("{:?}",root.output_table.full_values());
    //     println!("{:?}",split0);
    //
    //     // panic!();
    // }
    //
    // #[test]
    // fn node_test_subsample() {
    //
    //     let mut root = simple_node();
    //
    //
    //     for i in 0..1000 {
    //         let sub = root.subsample(8, 2, 2);
    //         let split_option = sub.rayon_best_split();
    //         eprintln!("{:?}",sub.strip_clone());
    //         let (draw_order,drop_set) = sub.input_rank_table().sort_by_feature(0);
    //         eprintln!("{:?}",(&draw_order,&drop_set));
    //         eprintln!("{:?}",sub.output_rank_table().order_dispersions(&draw_order,&drop_set,&sub.feature_weights));
    //         eprintln!("{:?}",split_option.unwrap());
    //         // if let Some(split) = split_option {
    //         //     root.clone().derive_complete_by_split(&split,None);
    //         // }
    //     }
    //
    // }


    // #[test]
    // fn node_test_split() {
    //
    //     let mut root = simple_node();
    //
    //     let split = root.rayon_best_split();
    //
    //     println!("{:?}",split);
    //
    //     assert_eq!(split.dispersion,2822.265625);
    //     assert_eq!(split.value, 5.);
    // }
    //
    // #[test]
    // fn node_test_simple() {
    //
    //     let mut root = simple_node();
    //
    //     println!("Created node");
    //     // root.split_node();
    //     let children = root.braid_split_node(8,1,1);
    //     root.children = children.unwrap();
    //
    //     eprintln!("{:?}",root.braids[0]);
    //
    //     assert_eq!(&root.children[0].sample_names(),&vec!["1".to_string(),"3".to_string(),"4".to_string(),"5".to_string()]);
    //     assert_eq!(&root.children[1].sample_names(),&vec!["0".to_string(),"6".to_string(),"7".to_string()]);
    //
    //     // assert_eq!(root.children[0].samples(),&vec!["1".to_string(),"4".to_string(),"5".to_string()]);
    //     // assert_eq!(root.children[1].samples(),&vec!["0".to_string(),"3".to_string(),"6".to_string(),"7".to_string()]);
    //
    //
    //     // assert_eq!(&root.children[0].sample_names(),&vec!["1".to_string(),"2".to_string(),"3".to_string(),"4".to_string(),"5".to_string()]);
    //     // assert_eq!(&root.children[1].sample_names(),&vec!["0".to_string(),"2".to_string(),"6".to_string(),"7".to_string()]);
    //     //
    //     // assert_eq!(&root.children[0].output_table.full_values(),&vec![vec![-3.,0.,5.,-2.,-1.]]);
    //     // assert_eq!(&root.children[1].output_table.full_values(),&vec![vec![10.,0.,15.,20.]]);
    //
    // }
    //


    #[test]
    fn node_test_json() {
        let n = simple_node();
        let ns = n.to_string();
    }

}
