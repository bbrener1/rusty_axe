
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
use rand::prelude::*;

use rayon::prelude::*;

use crate::rank_matrix::{RankMatrix};
use crate::Feature;
use crate::Sample;
use crate::io::Parameters;
use crate::io::DispersionMode;
use crate::Filter;
use crate::utils::{argmin,argsort,jaccard,arr_from_vec2,ArgSort,ArgSortII};
use crate::random_forest::Prototype;

use std::fs::File;
use std::io::Write;
use std::io::Read;
use std::error::Error;

use std::fs;
use std::path::Path;
use std::ffi::OsStr;
use std::env;

extern crate zip;
use zip::ZipWriter;
use zip::write::FileOptions;

use crate::fast_nipal_vector::{project,Projection};

use rayon::prelude::*;

// Prefer braid thickness to to be odd to make consensus braids work well
// const BRAID_THICKNESS: usize = 3;

#[derive(Clone,Debug)]
pub struct Node {

    pub prototype: bool,

    input_features: Vec<Feature>,
    output_features: Vec<Feature>,
    samples: Vec<Sample>,

    filter: Option<Filter>,
    input_projection: Option<Projection>,
    output_projection: Option<Projection>,

    pub depth: usize,
    pub children: Vec<Node>,

}

impl Node {

    pub fn prototype<'a>(
        input_features:&'a[Feature],
        output_features:&'a[Feature],
        samples:&'a[Sample],
        parameters: &'a Parameters,
    ) -> Node {

        let new_node = Node {

            prototype: true,

            input_features: input_features.to_owned(),
            output_features: output_features.to_owned(),
            samples: samples.to_owned(),

            depth: 0,
            children: vec![],

            filter: None,
            input_projection: None,
            output_projection: None,

        };

        new_node

    }

    fn derive_prototype(&self,samples:Vec<Sample>,filter:Option<Filter>) -> Node {
        if !self.prototype {panic!("Attempted to derive a prototype from a slim node")}
        Node {
            prototype: true,

            input_features: self.input_features.clone(),
            output_features: self.output_features.clone(),
            samples: samples,

            depth: self.depth+1,
            children: vec![],

            filter: filter,
            input_projection: None,
            output_projection: None,
        }
    }

    fn bootstrap(&self,parameters:&Parameters) -> (Vec<Feature>,Vec<Feature>,Vec<Sample>) {

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
        let input_index_bootstrap: Vec<usize> = (0..parameters.input_feature_subsample).map(|_| rng.gen_range(0..input_feature_indices.len())).collect();
        let output_index_bootstrap: Vec<usize> = (0..parameters.output_feature_subsample).map(|_| rng.gen_range(0..output_feature_indices.len())).collect();

        let input_feature_bootstrap: Vec<Feature> = input_index_bootstrap.iter().map(|&i| self.input_features[i].clone()).collect();
        let output_feature_bootstrap: Vec< Feature> = output_index_bootstrap.iter().map(|&i| self.output_features[i].clone()).collect();

        let sample_index_bootstrap: Vec<usize> = (0..parameters.sample_subsample).map(|i| rng.gen_range(0..self.samples.len())).collect();
        let sample_bootstrap: Vec<Sample> = sample_index_bootstrap.iter().map(|i| self.samples[*i].clone()).collect();

        (input_feature_bootstrap,output_feature_bootstrap,sample_bootstrap)
    }

    fn derive_bootstrap(&self,parameters:&Parameters) -> Node {
        let (input_bootstrap,output_bootstrap,sample_bootstrap) = self.bootstrap(parameters);
        let node = Node {

            prototype: false,

            input_features: input_bootstrap,
            output_features: output_bootstrap,
            samples: sample_bootstrap,

            depth: self.depth,
            children: vec![],

            filter: None,
            input_projection: None,
            output_projection: None,

        };
        node
    }


    fn input_projection(&mut self, prototype:&Prototype,parameters:&Parameters) -> &Projection {
        self.input_projection.get_or_insert({
            let feature_indices: Vec<usize> = self.input_features.iter().map(|f| f.index).collect();
            let sample_indices: Vec<usize> = self.samples.iter().map(|s| s.index).collect();
            let input_array = prototype.input_array.select(Axis(0),&sample_indices).select(Axis(1),&feature_indices);
            project(input_array,parameters.reduction).expect("Projection failed")
        })
    }

    fn output_projection(&mut self, prototype:&Prototype,parameters:&Parameters) -> &Projection {
        self.output_projection.get_or_insert({
            let feature_indices: Vec<usize> = self.output_features.iter().map(|f| f.index).collect();
            let sample_indices: Vec<usize> = self.samples.iter().map(|s| s.index).collect();
            let output_array = prototype.output_array.select(Axis(0),&sample_indices).select(Axis(1),&feature_indices);
            project(output_array,parameters.reduction).expect("Projection failed")
        })
    }

    fn input_rank_matrix(&mut self,prototype:&Prototype,parameters:&Parameters) -> RankMatrix {
        if parameters.reduce_input {
            let projection = self.input_projection(prototype,parameters);
            RankMatrix::from_array(&projection.loadings,parameters)
        }
        else {
            let feature_indices: Vec<usize> = self.input_features.iter().map(|f| f.index).collect();
            let sample_indices: Vec<usize> = self.samples.iter().map(|s| s.index).collect();
            prototype.input_ranks.derive_specified(&feature_indices,&sample_indices)
        }
    }

    fn output_rank_matrix(&mut self,prototype:&Prototype,parameters:&Parameters) -> RankMatrix {
        if parameters.reduce_output {
            let projection = self.output_projection(prototype,parameters);
            RankMatrix::from_array(&projection.loadings,parameters)
        }
        else {
            let feature_indices: Vec<usize> = self.output_features.iter().map(|f| f.index).collect();
            let sample_indices: Vec<usize> = self.samples.iter().map(|s| s.index).collect();
            prototype.output_ranks.derive_specified(&feature_indices,&sample_indices)
        }
    }

    pub fn local_split(&mut self,prototype:&Prototype,parameters:&Parameters) -> Option<(Filter,Filter)> {

        let input_ranks = self.input_rank_matrix(prototype, parameters);
        let output_ranks = self.output_rank_matrix(prototype, parameters);

        let (local_feature,local_sample,local_threshold) = RankMatrix::split_input_output(input_ranks,output_ranks,parameters)?;
        //
        // println!("Local Split: {},{},{}",local_feature,local_sample,local_threshold);

        let (left_filter,right_filter) = if parameters.reduce_input {
            let input_features = self.input_features.clone();
            let Projection {
                weights, means, ..
            } = self.input_projection(prototype,parameters);
            let left_filter = Filter::new(input_features.clone(),means.row(local_feature).to_vec(),weights.row(local_feature).to_vec(),local_threshold,false);
            let right_filter = Filter::new(input_features,means.row(local_feature).to_vec(),weights.row(local_feature).to_vec(),local_threshold,true);
            (left_filter,right_filter)
        }
        else {
            let left_filter = Filter::new(vec![self.input_features[local_feature].clone(),],vec![0.,],vec![1.,],local_threshold,false);
            let right_filter = Filter::new(vec![self.input_features[local_feature].clone(),],vec![0.,],vec![1.,],local_threshold,true);
            (left_filter,right_filter)
        };

        Some((left_filter,right_filter))
    }

    pub fn split(&mut self, prototype:&Prototype,parameters:&Parameters) -> Option<&mut [Node]> {

        if self.depth > parameters.depth_cutoff {
            // println!("Depth exceeded");
            return None
        };

        if !self.prototype {panic!("Attempted to split on a non-prototype node")};

        let mut slim_node = self.derive_bootstrap(parameters);
        let (left_filter,right_filter) = slim_node.local_split(prototype, parameters)?;

        let sample_indices: Vec<usize> = self.samples.iter().map(|s| s.index).collect();
        let local_inputs = prototype.input_array.select(Axis(0),&sample_indices);
        let left_samples: Vec<Sample> = left_filter.filter_matrix(&local_inputs).into_iter().map(|i| self.samples[i].clone()).collect();
        let right_samples: Vec<Sample> = right_filter.filter_matrix(&local_inputs).into_iter().map(|i| self.samples[i].clone()).collect();

        if (left_samples.len() < parameters.leaf_size_cutoff) || (right_samples.len() < parameters.leaf_size_cutoff) {
            // println!("LL:{},{}",left_samples.len(),right_samples.len());
            // println!("leaf size cutoff");
            return None
        };

        let mut left_child = self.derive_prototype(left_samples, Some(left_filter));
        let mut right_child = self.derive_prototype(right_samples, Some(right_filter));

        let children = vec![left_child,right_child];
        self.children = children;

        Some(&mut self.children)

    }

    pub fn grow(&mut self, prototype:&Prototype, parameters:&Parameters) {
        if let Some(children) = self.split(prototype,parameters) {
            for child in children.iter_mut() {
                child.grow(prototype,parameters);
                // println!("D:{:?}",child.depth);
            }
        }
    }

    pub fn blank_node() -> Node {
        let input_features = &vec![][..];
        let output_features = &vec![][..];
        let samples = &vec![][..];
        let parameters = Parameters::empty();
        Node::prototype(input_features,output_features,samples,&parameters,)
    }

    pub fn set_children(&mut self, children: Vec<Node>) {
        self.children = children;
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

    pub fn to_serial(&self) -> SerialNode {
        let serialized_children = self.children.iter().map(|c| c.to_serial()).collect();
        SerialNode {
            samples: self.samples.iter().map(|s| s.index).collect(),
            filter: self.filter.clone(),
            depth: self.depth,
            children: serialized_children,
        }
    }

}

#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct SerialNode {
    samples: Vec<usize>,
    filter: Option<Filter>,
    depth: usize,
    children: Vec<SerialNode>,

}

impl SerialNode {

    pub fn dump(self, address: String) -> Result<(),std::io::Error> {
        use std::fs::OpenOptions;
        use std::io::Write;
        let mut handle = OpenOptions::new().write(true).truncate(true).create(true).open(address.clone())?;
        let string = self.to_string()?;
        handle.write(string.as_bytes())?;
        Ok(())
    }

    // pub fn dump(self, address: String) -> Result<(),std::io::Error> {
    //     use std::fs::OpenOptions;
    //     use std::io::Write;
    //     let mut handle = OpenOptions::new().write(true).truncate(true).create(true).open(address.clone())?;
    //     let mut zip = ZipWriter::new(handle);
    //     zip.start_file(address,FileOptions::default());
    //     let string = self.to_string()?;
    //     zip.write(string.as_bytes())?;
    //     zip.finish()?;
    //     Ok(())
    // }

    pub fn to_string(self) -> Result<String,serde_json::Error> {
        serde_json::to_string(&self)
    }

    pub fn from_str(input:&str) -> Result<SerialNode,serde_json::Error> {
        serde_json::from_str(input)
    }
}


//
//
//
// #[cfg(test)]
// mod node_testing {
//
//     use super::*;
//     // use ndarray_linalg;
//
//     fn blank_parameter() -> Parameters {
//         let mut parameters = Parameters::empty();
//         parameters
//     }
//
//     fn blank_counts() -> (Array2<f64>,Array2<f64>) {
//         (arr_from_vec2(vec![vec![]]),arr_from_vec2(vec![vec![]]))
//     }
//
//     fn blank_node() -> Node {
//         let (input_counts,output_counts) = blank_counts();
//         let input_features = &vec![][..];
//         let output_features = &vec![][..];
//         let samples = &vec![][..];
//         let parameters = blank_parameter();
//         let feature_weight_option = None;
//         Node::prototype(input_features,output_features,samples,&parameters,feature_weight_option)
//     }
//
//     fn trivial_node() -> Node {
//         let (input_counts,output_counts) = blank_counts();
//         let input_features = &vec![Feature::q(&1)][..];
//         let output_features = &vec![Feature::q(&2)][..];
//         let samples = &vec![][..];
//         let parameters = blank_parameter();
//         let feature_weight_option = None;
//         Node::prototype(&input_counts,&output_counts,input_features,output_features,samples,&parameters,feature_weight_option)
//     }
//
//     fn simple_node() -> Node {
//         let input_counts = arr_from_vec2(vec![vec![10.,-3.,0.,5.,-2.,-1.,15.,20.]]);
//         let output_counts = arr_from_vec2(vec![vec![10.,-3.,0.,5.,-2.,-1.,15.,20.]]);
//         let input_features = &vec![Feature::q(&0)][..];
//         let output_features = &vec![Feature::q(&0)][..];
//         let samples = &Sample::vec(vec![0,1,2,3,4,5,6,7])[..];
//         let parameters = blank_parameter();
//         let feature_weight_option = None;
//         Node::prototype(&input_counts,&output_counts,input_features,output_features,samples,&parameters,feature_weight_option)
//     }
//
//     #[test]
//     fn node_test_blank() {
//         let mut root = blank_node();
//         root.mads();
//         root.medians();
//     }
//
//     #[test]
//     fn node_test_trivial() {
//         let mut root = trivial_node();
//         root.mads();
//         root.medians();
//     }
//
//     //
//     // #[test]
//     // fn node_test_dispersions() {
//     //
//     //     let mut root = simple_node();
//     //
//     //     let split0 = root.feature_index_split(0).unwrap();
//     //
//     //     println!("{:?}",root.samples());
//     //     println!("{:?}",root.output_table.full_values());
//     //     println!("{:?}",split0);
//     //
//     //     // panic!();
//     // }
//     //
//     // #[test]
//     // fn node_test_subsample() {
//     //
//     //     let mut root = simple_node();
//     //
//     //
//     //     for i in 0..1000 {
//     //         let sub = root.subsample(8, 2, 2);
//     //         let split_option = sub.rayon_best_split();
//     //         eprintln!("{:?}",sub.strip_clone());
//     //         let (draw_order,drop_set) = sub.input_rank_table().sort_by_feature(0);
//     //         eprintln!("{:?}",(&draw_order,&drop_set));
//     //         eprintln!("{:?}",sub.output_rank_table().order_dispersions(&draw_order,&drop_set,&sub.feature_weights));
//     //         eprintln!("{:?}",split_option.unwrap());
//     //         // if let Some(split) = split_option {
//     //         //     root.clone().derive_complete_by_split(&split,None);
//     //         // }
//     //     }
//     //
//     // }
//
//
//     // #[test]
//     // fn node_test_split() {
//     //
//     //     let mut root = simple_node();
//     //
//     //     let split = root.rayon_best_split();
//     //
//     //     println!("{:?}",split);
//     //
//     //     assert_eq!(split.dispersion,2822.265625);
//     //     assert_eq!(split.value, 5.);
//     // }
//     //
//     // #[test]
//     // fn node_test_simple() {
//     //
//     //     let mut root = simple_node();
//     //
//     //     println!("Created node");
//     //     // root.split_node();
//     //     let children = root.braid_split_node(8,1,1);
//     //     root.children = children.unwrap();
//     //
//     //     eprintln!("{:?}",root.braids[0]);
//     //
//     //     assert_eq!(&root.children[0].sample_names(),&vec!["1".to_string(),"3".to_string(),"4".to_string(),"5".to_string()]);
//     //     assert_eq!(&root.children[1].sample_names(),&vec!["0".to_string(),"6".to_string(),"7".to_string()]);
//     //
//     //     // assert_eq!(root.children[0].samples(),&vec!["1".to_string(),"4".to_string(),"5".to_string()]);
//     //     // assert_eq!(root.children[1].samples(),&vec!["0".to_string(),"3".to_string(),"6".to_string(),"7".to_string()]);
//     //
//     //
//     //     // assert_eq!(&root.children[0].sample_names(),&vec!["1".to_string(),"2".to_string(),"3".to_string(),"4".to_string(),"5".to_string()]);
//     //     // assert_eq!(&root.children[1].sample_names(),&vec!["0".to_string(),"2".to_string(),"6".to_string(),"7".to_string()]);
//     //     //
//     //     // assert_eq!(&root.children[0].output_table.full_values(),&vec![vec![-3.,0.,5.,-2.,-1.]]);
//     //     // assert_eq!(&root.children[1].output_table.full_values(),&vec![vec![10.,0.,15.,20.]]);
//     //
//     // }
//     //
//
//
//     #[test]
//     fn node_test_json() {
//         let n = simple_node();
//         let ns = n.to_string();
//     }
//
// }
