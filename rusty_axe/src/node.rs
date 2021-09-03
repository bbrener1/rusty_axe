
use std::f64;
use serde_json;

use ndarray::prelude::*;

extern crate rand;
use rand::prelude::*;

use crate::rank_matrix::{RankMatrix};
use crate::Feature;
use crate::Sample;
use crate::io::Parameters;
use crate::io::DispersionMode;
use crate::Filter;
use crate::random_forest::Prototype;

use crate::fast_nipal_vector::{project,Projection};

#[derive(Clone,Debug)]
pub struct Node {

    pub prototype: bool,

    input_features: Vec<Feature>,
    output_features: Vec<Feature>,
    samples: Vec<Sample>,

    filter: Option<Filter>,
    input_projection: Option<Projection>,
    output_projection: Option<Projection>,

    means: Option<Vec<f64>>,
    medians: Option<Vec<f64>>,

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

            means: None,
            medians: None,

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

            means: None,
            medians: None,

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

        let sample_index_bootstrap: Vec<usize> = (0..parameters.sample_subsample).map(|_| rng.gen_range(0..self.samples.len())).collect();
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

            means: None,
            medians: None,

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
    //

    pub fn candidate_filters(&mut self,prototype:&Prototype,parameters:&Parameters) -> Vec<(Filter,Filter)> {

        let input_ranks = self.input_rank_matrix(prototype, parameters);
        let output_ranks = self.output_rank_matrix(prototype, parameters);

        let minima = RankMatrix::split_candidates(input_ranks,output_ranks);

        let input_features = self.input_features.clone();

        let filters = if parameters.reduce_input {
            minima.into_iter()
            .map(|(local_feature,_,local_threshold)| {

                let Projection {
                    weights, means, ..
                } = self.input_projection(prototype,parameters);

                let left_filter = Filter::new(input_features.clone(),means.row(local_feature).to_vec(),weights.row(local_feature).to_vec(),local_threshold,false);
                let right_filter = Filter::new(input_features.clone(),means.row(local_feature).to_vec(),weights.row(local_feature).to_vec(),local_threshold,true);
                (left_filter,right_filter)
            }).collect()
        }
        else {
            minima.into_iter()
            .map(|(local_feature,_,local_threshold)| {
                let left_filter = Filter::new(vec![input_features[local_feature].clone(),],vec![0.,],vec![1.,],local_threshold,false);
                let right_filter = Filter::new(vec![input_features[local_feature].clone(),],vec![0.,],vec![1.,],local_threshold,true);
                (left_filter,right_filter)
            }).collect()
        };

        filters

    }

    pub fn local_split(&mut self,prototype:&Prototype,parameters:&Parameters) -> Option<(Filter,Filter)> {
        let candidates = self.candidate_filters(prototype, parameters);
        candidates.get(0).map(|t| t.clone())
    }

    pub fn split(&mut self, prototype:&Prototype,parameters:&Parameters) -> Option<&mut [Node]> {

        if self.depth >= parameters.depth_cutoff {
            return None
        };

        if !self.prototype {panic!("Attempted to split on a non-prototype node")};

        let mut slim_node = self.derive_bootstrap(parameters);
        let candidate_filters = slim_node.candidate_filters(prototype,parameters);
        let sample_indices: Vec<usize> = self.samples.iter().map(|s| s.index).collect();
        let inputs = prototype.input_array.select(Axis(0),&sample_indices);

        let mut selected_candidates = None;

        for (f_left,f_right) in candidate_filters {
            let left_samples: Vec<Sample> = f_left.filter_matrix(&inputs).into_iter().map(|i| self.samples[i].clone()).collect();
            let right_samples: Vec<Sample> = f_right.filter_matrix(&inputs).into_iter().map(|i| self.samples[i].clone()).collect();
            if left_samples.len() > parameters.leaf_size_cutoff && right_samples.len() > parameters.leaf_size_cutoff {
                selected_candidates = Some((f_left,f_right,left_samples,right_samples));
                break
            }
        }

        let (left_filter,right_filter,left_samples,right_samples) = selected_candidates?;

        let left_child = self.derive_prototype(left_samples, Some(left_filter));
        let right_child = self.derive_prototype(right_samples, Some(right_filter));

        self.means = Some(self.output_rank_matrix(prototype, parameters).means());
        self.medians = Some(self.output_rank_matrix(prototype, parameters).medians());

        let children = vec![left_child,right_child];
        self.children = children;

        Some(&mut self.children)

    }

    pub fn grow(&mut self, prototype:&Prototype, parameters:&Parameters) {
        if let Some(children) = self.split(prototype,parameters) {
            for child in children.iter_mut() {
                child.grow(prototype,parameters);
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
            means: self.means.clone(),
            medians: self.medians.clone(),
            filter: self.filter.clone(),
            depth: self.depth,
            children: serialized_children,
        }
    }

}

#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct SerialNode {
    samples: Vec<usize>,
    means: Option<Vec<f64>>,
    medians: Option<Vec<f64>>,
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


    pub fn to_string(self) -> Result<String,serde_json::Error> {
        serde_json::to_string(&self)
    }

    pub fn from_str(input:&str) -> Result<SerialNode,serde_json::Error> {
        serde_json::from_str(input)
    }
}


#[cfg(test)]
mod node_testing {

    use super::*;
    use crate::io::NormMode;

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
        Prototype::new(iris(),iris(),&parameters)
    }

    pub fn iris_node(parameters:&Parameters) -> Node {
        let input = (0..4).map(|i| Feature::q(&i)).collect::<Vec<Feature>>();
        let output = (0..4).map(|i| Feature::q(&i)).collect::<Vec<Feature>>();
        let samples = (0..150).map(|i| Sample::q(&i)).collect::<Vec<Sample>>();
        Node::prototype(&input, &output, &samples, &parameters)
    }

    #[test]
    fn node_test_iris() {
        let mut parameters = Parameters::empty();
        parameters.standardize = true;
        parameters.reduce_input = true;
        parameters.reduce_output = true;
        parameters.reduction = 4;
        parameters.norm_mode = NormMode::L1;
        parameters.dispersion_mode = DispersionMode::MAD;
        parameters.split_fraction_regularization = 1.;
        let mut root = iris_node(&parameters);
        let prototype = iris_prototype();
        let (left,right) = root.local_split(&prototype, &parameters).unwrap();
        println!("Filters: {:?}", (&left,&right));
        let left_children = left.filter_matrix(&prototype.input_array);
        let right_children = right.filter_matrix(&prototype.input_array);
        eprintln!("{:?}",parameters);
        eprintln!("{:?}",left_children);
        eprintln!("{:?}",right_children);
        eprintln!("{:?}",left_children.len());
        eprintln!("{:?}",right_children.len());
        panic!();
    }


}
