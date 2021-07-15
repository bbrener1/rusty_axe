// #![feature(test)]

// extern crate test;
// use test::Bencher;
//
#[macro_use]
extern crate serde_derive;

extern crate ndarray;

extern crate serde;
extern crate serde_json;
extern crate rand;
extern crate smallvec;
extern crate rayon;

extern crate num_traits;

mod rank_vector;
mod rank_matrix;
mod utils;
mod io;
pub mod node;
mod random_forest;
mod fast_nipal_vector;
mod hash_rv;

use ndarray::prelude::*;
use std::env;
use crate::io::Parameters;
use crate::random_forest::Forest;
use std::io::Error;

fn main() -> Result<(),Error> {
    let mut arg_iter = env::args();

    let parameters: Parameters = Parameters::read(&mut arg_iter);

    println!("Read parameters");

    let input = parameters.input_array();
    let output = parameters.output_array();

    let mut forest = Forest::initialize_from(input,output,parameters);

    forest.generate()
}



#[derive(Debug,Clone,Serialize,Deserialize,PartialEq,Eq,Hash)]
pub struct Feature {
    name: Option<String>,
    index: usize,
}

impl Feature {

    pub fn vec(input: Vec<usize>) -> Vec<Feature> {
        input.iter().map(|x| Feature::q(x)).collect()
    }

    pub fn nvec(input: &Vec<String>) -> Vec<Feature> {
        input.iter().enumerate().map(|(i,f)| Feature::new(f,&i)).collect()
    }

    pub fn q(index:&usize) -> Feature {
        Feature {name: None,index:*index}
    }

    pub fn new(name:&str,index:&usize) -> Feature {
        Feature {name: Some(name.to_owned()),index:*index}
    }

    pub fn name(&self) -> String {
        self.name.clone().unwrap_or(self.index.to_string())
    }

    pub fn index(&self) -> &usize {
        &self.index
    }
}

#[derive(Debug,Clone,Serialize,Deserialize,PartialEq,Eq,Hash)]
pub struct Sample {
    name: Option<String>,
    index: usize,
}

impl Sample {

    pub fn vec(input: Vec<usize>) -> Vec<Sample> {
        input.iter().map(|x| Sample::q(x)).collect()
    }

    pub fn nvec(input: &Vec<String>) -> Vec<Sample> {
        input.iter().enumerate().map(|(i,s)| Sample::new(s,&i)).collect()
    }

    pub fn q(index:&usize) -> Sample {
        Sample {name: None,index:*index}
    }

    pub fn new(name:&str,index:&usize) -> Sample {
        Sample {name: Some(name.to_owned()),index:*index}
    }

    pub fn name(&self) -> String {
        self.name.clone().unwrap_or(self.index.to_string())
    }

    pub fn index(&self) -> &usize {
        &self.index
    }

}

#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct Filter {
    reduction: Reduction,
    split: f64,
    orientation: bool,
}

impl Filter {

    // Filtering only works on matrices with full features, since the projection requires accurate
    // indices

    pub fn filter_matrix(&self, mtx: &Array2<f64>) -> Vec<usize> {
        let scores = self.reduction.score_matrix(mtx);
        if self.orientation {
            scores.into_iter().enumerate().filter(|(i,s)| *s > self.split).map(|(i,s)| i).collect()
        }
        else {
            scores.into_iter().enumerate().filter(|(i,s)| *s <= self.split).map(|(i,s)| i).collect()
        }
    }

    pub fn new(features:Vec<Feature>,means:Vec<f64>,scores:Vec<f64>,split:f64,orientation:bool) -> Filter {
        let reduction = Reduction {
            features,
            means,
            scores,
        };
        Filter {
            reduction,
            split,
            orientation,
        }
    }
}



#[derive(Clone,Serialize,Deserialize,Debug)]

// A forest projection allows us to form projections from multiple features of a random forest
// calculated elsewhere via NIPALS.

pub struct Reduction {
    features: Vec<Feature>,
    means: Vec<f64>,
    scores: Vec<f64>,
}

impl Reduction {

    pub fn new(features:Vec<Feature>,means:Vec<f64>,scores:Vec<f64>) -> Reduction {
        Reduction {
            features,
            means,
            scores,
        }
    }

    pub fn trivial(feature:Feature) -> Reduction {
        Reduction {
            features: vec![feature],
            means: vec![0.],
            scores: vec![1.],
        }
    }

// Scoring samples only works on a vector with full features because the feature indices must be accurate

    pub fn score_sample(&self,sample:&Array1<f64>) -> f64 {
        let mut score = 0.;
        for (feature,(mean,weight)) in self.features.iter().zip(self.means.iter().zip(self.scores.iter())) {
            let index = feature.index;
            score += (sample[index] - mean) * weight;
        }
        score
    }

// Likewise scoring a matrix only works on a matrix with full features, because feature indices must be accurate

    pub fn score_matrix(&self,mtx:&Array2<f64>) -> Array1<f64> {
        let means = Array1::from(self.means.clone());
        let scores = Array1::from(self.scores.clone());
        let mut selected = mtx.select(Axis(1),&self.features.iter().map(|f| f.index).collect::<Vec<usize>>()).clone();
        for mut r in selected.axis_iter_mut(Axis(0)) {
            r -= &means;
        };
        selected.dot(&scores)
    }

}
//
