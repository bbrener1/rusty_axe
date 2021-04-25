// #![feature(test)]

// extern crate test;
// use test::Bencher;
//
// #[macro_use]
// extern crate serde_derive;
// #[macro_use]
// extern crate ndarray;
//
// extern crate serde;
// extern crate serde_json;
// extern crate num_cpus;
// extern crate rand;
// extern crate time;
// extern crate smallvec;
// extern crate rayon;
//
// extern crate num_traits;
//
// mod rank_vector;
// mod rank_table;
// mod utils;
// mod io;
// mod segment_vector;
// pub mod io;
// pub mod node;
// mod random_forest;
// mod randutils;
// mod fast_nipal_vector;
//
// use std::env;
// use std::io as sio;
// use std::f64;
// use std::collections::{HashMap,HashSet};
// use std::fs::File;
// use std::io::stdin;
// use std::io::prelude::*;
// use std::cmp::PartialOrd;
// use std::cmp::Ordering;
// use std::fmt::Debug;
// use std::sync::Arc;
//
// use io::DispersionMode;
// use rank_vector::{RankVector,Node};



//
// fn read_header(location: &str) -> Vec<String> {
//
//     println!("Reading header: {}", location);
//
//     let mut header_map = HashMap::new();
//
//     let header_file = File::open(location).expect("Header file error!");
//     let mut header_file_iterator = sio::BufReader::new(&header_file).lines();
//
//     for (i,line) in header_file_iterator.by_ref().enumerate() {
//         let feature = line.unwrap_or("error".to_string());
//         let mut renamed = feature.clone();
//         let mut j = 1;
//         while header_map.contains_key(&renamed) {
//             renamed = [feature.clone(),j.to_string()].join("");
//             eprintln!("WARNING: Two individual features were named the same thing: {}",feature);
//             j += 1;
//         }
//         header_map.insert(renamed,i);
//     };
//
//     let mut header_inter: Vec<(String,usize)> = header_map.iter().map(|x| (x.0.clone().clone(),x.1.clone())).collect();
//     header_inter.sort_unstable_by_key(|x| x.1);
//     let header_vector: Vec<String> = header_inter.into_iter().map(|x| x.0).collect();
//
//     println!("Read {} lines", header_vector.len());
//
//     header_vector
// }
//
// fn read_sample_names(location: &str) -> Vec<String> {
//
//     let mut header_vector = Vec::new();
//
//     let sample_name_file = File::open(location).expect("Sample name file error!");
//     let mut sample_name_lines = sio::BufReader::new(&sample_name_file).lines();
//
//     for line in sample_name_lines.by_ref() {
//         header_vector.push(line.expect("Error reading header line!").trim().to_string())
//     }
//
//     header_vector
// }
//
//

// fn modified_competition_ranking(input:&[f64]) -> Vec<usize> {
//     let mut intermediate1 = input.iter().enumerate().collect::<Vec<(usize,&f64)>>();
//     intermediate1.sort_unstable_by(|a,b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Greater));
//     let mut intermediate2 = intermediate1.iter().enumerate().map(|(rank,(position,value))| (rank+1,(*position,*value))).collect::<Vec<(usize,(usize,&f64))>>();
//     for i in 0..(intermediate2.len().max(1)-1) {
//         let (r1,(_,v1)) = intermediate2[i];
//         let (_,(_,v2)) = intermediate2[i+1];
//         if v1 == v2 {
//             intermediate2[i+1].0 = r1;
//         }
//     }
//     intermediate2.sort_unstable_by(|a,b| ((a.1).0).cmp(&(b.1).0));
//     intermediate2.into_iter().map(|(rank,(position,value))| rank).collect()
// }
//
// fn tsv_format<T:Debug>(input:&Vec<Vec<T>>) -> String {
//
//     input.iter().map(|x| x.iter().map(|y| format!("{:?}",y)).collect::<Vec<String>>().join("\t")).collect::<Vec<String>>().join("\n")
//
// }
//
// fn median(input: &Vec<f64>) -> (usize,f64) {
//     let mut index = 0;
//     let mut value = 0.;
//
//     let mut sorted_input = input.clone();
//     sorted_input.sort_unstable_by(|a,b| a.partial_cmp(&b).unwrap_or(Ordering::Greater));
//
//     if sorted_input.len() % 2 == 0 {
//         index = sorted_input.len()/2;
//         value = (sorted_input[index-1] + sorted_input[index]) / 2.
//     }
//     else {
//         if sorted_input.len() % 2 == 1 {
//             index = (sorted_input.len()-1)/2;
//             value = sorted_input[index]
//         }
//         else {
//             panic!("Median failed!");
//         }
//     }
//     (index,value)
// }
//
// fn mean(input: &Vec<f64>) -> f64 {
//     input.iter().sum::<f64>() / (input.len() as f64)
// }
//
// fn covariance(vec1:&Vec<f64>,vec2:&Vec<f64>) -> f64 {
//
//     if vec1.len() != vec2.len() {
//         panic!("Tried to compute covariance for unequal length vectors: {}, {}",vec1.len(),vec2.len());
//     }
//
//     let mean1: f64 = mean(vec1);
//     let mean2: f64 = mean(vec1);
//
//     let covariance = vec1.iter().zip(vec2.iter()).map(|(x,y)| (x - mean1) * (y - mean2)).sum::<f64>() / (vec1.len() as f64 - 1.);
//
//     if covariance.is_nan() {0.} else {covariance}
//
// }
//
// pub fn variance(input: &Vec<f64>) -> f64 {
//
//     let mean = mean(input);
//
//     let var = input.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (input.len() as f64 - 1.).max(1.);
//
//     if var.is_nan() {0.} else {var}
// }
//
//
// pub fn std_dev(input: &Vec<f64>) -> f64 {
//
//     let mean = mean(input);
//
//     let std_dev = (input.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (input.len() as f64 - 1.).max(1.)).sqrt();
//
//     if std_dev.is_nan() {0.} else {std_dev}
// }
//
// fn pearsonr(vec1:&Vec<f64>,vec2:&Vec<f64>) -> f64 {
//
//     if vec1.len() != vec2.len() {
//         panic!("Tried to compute correlation for unequal length vectors: {}, {}",vec1.len(),vec2.len());
//     }
//
//     let mean1: f64 = mean(vec1);
//     let mean2: f64 = mean(vec2);
//
//     let dev1: Vec<f64> = vec1.iter().map(|x| (x - mean1)).collect();
//     let dev2: Vec<f64> = vec2.iter().map(|x| (x - mean2)).collect();
//
//     let covariance = dev1.iter().zip(dev2.iter()).map(|(x,y)| x * y).sum::<f64>() / (vec1.len() as f64 - 1.);
//
//     let std_dev1 = (dev1.iter().map(|x| x.powi(2)).sum::<f64>() / (vec1.len() as f64 - 1.).max(1.)).sqrt();
//     let std_dev2 = (dev2.iter().map(|x| x.powi(2)).sum::<f64>() / (vec2.len() as f64 - 1.).max(1.)).sqrt();
//
//     // println!("{},{}", std_dev1,std_dev2);
//
//     let r = covariance / (std_dev1*std_dev2);
//
//     if r.is_nan() {0.} else {r}
//
// }
//
//
// pub fn jaccard(a:&[bool],b:&[bool]) -> f64 {
//     assert!(a.len() == b.len());
//     assert!(a.len() != 0);
//     let i = a.iter().zip(b.iter()).filter(|(a,b)| **a && **b).count() as f64;
//     let u = a.iter().zip(b.iter()).filter(|(a,b)| **a || **b).count() as f64;
//     return 1. - (i/u)
// }
//
// #[cfg(test)]
// pub mod tree_lib_tests {
//
//     use super::*;
//
//     #[test]
//     fn test_argmin() {
//
//         let na = std::f64::NAN;
//
//         assert_eq!(argmin(&vec![1.]),Some((0,1.)));
//         assert_eq!(argmin(&vec![1.,2.]),Some((0,1.)));
//         assert_eq!(argmin(&vec![]),None);
//         assert_eq!(argmin(&vec![1.,na]),Some((0,1.)));
//         assert_eq!(argmin(&vec![na,1.]),Some((1,1.)));
//         assert_eq!(argmin(&vec![na,na]),None);
//         assert_eq!(argmin(&vec![1.,1.]),Some((0,1.)));
//
//     }
//
//     #[test]
//     fn test_argmax() {
//
//         let na = std::f64::NAN;
//
//         assert_eq!(argmax(&vec![1.]),Some((0,1.)));
//         assert_eq!(argmax(&vec![1.,2.]),Some((1,2.)));
//         assert_eq!(argmax(&vec![]),None);
//         assert_eq!(argmax(&vec![1.,na]),Some((0,1.)));
//         assert_eq!(argmax(&vec![na,1.]),Some((1,1.)));
//         assert_eq!(argmax(&vec![na,na]),None);
//         assert_eq!(argmax(&vec![1.,1.]),Some((0,1.)));
//
//
//     }
//
//     #[test]
//     fn test_gn_argmax() {
//
//         let na = std::f64::NAN;
//
//         assert_eq!(gn_argmax(vec![1.].iter()),Some((0)));
//         assert_eq!(gn_argmax(vec![1.,2.].iter()),Some((1)));
//         assert_eq!(gn_argmax(Vec::<f64>::new().iter()),None);
//         assert_eq!(gn_argmax(vec![1.,na].iter()),Some((0)));
//         assert_eq!(gn_argmax(vec![na,1.].iter()),Some((1)));
//         assert_eq!(gn_argmax(vec![na,na].iter()),None);
//         assert_eq!(gn_argmax(vec![1.,1.].iter()),Some((0)));
//
//
//     }
//
//     #[test]
//     fn test_modified_competition_ranking() {
//
//         let a = vec![2.,0.,1.,2.,0.,3.];
//         let b = modified_competition_ranking(&a);
//         let c = vec![4,1,3,4,1,6];
//
//         let d: Vec<f64> = Vec::new();
//         let e: Vec<usize> = Vec::new();
//
//         assert_eq!(b,c);
//         assert_eq!(e,modified_competition_ranking(&d));
//
//     }
//
//     #[test]
//     fn test_jaccard() {
//
//         let a = vec![true,true,false,false,true];
//         let b = vec![false,true,false,true,true];
//
//         assert_eq!(jaccard(&a, &b),0.5);
//
//     }
//
// }
