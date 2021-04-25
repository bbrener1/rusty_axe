use ndarray::prelude::*;
use std::cmp::Ordering;
use std::f64;
use crate::fast_nipal_vector::{Projector,Projection};

fn mean(input: &Vec<f64>) -> f64 {
    input.iter().sum::<f64>() / (input.len() as f64)
}

fn covariance(vec1:&Vec<f64>,vec2:&Vec<f64>) -> f64 {

    if vec1.len() != vec2.len() {
        panic!("Tried to compute covariance for unequal length vectors: {}, {}",vec1.len(),vec2.len());
    }

    let mean1: f64 = mean(vec1);
    let mean2: f64 = mean(vec1);

    let covariance = vec1.iter().zip(vec2.iter()).map(|(x,y)| (x - mean1) * (y - mean2)).sum::<f64>() / (vec1.len() as f64 - 1.);

    if covariance.is_nan() {0.} else {covariance}

}


pub fn project_vec(vec:Vec<Vec<f64>>,n:usize) -> Projection {
    let arr = arr_from_vec2(vec);
    let projector = Projector::from(arr);
    let output = projector.calculate_n_projections(n).unwrap();
    output
}

pub fn arr_from_vec2(vec:Vec<Vec<f64>>) -> Array2<f64> {
    let dim0 = vec.len();
    let dim1 = vec[0].len();
    Array::from_iter(vec.into_iter().flat_map(|a| a.into_iter())).into_shape((dim0,dim1)).unwrap()
}

pub fn vec2_from_arr(arr: &Array2<f64>) -> Vec<Vec<f64>> {
    arr.axis_iter(Axis(0)).map(|r| r.to_vec()).collect()
}

pub fn variance(input: &Vec<f64>) -> f64 {

    let mean = mean(input);

    let var = input.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (input.len() as f64 - 1.).max(1.);

    if var.is_nan() {0.} else {var}
}


pub fn std_dev(input: &Vec<f64>) -> f64 {

    let mean = mean(input);

    let std_dev = (input.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (input.len() as f64 - 1.).max(1.)).sqrt();

    if std_dev.is_nan() {0.} else {std_dev}
}

pub fn pearsonr(vec1:&Vec<f64>,vec2:&Vec<f64>) -> f64 {

    if vec1.len() != vec2.len() {
        panic!("Tried to compute correlation for unequal length vectors: {}, {}",vec1.len(),vec2.len());
    }

    let mean1: f64 = mean(vec1);
    let mean2: f64 = mean(vec2);

    let dev1: Vec<f64> = vec1.iter().map(|x| (x - mean1)).collect();
    let dev2: Vec<f64> = vec2.iter().map(|x| (x - mean2)).collect();

    let covariance = dev1.iter().zip(dev2.iter()).map(|(x,y)| x * y).sum::<f64>() / (vec1.len() as f64 - 1.);

    let std_dev1 = (dev1.iter().map(|x| x.powi(2)).sum::<f64>() / (vec1.len() as f64 - 1.).max(1.)).sqrt();
    let std_dev2 = (dev2.iter().map(|x| x.powi(2)).sum::<f64>() / (vec2.len() as f64 - 1.).max(1.)).sqrt();

    // println!("{},{}", std_dev1,std_dev2);

    let r = covariance / (std_dev1*std_dev2);

    if r.is_nan() {0.} else {r}

}

fn row_echelon(mtx: &Vec<Vec<f64>>) {
    let mut working = matrix_flip(mtx);
    let dim = mtx_dim(&working);
    let mut column_order: Vec<usize> = (0..dim.0).collect();
    let mut row_order: Vec<usize> = (0..dim.1).collect();
    for i in 0..working.len() {
        let column = &working[i];
        let first_value = column.iter().find(|x| x.abs() > 0.);

    };
}


pub fn l1_sum(mtx_in:&Vec<Vec<f64>>, weights: &[f64]) -> Vec<f64> {
    let weight_sum = weights.iter().sum::<f64>();

    let sample_sums = mtx_in.iter().map(|sample| {
        sample.iter().enumerate().map(|(i,feature)| feature * weights[i] ).sum::<f64>() / weight_sum
    }).map(|sum| if sum.is_normal() || sum == 0. {sum} else {f64::INFINITY}).collect();
    sample_sums
}

pub fn l2_sum(mtx_in:&Vec<Vec<f64>>, weights: &[f64]) -> Vec<f64> {
    let weight_sum = weights.iter().sum::<f64>();

    let sample_sums = mtx_in.iter().map(|sample| {
        sample.iter().enumerate().map(|(i,feature)| feature.powi(2) * weights[i] ).sum::<f64>() / weight_sum
    }).map(|sum| if sum.is_normal() || sum == 0. {sum} else {f64::INFINITY}).collect();
    sample_sums
}

pub fn argmin(in_vec: &[f64]) -> Option<(usize,f64)> {
    let mut minimum = None;
    for (j,&val) in in_vec.iter().enumerate() {
        let check = if let Some((i,m)) = minimum.take() {
            match val.partial_cmp(&m).unwrap_or(Ordering::Greater) {
                Ordering::Less => {Some((j,val))},
                Ordering::Equal => {Some((i,m))},
                Ordering::Greater => {Some((i,m))},
            }
        }
        else {
            if !val.is_nan() {
                Some((j,val))
            }
            else {
                None
            }
        };
        minimum = check;

    };
    minimum
}

pub fn argmax(in_vec: &[f64]) -> Option<(usize,f64)> {
    let mut maximum = None;
    for (j,&val) in in_vec.iter().enumerate() {
        let check = if let Some((i,m)) = maximum.take() {
            match val.partial_cmp(&m).unwrap_or(Ordering::Less) {
                Ordering::Less => {Some((i,m))},
                Ordering::Equal => {Some((i,m))},
                Ordering::Greater => {Some((j,val))},
            }
        }
        else {
            if !val.is_nan() {
                Some((j,val))
            }
            else {
                None
            }
        };
        maximum = check;

    };
    maximum
}

pub fn gn_argmax<T:Iterator<Item=U>,U:PartialOrd + PartialEq>(input: T) -> Option<usize> {
    let mut maximum: Option<(usize,U)> = None;
    for (j,val) in input.enumerate() {
        let check = if let Some((i,m)) = maximum.take() {
            match val.partial_cmp(&m).unwrap_or(Ordering::Less) {
                Ordering::Less => {Some((i,m))},
                Ordering::Equal => {Some((i,m))},
                Ordering::Greater => {Some((j,val))},
            }
        }
        else {
            if val.partial_cmp(&val).is_some() { Some((j,val)) }
            else { None }
        };
        maximum = check;

    };
    maximum.map(|(i,m)| i)
}

pub fn jaccard(a:&[bool],b:&[bool]) -> f64 {
    assert!(a.len() == b.len());
    assert!(a.len() != 0);
    let i = a.iter().zip(b.iter()).filter(|(a,b)| **a && **b).count() as f64;
    let u = a.iter().zip(b.iter()).filter(|(a,b)| **a || **b).count() as f64;
    return 1. - (i/u)
}

pub fn argsort(input: &Vec<f64>) -> Vec<(usize,f64)> {
    let mut intermediate1 = input.iter().enumerate().collect::<Vec<(usize,&f64)>>();
    intermediate1.sort_unstable_by(|a,b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Greater));
    let mut intermediate2 = intermediate1.iter().enumerate().collect::<Vec<(usize,&(usize,&f64))>>();
    intermediate2.sort_unstable_by(|a,b| ((a.1).0).cmp(&(b.1).0));
    let out = intermediate2.iter().map(|x| (x.0,((x.1).1).clone())).collect();
    out
}

pub fn slow_median(values: Vec<f64>) -> f64 {
    let median: f64;
    if values.len() < 1 {
        return 0.
    }

    if values.len()%2==0 {
        median = (values[values.len()/2] + values[values.len()/2 - 1]) as f64 / 2.;
    }
    else {
        median = values[(values.len()-1)/2];
    }

    median

}

pub fn slow_mad(values: Vec<f64>) -> f64 {
    let median: f64;
    if values.len() < 1 {
        return 0.
    }
    if values.len()%2==0 {
        median = (values[values.len()/2] + values[values.len()/2 - 1]) as f64 / 2.;
    }
    else {
        median = values[(values.len()-1)/2];
    }

    let mut abs_deviations: Vec<f64> = values.iter().map(|x| (x-median).abs()).collect();

    abs_deviations.sort_by(|a,b| a.partial_cmp(&b).unwrap_or(Ordering::Greater));

    let mad: f64;
    if abs_deviations.len()%2==0 {
        mad = (abs_deviations[abs_deviations.len()/2] + abs_deviations[abs_deviations.len()/2 - 1]) as f64 / 2.;
    }
    else {
        mad = abs_deviations[(abs_deviations.len()-1)/2];
    }

    mad

}

pub fn slow_ssme(values: Vec<f64>) -> f64 {
    let median = slow_median(values.clone());
    values.iter().map(|x| (x - median).powi(2)).sum()
}

pub fn slow_sme(values: Vec<f64>) -> f64 {
    let median = slow_median(values.clone());
    values.iter().map(|x| (x - median).abs()).sum()
}


pub fn matrix_flip<T:Clone>(in_mat: &Vec<Vec<T>>) -> Vec<Vec<T>> {

    let dim = mtx_dim(in_mat);

    let mut out = vec![Vec::with_capacity(dim.0);dim.1];

    for (i,iv) in in_mat.iter().enumerate() {
        for (j,jv) in iv.iter().enumerate() {
            out[j].push(jv.clone());
        }
    }

    out
}

pub fn mtx_dim<T>(in_mat: &Vec<Vec<T>>) -> (usize,usize) {
    (in_mat.len(),in_mat.get(0).unwrap_or(&vec![]).len())
}


pub fn iris_array() -> Array2<f64> {
    array![[5.1,3.5,1.4,0.2],
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
    [5.9,3.0,5.1,1.8]]
}
