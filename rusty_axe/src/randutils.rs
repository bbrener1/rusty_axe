use rand::{Rng,ThreadRng,thread_rng};
use ndarray::prelude::*;
use std::cmp::Ordering;
use std::f64;
use std::mem::swap;
use std::ops::Add;
use std::fmt::Debug;
use std::io;
use std::io::prelude::*;
use std::fs::File;
use std::fs::OpenOptions;
use crate::modified_competition_ranking;

pub fn logit(p:f64) -> f64 {
    (p/(1.-p)).ln()
}

pub fn weighted_choice<T: Rng>(weights: &Vec<f64>, rng: &mut T) -> usize {

    let mut descending_weight:f64 = weights.iter().sum();

    let choice = rng.gen::<f64>() * descending_weight;

    for (i,weight) in weights.iter().enumerate() {
        descending_weight -= *weight;
        // println!("descending:{}",descending_weight);
        if choice > descending_weight {
            // println!("choice:{}",choice);
            return i
        }
    }

    0
}

pub fn weighted_sampling_with_replacement(draws: usize,weights: &Vec<f64>) -> Vec<usize> {
    let mut rng = thread_rng();
    let mut drawn_indecies: Vec<usize> = Vec::with_capacity(draws);
    let mut weight_sum: f64 = weights.iter().sum();

    let mut weighted_choices: Vec<f64> = (0..draws).map(|_| rng.gen_range::<f64,f64,f64>(0.,weight_sum)).collect();
    weighted_choices.sort_unstable_by(|a,b| a.partial_cmp(&b).unwrap_or(Ordering::Greater));

    let mut current_choice = weighted_choices.pop().unwrap_or(-1.);

    'f_loop: for (i,element) in weights.iter().enumerate() {
        weight_sum -= *element;
        // println!("descending:{}",descending_weight);
        'w_loop: while weight_sum < current_choice {

                // println!("choice:{}",choice);

                // if weighted_choices.len()%1000 == 0 {
                //     if weighted_choices.len() > 0 {
                //         // println!("{}",weighted_choices.len());
                //     }
                // }

            drawn_indecies.push(i);
            if let Some(new_choice) = weighted_choices.pop() {
                current_choice = new_choice;
            }
            else {
                break 'f_loop;
            }
        }

    }
    drawn_indecies
}



pub fn weighted_sampling_with_increasing_similarity(draws:usize,weights:Option<&Vec<f64>>,similarity:&Array<f64,Ix2>) -> Vec<usize> {
    let n = similarity.shape()[0];
    let mut log_odds: Array<f64,Ix1> = Array::zeros(n);
    log_odds += &weights.unwrap_or(&vec![1.;n]).iter().map(|v| v.ln()).collect::<Array<f64,Ix1>>();
    let mut selected_indecies = Vec::with_capacity(draws);
    let mut rng = thread_rng();

    for _ in 0..draws {

        let local_odds: Vec<f64> = log_odds.iter().map(|x:&f64| x.exp()).collect();

        let selection = weighted_choice(&local_odds, &mut rng);

        for (sim,lg_od) in similarity.row(selection).iter().zip(log_odds.iter_mut()) {
            *lg_od += sim.ln();
        }

        let log_max: f64 = *log_odds.iter().max_by(|a,b| a.partial_cmp(&b).unwrap_or(Ordering::Less)).unwrap_or(&0.);

        log_odds -= log_max;

        selected_indecies.push(selection);
    }

    selected_indecies
}

pub fn read_matrix(location:&str) -> Array<f64,Ix2> {

    let count_array_file = File::open(location).expect("Count file error!");
    let mut count_array_lines = io::BufReader::new(&count_array_file).lines();

    let mut count_array: Vec<Vec<f64>> = Vec::new();

    for (i,line) in count_array_lines.by_ref().enumerate() {
        let mut gene_vector = Vec::new();
        let gene_line = line.expect("Readline error");
        for (j,gene) in gene_line.split_whitespace().enumerate() {
            if j == 0 && i%200==0{
                print!("\n");
            }
            if i%200==0 && j%200 == 0 {
                print!("{} ", gene.parse::<f64>().unwrap_or(-1.) );
            }
            // if !((gene.0 == 1686) || (gene.0 == 4660)) {
            //     continue
            // }
            match gene.parse::<f64>() {
                Ok(exp_val) => {
                    gene_vector.push(exp_val);
                },
                Err(msg) => {
                    if gene != "nan" && gene != "NAN" {
                        println!("Couldn't parse a cell in the text file, Rust sez: {:?}",msg);
                        println!("Cell content: {:?}", gene);
                    }
                    gene_vector.push(f64::NAN);
                }
            }
        }
        count_array.push(gene_vector);
        if i % 100 == 0 {
            println!("{}", i);
        }
    };

    println!("===========");
    println!("{},{}", count_array.len(),count_array.get(0).unwrap_or(&vec![]).len());

    let (r,c) = (count_array.len(),count_array.get(0).unwrap_or(&vec![]).len());
    let mut array = Array::zeros((r,c));
    for (mut a,c) in array.axis_iter_mut(Axis(0)).zip(count_array.into_iter()) {
        a.assign(&mut Array::from_vec(c));
    }
    array
}

pub fn correlation(p1: ArrayView<f64,Ix1>,p2: ArrayView<f64,Ix1>) -> f64 {

    if p1.len() != p2.len() {
        panic!("Tried to compute correlation for unequal length vectors: {}, {}",p1.len(),p2.len());
    }

    let mean1: f64 = p1.sum() / p1.shape()[0] as f64;
    let mean2: f64 = p2.sum() / p2.shape()[0] as f64;

    let dev1: Vec<f64> = p1.iter().map(|x| (x - mean1)).collect();
    let dev2: Vec<f64> = p2.iter().map(|x| (x - mean2)).collect();

    let covariance = dev1.iter().zip(dev2.iter()).map(|(x,y)| x * y).sum::<f64>() / (p1.len() as f64 - 1.);

    let std_dev1 = (dev1.iter().map(|x| x.powi(2)).sum::<f64>() / (p1.len() as f64 - 1.).max(1.)).sqrt();
    let std_dev2 = (dev2.iter().map(|x| x.powi(2)).sum::<f64>() / (p2.len() as f64 - 1.).max(1.)).sqrt();

    // println!("{},{}", std_dev1,std_dev2);

    let r = covariance / (std_dev1*std_dev2);

    if r.is_nan() {0.} else {r}

}

pub fn correlation_matrix(mtx:&ArrayView<f64,Ix2>) -> Array<f64,Ix2> {

    let r = mtx.shape()[0];
    let c = mtx.shape()[1];

    eprintln!("r:{},c:{}",r,c);

    let mut correlations: Array<f64,Ix2> = Array::zeros((r,c));

    let means = mtx.mean_axis(Axis(0));

    eprintln!("means");
    eprintln!("{:?}",means);

    let mut centered: Array<f64,Ix2> = Array::zeros((r,c));

    for i in 0..c {
        centered.column_mut(i).assign(&(&mtx.column(i) - means[i]));
    }

    let covariance_est = centered.t().dot(&centered) / r as f64;

    let std = covariance_est.diag().mapv(|v| v.sqrt());

    let std_outer = outer_product(&std.view(), &std.view());

    let r = covariance_est / std_outer;

    return r

}

pub fn outer_product(a:&ArrayView<f64,Ix1>,b:&ArrayView<f64,Ix1>) -> Array<f64,Ix2> {
    let ad = a.dim();
    let bd = b.dim();

    let mut out = Array::zeros((ad,bd));

    for (mut out_r,ai) in out.axis_iter_mut(Axis(0)).zip(a.iter()) {
        out_r.assign(&(*ai * b));
    }

    out
}


pub fn ranking_matrix(mtx: &ArrayView<f64,Ix2>) -> Array<f64,Ix2> {
    let mut output = mtx.to_owned();
    for i in 0..mtx.shape()[1] {
        let c: Vec<f64> = mtx.column(i).to_vec();
        let mut ranking: Array<f64,Ix1> = Array::from_vec(modified_competition_ranking(&c)).mapv(|v| v as f64);
        output.column_mut(i).assign(&mut ranking);
    }
    output
}

pub fn spearman_matrix(mtx: &ArrayView<f64,Ix2>) -> Array<f64,Ix2> {
    let rank_mtx = ranking_matrix(mtx);
    return correlation_matrix(&rank_mtx.view())
}

pub fn tsv_format<T:Debug,D:ndarray::RemoveAxis>(input:&Array<T,D>) -> String {

    input.axis_iter(Axis(0)).map(|x| x.iter().map(|y| format!("{:?}",y)).collect::<Vec<String>>().join("\t")).collect::<Vec<String>>().join("\n")

}

pub fn write_array<T:Debug,D:ndarray::RemoveAxis>(input:&Array<T,D>,location:&str) -> Result<(),std::io::Error> {
    let mut handle = OpenOptions::new().create(true).append(true).open(location)?;
    handle.write(tsv_format(input).as_bytes())?;

    Ok(())

}

#[cfg(test)]
pub mod randutil_test {

    use super::*;

    fn load_nesterowa() -> Array<f64,Ix2> {
        read_matrix("../../work/nesterowa_counts.txt")
    }

    fn load_iris() -> Array<f64,Ix2> {
        read_matrix("/Users/boris/taylor/vision/rust_prototype/rusty_lumberjack/testing/iris.trunc")
    }

    #[test]
    fn randutil_test() {
        let iris = load_iris();
        eprintln!("Ranking");
        eprintln!("{:?}",ranking_matrix(&iris.view()));
        eprintln!("Corr");
        eprintln!("{:?}",correlation_matrix(&iris.view()));
        eprintln!("Spearman's");
        eprintln!("{:?}",spearman_matrix(&iris.view()));
        panic!();
    }

    #[test]
    fn resampling_test() {
        let iris = load_iris();
        let absolute_rho = spearman_matrix(&iris.view()).mapv(|v| v.abs());
        eprintln!("{:?}",absolute_rho);
        for _ in 0..100 {
            eprintln!("{:?}",weighted_sampling_with_increasing_similarity(10, None, &absolute_rho));
        }
        panic!();
    }
}
