use std::f64;
use ndarray::prelude::*;



// TODO: Bugcheck when a 1x1 is passed in

#[derive(Clone,Debug)]
pub struct Projector {
    means: Array1<f64>,
    scale_factors: Array1<f64>,
    array:Array2<f64>,
    weights:Array1<f64>,
    loadings:Array1<f64>,
    weight_norm: f64,
    smallnum: f64,
}

impl Projector {
    pub fn from(input:Array2<f64>) -> Projector {
        let smallnum = 10e-4 ;
        let mut copy = input.clone();
        let means = copy.mean_axis(Axis(0)).unwrap();
        let scale_factors = copy.sum_axis(Axis(1));
        copy = center(copy);
        let loadings = Array::ones(copy.dim().0);
        let weights = Array::ones(copy.dim().1);
        let weight_norm = f64::MAX;
        Projector {
            means,
            scale_factors,
            array:copy,
            weights,
            loadings,
            weight_norm,
            smallnum,
        }
    }

    pub fn calculate_projection(&mut self) -> Option<(Array1<f64>,Array1<f64>,Array1<f64>,Array1<f64>)> {
        self.weights.fill(1.);
        self.loadings.fill(1.);
        for _ in 0..10000 {
        // loop {
            self.weights = self.array.t().dot(&self.loadings);
            let new_norm = (self.weights.iter().map(|x| x.powi(2)).sum::<f64>()).sqrt();
            let delta = (self.weight_norm - new_norm).abs();
            self.weights.mapv_inplace(|x| x/new_norm);
            if delta < self.smallnum {
                self.weights *= -1.;
                self.loadings *= -1.;
                self.array -= &outer(&self.loadings,&self.weights);
                return Some((self.loadings.clone(),self.weights.clone(),self.means.clone(),self.scale_factors.clone()))
            }
            else {self.weight_norm = new_norm};
            self.loadings = self.array.dot(&self.weights);
        }
        None
    }

    pub fn calculate_n_projections(mut self,n:usize) -> Option<Projection> {
        let mut loadings = Array2::zeros((n,self.array.dim().0));
        let mut weights = Array2::zeros((n,self.array.dim().1));
        let mut means = Array2::zeros((n,self.array.dim().1));
        let mut scale_factors = Array2::zeros((n,self.array.dim().0));
        for i in 0..n {
            // println!("Projection {:?}",i);
            // println!("{:?}",self);
            let (n_loadings,n_scores,n_means,n_scale_factors)= self.calculate_projection()?;
            loadings.row_mut(i).assign(&n_loadings);
            weights.row_mut(i).assign(&n_scores);
            means.row_mut(i).assign(&n_means);
            scale_factors.row_mut(i).assign(&n_scale_factors);
        }
        let projection = Projection {
            loadings,
            weights,
            means,
            scale_factors
        };

        Some(projection)
    }

}

#[derive(Clone,Debug)]
pub struct Projection {
    pub loadings: Array2<f64>,
    pub weights: Array2<f64>,
    pub means: Array2<f64>,
    pub scale_factors: Array2<f64>,
}

pub fn project(arr:Array2<f64>,n:usize) -> Option<Projection> {
    let projector = Projector::from(arr);
    projector.calculate_n_projections(n)
}

fn outer(v1:&Array1<f64>,v2:&Array1<f64>) -> Array2<f64> {
    let m = v1.len();
    let n = v2.len();
    let mut output = Array2::zeros((m,n));
    for mut row in output.axis_iter_mut(Axis(0)) {
        row.assign(v2);
    }
    for mut column in output.axis_iter_mut(Axis(1)) {
        column *= v1;
    }
    output
}


fn center(mut input:Array2<f64>) -> Array2<f64> {
    let means = input.mean_axis(Axis(0)).unwrap();
    for mut row in input.axis_iter_mut(Axis(0)) {
        row -= &means;
    }
    input
}

#[cfg(test)]
mod nipals_tests {

    use super::*;
    use rand::{thread_rng,Rng};
    use rand::distributions::Standard;
    use crate::utils::iris_array;
    // use test::Bencher;


    // #[test]
    // fn iris_projection() {
    //     let iris = iris_array();
    //     let iris_m = &iris - 6.;
    //     println!("{:?}",iris);
    //     let projection = Projector::from(iris).calculate_n_projections(4);
    //     println!("{:?}",projection);
    //     panic!();
    // }
}
