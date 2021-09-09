use ndarray::prelude::*;
use std::cmp::Ordering;
use std::f64;


pub fn arr_from_vec2(vec:Vec<Vec<f64>>) -> Array2<f64> {
    let dim0 = vec.len();
    let dim1 = vec[0].len();
    Array::from_iter(vec.into_iter().flat_map(|a| a.into_iter())).into_shape((dim0,dim1)).unwrap()
}


pub fn argsort<I: Iterator<Item=T>,T: PartialOrd + Clone>(s:I) -> Vec<(usize,T)>{
    let mut paired: Vec<(usize,T)> = s.map(|t| t.clone()).enumerate().collect();
    paired.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    paired
}


pub fn argmax<T:Iterator<Item=U>,U:PartialOrd + PartialEq>(input: T) -> Option<usize> {
    let mut maximum: Option<(usize,U)> = None;
    for (j,val) in input.enumerate() {
        let check =
            if let Some((i,m)) = maximum.take() {
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
    maximum.map(|(i,_)| i)
}


pub fn argmax_v<T:Iterator<Item=U>,U:PartialOrd + PartialEq>(input: T) -> Option<(usize,U)> {
    let mut maximum: Option<(usize,U)> = None;
    for (j,val) in input.enumerate() {
        let check =
            if let Some((i,m)) = maximum.take() {
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
    maximum
}


pub fn argmin<T:Iterator<Item=U>,U:PartialOrd + PartialEq>(input: T) -> Option<usize> {
    let mut minimum: Option<(usize,U)> = None;
    for (j,val) in input.enumerate() {
        let check =
            if let Some((i,m)) = minimum.take() {
                match val.partial_cmp(&m).unwrap_or(Ordering::Less) {
                    Ordering::Greater => {Some((i,m))},
                    Ordering::Equal => {Some((i,m))},
                    Ordering::Less => {Some((j,val))},
                }
            }
            else {
                if val.partial_cmp(&val).is_some() { Some((j,val)) }
                else { None }
            };
        minimum = check;

    };
    minimum.map(|(i,_)| i)
}


pub fn argmin_v<T:Iterator<Item=U>,U:PartialOrd + PartialEq>(input: T) -> Option<(usize,U)> {
    let mut minimum: Option<(usize,U)> = None;
    for (j,val) in input.enumerate() {
        let check =
            if let Some((i,m)) = minimum.take() {
                match val.partial_cmp(&m).unwrap_or(Ordering::Less) {
                    Ordering::Greater => {Some((i,m))},
                    Ordering::Equal => {Some((i,m))},
                    Ordering::Less => {Some((j,val))},
                }
            }
            else {
                if val.partial_cmp(&val).is_some() { Some((j,val)) }
                else { None }
            };
        minimum = check;

    };
    minimum
}

pub trait ArgMinMax<I:PartialOrd+PartialEq> : Iterator<Item=I> + Sized {

    fn argmax(self) -> Option<usize> {
        argmax(self)
    }

    fn argmax_v(self) -> Option<(usize,I)> {
        argmax_v(self)
    }

    fn argmin(self) -> Option<usize> {
        argmin(self)
    }

    fn argmin_v(self) -> Option<(usize,I)> {
        argmin_v(self)
    }
}


pub trait ArgMinMaxII<I:PartialOrd+PartialEq> : IntoIterator<Item=I> + Sized {

    fn argmax(self) -> Option<usize> {
        argmax(self.into_iter())
    }

    fn argmax_v(self) -> Option<(usize,I)> {
        argmax_v(self.into_iter())
    }

    fn argmin(self) -> Option<usize> {
        argmin(self.into_iter())
    }

    fn argmin_v(self) -> Option<(usize,I)> {
        argmin_v(self.into_iter())
    }
}

impl<I:Iterator<Item=IT>,IT:PartialOrd+PartialEq> ArgMinMax<IT> for I {}
impl<I:IntoIterator<Item=IT>,IT:PartialOrd+PartialEq> ArgMinMaxII<IT> for I {}

pub trait ArgSort<I:PartialOrd+PartialEq+Clone> : Iterator<Item=I> + Sized {
    fn argsort(self) -> Vec<(usize,I)> {
        argsort(self)
    }
}
pub trait ArgSortII<I:PartialOrd+PartialEq+Clone> : IntoIterator<Item=I> + Sized {
    fn argsort(self) -> Vec<(usize,I)> {
        argsort(self.into_iter())
    }
}

impl<I:Iterator<Item=IT>,IT:PartialOrd+PartialEq+Clone> ArgSort<IT> for I {}
impl<I:IntoIterator<Item=IT>,IT:PartialOrd+PartialEq+Clone> ArgSortII<IT> for I {}


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

#[cfg(test)]
pub mod test_utils {

    use super::*;

    pub fn iris() -> Array2<f64> {
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


    pub fn slow_ssme(values: Vec<f64>) -> f64 {
        let median = slow_median(values.clone());
        values.iter().map(|x| (x - median).powi(2)).sum()
    }

    pub fn slow_sme(values: Vec<f64>) -> f64 {
        let median = slow_median(values.clone());
        values.iter().map(|x| (x - median).abs()).sum()
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



}
