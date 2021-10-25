#[allow(dead_code)]

use std::cmp::Ordering;

pub fn argsort<I: Iterator<Item=T>,T: PartialOrd + Clone>(s:I) -> Vec<(usize,T)>{
    let mut paired: Vec<(usize,T)> = s.map(|t| t.clone()).enumerate().collect();
    paired.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    paired
}


pub fn argmax<T:Iterator<Item=U>,U:PartialOrd + PartialEq>(input: T) -> Option<usize> {
    let mut maximum = ArgMinMax::argmax_v(input);
    maximum.map(|(i,_)| i)
}


pub fn argmax_v<T:Iterator<Item=U>,U:PartialOrd + PartialEq>(input: T) -> Option<(usize,U)> {
    let mut maximum: Option<(usize,U)> = None;
    for (j,val) in input.enumerate() {
        let check =
            if let Some((i,m)) = maximum.take() {
                // Note the unwrap or defaults nan to less than
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
    let minimum = ArgMinMax::argmin_v(input);
    minimum.map(|(i,v)| i)
}


pub fn argmin_v<T:Iterator<Item=U>,U:PartialOrd + PartialEq>(input: T) -> Option<(usize,U)> {
    let mut minimum: Option<(usize,U)> = None;
    for (j,val) in input.enumerate() {
        let check =
            if let Some((i,m)) = minimum.take() {
                // Note unwrap or defaults nan to greater
                match val.partial_cmp(&m).unwrap_or(Ordering::Greater) {
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


mod arg_testing {

    use super::*;

    #[test]
    fn test_argmin() {

        let na = std::f64::NAN;

        assert_eq!(vec![1.].argmin_v(),Some((0,1.)));
        assert_eq!(vec![1.,2.].argmin_v(),Some((0,1.)));
        assert_eq!(Vec::<f64>::new().argmin_v(),None);
        assert_eq!(vec![1.,na].argmin_v(),Some((0,1.)));
        assert_eq!(vec![na,1.].argmin_v(),Some((1,1.)));
        assert_eq!(vec![na,na].argmin_v(),None);
        assert_eq!(vec![1.,1.].argmin_v(),Some((0,1.)));

    }

    #[test]
    fn test_argmax() {

        let na = std::f64::NAN;

        assert_eq!(vec![1.].argmax_v(),Some((0,1.)));
        assert_eq!(vec![1.,2.].argmax_v(),Some((1,2.)));
        assert_eq!(Vec::<f64>::new().argmax_v(),None);
        assert_eq!(vec![1.,na].argmax_v(),Some((0,1.)));
        assert_eq!(vec![na,1.].argmax_v(),Some((1,1.)));
        assert_eq!(vec![na,na].argmax_v(),None);
        assert_eq!(vec![1.,1.].argmax_v(),Some((0,1.)));


    }
}
