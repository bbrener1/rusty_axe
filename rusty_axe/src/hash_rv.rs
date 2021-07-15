use std::f64;
use std::collections::HashSet;
use std::collections::HashMap;
use std::cmp::Ordering;
use smallvec::SmallVec;
use std::ops::Index;
use std::ops::IndexMut;
use std::fmt::Debug;
use std::clone::Clone;
use std::borrow::{Borrow,BorrowMut};
use ndarray::Array1;
use crate::utils::{slow_sme,slow_ssme,slow_mad,slow_median,argsort,ArgSort,ArgSortII};
use crate::io::DispersionMode;

#[derive(Clone,Serialize,Deserialize,Debug)]
pub struct HashRV {
    segments: [Segment;4],
    segment_directory: HashMap<usize,usize>
}

#[derive(Clone,Serialize,Deserialize,Debug)]
pub struct Segment {
    arena: HashMap<i32,f64>,
    linkages: HashMap<i32,(i32,i32)>,
    sum: f64,
    squared_sum: f64
}

impl HashRV {

    pub fn with_capacity(capacity:usize) -> HashRV {
        HashRV {
            segments:
            [
                Segment::with_capacity(capacity),
                Segment::with_capacity(capacity),
                Segment::with_capacity(capacity),
                Segment::with_capacity(capacity),
            ],
            segment_directory: HashMap::with_capacity(capacity),
        }
    }

    pub fn push(&mut self, index:usize,value:f64) {
        self.segments[3].push_right(index,value);
        self.segment_directory.insert(index,3);
    }

    // pub fn shift_left(&mut self) {
    //     if let Some((index,value)) = self.segments[0].pop_right() {
    //         self.segments[1].push_left(index,value);
    //     }
    //     if let Some((index,value)) = self.segments[2].pop_right() {
    //         self.segments[3].push_left(index,value);
    //     }
    // }
    //
    // pub fn shift_right(&mut self) {
    //     if let Some((index,value)) = self.segments[0].pop_right() {
    //         self.segments[1].push_left(index,value);
    //     }
    //     if let Some((index,value)) = self.segments[2].pop_right() {
    //         self.segments[3].push_left(index,value);
    //     }
    // }

}

impl Segment {

    pub fn with_capacity(capacity:usize) -> Segment {
        let mut arena = HashMap::with_capacity(capacity);
        let mut linkages = HashMap::with_capacity(capacity + 2);
        let sum = 0.;
        let squared_sum = 0.;
        linkages.insert(-2,(-2,-1));
        linkages.insert(-1,(-2,-1));
        Segment {
            arena,
            linkages,
            sum,
            squared_sum,
        }
    }


    pub fn len(&self) -> usize {
        self.arena.len()
    }

    pub fn left(&self) -> Option<usize> {
        if self.len() > 0 {
            Some(self.linkages[&-2].1 as usize)
        }
        else {None}
    }

    pub fn right(&self) -> Option<usize> {
        if self.len() > 0 {
            Some(self.linkages[&-1].0 as usize)
        }
        else {None}
    }

    pub fn insert(&mut self,index_u:usize,value:f64,left:i32,right:i32) {
        let index = index_u as i32;
        self.arena.insert(index,value);

        self.linkages.insert(index,(left,right));
        self.linkages.get_mut(&left).expect("Bad link").1 = index;
        self.linkages.get_mut(&right).expect("Bad link").0 = index;

        self.sum += value;
        self.squared_sum += value.powi(2);

    }

    pub fn pop(&mut self, index_u: usize) -> Option<(usize,f64)> {
        let index = index_u as i32;
        let value = self.arena.remove(&index)?;

        let (left,right) = self.linkages.remove(&index).expect("Bad link");
        self.linkages.get_mut(&left).expect("Bad link").1 = right;
        self.linkages.get_mut(&right).expect("Bad link").0 = left;

        self.sum -= value;
        self.squared_sum -= value.powi(2);

        Some((index_u,value))
    }

    pub fn push_right(&mut self,index_u:usize,value:f64) {

        let index = index_u as i32;
        self.arena.insert(index,value);

        let current_right = self.linkages[&-1].0;
        let current_right_left = self.linkages[&current_right].0;

        self.linkages.insert(-1,(index,-1));
        self.linkages.insert(index,(current_right,-1));
        self.linkages.insert(current_right,(current_right_left,index));

        self.sum += value;
        self.squared_sum += value.powi(2);
    }

    pub fn push_left(&mut self,index_u:usize,value:f64) {

        let index = index_u as i32;
        self.arena.insert(index,value);

        let current_left = self.linkages[&-2].1;
        let current_left_right = self.linkages[&current_left].1;

        self.linkages.insert(-2,(-2,index));
        self.linkages.insert(index,(-2,current_left));
        self.linkages.insert(current_left,(index,current_left_right));

        self.sum += value;
        self.squared_sum += value.powi(2);
    }


    pub fn pop_left(&mut self) -> Option<(usize,f64)> {
        let current_left = self.linkages[&-2].1;
        if current_left < 0 {return None};
        let (_,current_left_right) = self.linkages.remove(&current_left)?;
        let current_left_right_right = self.linkages[&current_left_right].1;

        self.linkages.insert(-2,(-2,current_left_right));
        self.linkages.insert(current_left_right,(-2,current_left_right_right));

        Some((current_left as usize,self.arena.remove(&current_left)?))
    }

    pub fn pop_right(&mut self) -> Option<(usize,f64)> {
        let current_right = self.linkages[&-1].0;
        if current_right < 0 {return None};
        let (current_right_left,_) = self.linkages.remove(&current_right)?;
        let current_right_left_left = self.linkages[&current_right_left].1;

        self.linkages.insert(-1,(current_right_left,-1));
        self.linkages.insert(current_right_left,(current_right_left_left,-1));

        Some((current_right as usize,self.arena.remove(&current_right)?))
    }

    pub fn check_integrity(&self) {
        if self.arena.len() != self.linkages.len() + 2 {
            panic!("Arena and links desynced")
        };
        if self.linkages[&-2].0 != -2 || self.linkages[&-1].1 != -1 {
            panic!("Endcap link error")
        };

    }
}

pub struct RRVCrawler<'a> {
    vector: &'a HashRV,
    segment: usize,
    index: i32,
}

impl<'a> RRVCrawler<'a> {

    #[inline]
    fn new(input: &'a HashRV) -> RRVCrawler {
        let segment = 0;
        let first = -2;
        RRVCrawler{vector: input,segment: segment, index: first}
    }

    #[inline]
    fn start_at(input: &'a HashRV, first: usize) -> RRVCrawler {
        let segment = input.segment_directory[&first];
        let start_index = input.segments[segment].linkages[&(first as i32)].0;
        RRVCrawler{vector: input,segment: segment, index: start_index}
    }
}

impl<'a> Iterator for RRVCrawler<'a> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {

        if self.segment > 3 {
            return None
        }

        let (previous,next) = self.vector.segments[self.segment].linkages[&(self.index as i32)];

        if next == -1 {
            self.segment += 1;
            self.index = -2;
            return self.next()
        }

        self.index = next;

        return Some(next as usize)
    }
}

#[cfg(test)]
mod hash_rv_tests {

    use super::*;

    #[test]
    fn rank_vector_create_simple() {
        let mut hrv = HashRV::with_capacity(8);
        let values = vec![-3.,-2.,-1.,0.,5.,10.,15.,20.];
        for (i,v) in values.iter().enumerate() {
            hrv.push(i,*v);
        }

        println!("{:?}",hrv);

        let mut crawler = RRVCrawler::new(&hrv);

        while let Some(v) = crawler.next() {
            println!("{}",v);
        }

        panic!();
    }
}
