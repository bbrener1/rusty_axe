// #![feature(test)]



use std::io;
use std::f64;
use std::collections::HashMap;
use std::fs::File;
use std::io::prelude::*;
use std::io::stdin;
use std::cmp::PartialOrd;
use std::cmp::Ordering;
use std::fmt::Debug;
use std::sync::Arc;
use rayon::prelude::*;
use crate::{Feature,Sample};
use crate::utils::{matrix_flip,mtx_dim,arr_from_vec2};
use ndarray::Array2;

//::/ Author: Boris Brenerman
//::/ Created: 2017 academic year, Johns Hopkins University, Department of Biology, Taylor Lab
//::/ Current version created 2021 academic year, Johns Hopkins University, Department of Biology, under supervision of Alexis Battle

//::/ This is a forest-based regression/classification software package designed with single-cell RNAseq data in mind.
//::/
//::/ Currently implemented features are to generate decision trees that segment large 2-dimensional matrices, and prediction of samples based on these decision trees
//::/


//
// }


#[derive(Debug,Clone)]
pub struct Parameters {
    auto: bool,
    pub command: Command,
    pub unsupervised: bool,
    // pub stdin: bool,
    pub input_count_array_file: String,
    pub output_count_array_file: String,
    pub input_feature_header_file: Option<String>,
    pub output_feature_header_file: Option<String>,
    pub sample_header_file: Option<String>,
    pub report_address: String,

    pub processor_limit: usize,
    pub tree_limit: usize,
    pub leaf_size_cutoff: usize,
    pub depth_cutoff: usize,

    pub components: usize,

    pub sample_subsample: usize,
    pub input_feature_subsample: usize,
    pub output_feature_subsample: usize,

    pub reduce_input: bool,
    pub reduce_output: bool,
    pub reduction: usize,


    pub averaging_mode: AveragingMode,
    pub norm_mode: NormMode,
    pub weighing_mode: WeighingMode,
    pub dispersion_mode: DispersionMode,
    pub split_fraction_regularization: f64,

}

impl Parameters {

    pub fn empty() -> Parameters {
        let arg_struct = Parameters {
            auto: false,
            command: Command::Combined,
            unsupervised: false,
            // stdin:false,
            input_count_array_file: "".to_string(),
            output_count_array_file: "".to_string(),
            input_feature_header_file: None,
            output_feature_header_file: None,
            sample_header_file: None,
            report_address: "./".to_string(),

            // input_array: None,
            // output_array: None,
            // input_feature_names: vec![],
            // output_feature_names: vec![],
            // sample_names: vec![],

            processor_limit: 1,
            tree_limit: 1,
            leaf_size_cutoff: 1,
            depth_cutoff: 1,

            components: 1,

            sample_subsample: 1,
            input_feature_subsample: 1,
            output_feature_subsample: 1,

            reduce_input: false,
            reduce_output: false,
            reduction: 1,

            averaging_mode: AveragingMode::Arithmetic,
            norm_mode: NormMode::L2,
            weighing_mode: WeighingMode::Flat,
            dispersion_mode: DispersionMode::SSME,
            split_fraction_regularization: 1.,
        };
        arg_struct
    }

    pub fn read<T: Iterator<Item = String>>(args: &mut T) -> Parameters {

        let mut arg_struct = Parameters::empty();

        let mut supress_warnings = false;
        let mut continuation_flag = false;
        let mut continuation_argument: String = "".to_string();

        let blank = args.next().unwrap();
        eprintln!("Rust observes: {}",blank);

        while let Some((i,arg)) = args.enumerate().next() {
            if arg.clone().chars().next().unwrap_or('_') == '-' {
                continuation_flag = false;

            }
            match &arg[..] {
                "-sw" | "-suppress_warnings" => {
                    if i!=1 {
                        println!("If the supress warnings flag is not given first it may not function correctly.");
                    }
                    supress_warnings = true;
                },
                "-auto" | "-a"=> {
                    arg_struct.auto = true;
                //     arg_struct.auto()
                },
                // "-stdin" => {
                //     arg_struct.stdin = true;
                    // let single_array = Some(read_standard_in());
                    // arg_struct.input_array = single_array.clone();
                    // arg_struct.output_array = single_array;
                // }
                "-c" | "-counts" => {
                    let single_count_array_file = args.next().expect("Error parsing count location!");
                    arg_struct.unsupervised = true;
                    arg_struct.input_count_array_file = single_count_array_file.clone();
                    arg_struct.output_count_array_file = single_count_array_file;
                    // let single_array = Some(read_matrix(&single_count_array_file));
                    // arg_struct.input_array = single_array.clone();
                    // arg_struct.output_array = single_array;
                },
                "-components" => {
                    arg_struct.components = args.next().expect("Error reading components").parse::<usize>().expect("-braid not a number");
                },
                "-unsupervised" => {
                    arg_struct.unsupervised = true;
                },
                "-reduce_input" | "-ri" => {
                    arg_struct.reduce_input = args.next().unwrap().parse::<bool>().expect("Error parsing reduction argument");
                },
                "-reduce_output" | "-ro" => {
                    arg_struct.reduce_output = args.next().unwrap().parse::<bool>().expect("Error parsing reduction argument");
                },

                "-ic" | "-input_counts" | "-input" => {
                    arg_struct.input_count_array_file = args.next().expect("Failed to retrieve input location");
                    // arg_struct.input_array = Some(read_matrix(&args.next().expect("Error parsing input count location!")));
                }
                "-oc" | "-output_counts" | "-output" => {
                    arg_struct.output_count_array_file = args.next().expect("Failed to retrieve input location");
                    // arg_struct.output_array = Some(read_matrix(&args.next().expect("Error parsing output count location!")));
                }
                "-am" | "-averaging_mode" | "-averaging" => {
                    arg_struct.averaging_mode = AveragingMode::read(&args.next().expect("Error reading averaging mode"));
                }
                "-wm" | "-w" | "-weighing_mode" => {
                    arg_struct.weighing_mode = WeighingMode::read(&args.next().expect("Failed to read weighing mode!"));
                },
                "-dm" | "-dispersion_mode" => {
                    arg_struct.dispersion_mode = DispersionMode::read(&args.next().expect("Failed to read split mode"));
                },
                "-split_fraction_regularization" | "-sfr" => {
                    arg_struct.split_fraction_regularization = args.next().expect("Error processing SFR").parse::<f64>().expect("Error parsing SFR");
                }
                "-n" | "-norm" | "-norm_mode" => {
                    arg_struct.norm_mode = NormMode::read(&args.next().expect("Failed to read norm mode"));
                },
                "-t" | "-trees" => {
                    arg_struct.tree_limit = args.next().expect("Error processing tree count").parse::<usize>().expect("Error parsing tree count");
                },
                "-tg" | "-tree_glob" => {
                    continuation_flag = true;
                    continuation_argument = arg.clone();
                },
                "-p" | "-processors" | "-threads" => {
                    arg_struct.processor_limit = args.next().expect("Error processing processor limit").parse::<usize>().expect("Error parsing processor limit");
                    rayon::ThreadPoolBuilder::new().num_threads(arg_struct.processor_limit).build_global().unwrap();
                    std::env::set_var("OMP_NUM_THREADS",format!("{}",arg_struct.processor_limit));
                },
                "-o" | "-output" => {
                    arg_struct.report_address = args.next().expect("Error processing output destination")
                },
                "-ifh" | "-ih" | "-input_feature_header" => {
                    arg_struct.input_feature_header_file = Some(args.next().expect("Error processing feature file"));
                    // arg_struct.input_feature_names = read_header(arg_struct.input_feature_header_file.as_ref().unwrap());
                },
                "-ofh" | "-oh" | "-output_feature_header" => {
                    arg_struct.output_feature_header_file = Some(args.next().expect("Error processing feature file"));
                    // arg_struct.output_feature_names = read_header(arg_struct.output_feature_header_file.as_ref().unwrap());
                },
                "-h" | "-header" => {
                    let header_file = args.next().expect("Error processing feature file");
                    let header = read_header(&header_file);

                    arg_struct.input_feature_header_file = Some(header_file.clone());
                    arg_struct.output_feature_header_file = Some(header_file);

                    // arg_struct.input_feature_names = header.clone();
                    // arg_struct.output_feature_names = header;
                },
                "-s" | "-samples" => {
                    arg_struct.sample_header_file = Some(args.next().expect("Error processing feature file"));
                    // arg_struct.sample_names = read_sample_names(arg_struct.sample_header_file.as_ref().unwrap());
                }
                "-l" | "-leaves" => {
                    arg_struct.leaf_size_cutoff = args.next().expect("Error processing leaf limit").parse::<usize>().expect("Error parsing leaf limit");
                },
                "-depth" => {
                    arg_struct.depth_cutoff = args.next().expect("Error processing depth").parse::<usize>().expect("Error parsing depth");
                }
                "-if" | "-ifs" | "-in_features" | "-in_feature_subsample" | "-input_feature_subsample" => {
                    arg_struct.input_feature_subsample = args.next().expect("Error processing in feature arg").parse::<usize>().expect("Error in feature  arg");
                },
                "-of" | "-ofs" | "-out_features" | "-out_feature_subsample" | "-output_feature_subsample" => {
                    arg_struct.output_feature_subsample = args.next().expect("Error processing out feature arg").parse::<usize>().expect("Error out feature arg");
                },
                "-fs" | "-feature_sub" | "-feature_subsample" | "-feature_subsamples" => {
                    let fs = args.next().expect("Error processing feature subsample arg").parse::<usize>().expect("Error feature subsample arg");
                    arg_struct.input_feature_subsample = fs;
                    arg_struct.output_feature_subsample = fs;
                },
                "-ss" | "-sample_sub" | "-sample_subsample" | "-sample_subsamples" => {
                    arg_struct.sample_subsample = args.next().expect("Error processing sample subsample arg").parse::<usize>().expect("Error sample subsample arg");
                },
                "-reduction" | "-r"  => {
                    arg_struct.reduction = args.next().expect("Error reading number of components").parse::<usize>().expect("-not a number");
                },


                &_ => {
                    if continuation_flag {
                        // This block allows parsing multiple arguments to an option, but currently I'm not using any such options
                        // Retained for future use.

                        // match &continuation_argument[..] {
                        //     &_ => {
                        //         panic!("Continuation flag set but invalid continuation argument, debug prediction arg parse!");
                        //     }
                        // }
                    }
                    else if !supress_warnings {
                        // eprintln!("Warning, detected unexpected argument:{}. Ignoring, press enter to continue, or CTRL-C to stop. Were you trying to input multiple arguments? Only some options take multiple arguments. Watch out for globs(*, also known as wild cards), these count as multiple arguments!",arg);
                        // stdin().read_line(&mut String::new());
                        panic!(format!("Unexpected argument:{}",arg));
                    }
                }

            }
        }

        // assert!(arg_struct.input_array.as_ref().expect("Please specify input file").get(0).expect("Empty input!").len() == arg_struct.output_array.as_ref().expect("Please specify output file").get(0).expect("Empty output!").len(), "Unequal dimensions in input and output!");

        // if arg_struct.input_feature_header_file.is_none() {
        //     let dimensions = mtx_dim(arg_struct.input_array.as_ref().unwrap());
        //     arg_struct.input_feature_names = (0..dimensions.0).map(|x| x.to_string()).collect()
        // }
        // if arg_struct.output_feature_header_file.is_none() {
        //     let dimensions = mtx_dim(arg_struct.output_array.as_ref().unwrap());
        //     arg_struct.output_feature_names = (0..dimensions.0).map(|x| x.to_string()).collect()
        // }
        // if arg_struct.sample_header_file.is_none() {
        //     let dimensions = mtx_dim(arg_struct.input_array.as_ref().unwrap());
        //     arg_struct.sample_names = (0..dimensions.1).map(|x| x.to_string()).collect()
        // }

        // eprintln!("INPUT ARRAY FEATURES:{}", arg_struct.input_array.as_ref().unwrap().len());
        // eprintln!("OUTPUT ARRAY FEATURES:{}", arg_struct.output_array.as_ref().unwrap().len());
        // eprintln!("SAMPLE HEADER:{}", arg_struct.sample_names.len());

        arg_struct

    }

    pub fn input_array(&self) -> Array2<f64>{
        let vv = read_matrix(&self.input_count_array_file);
        arr_from_vec2(vv)
    }

    pub fn output_array(&self) -> Array2<f64>{
        let vv = read_matrix(&self.output_count_array_file);
        arr_from_vec2(vv)
    }

    pub fn input_feature_names(&self) -> Option<Vec<String>> {
        Some(read_header(self.input_feature_header_file.as_ref()?))
    }

    pub fn output_feature_names(&self) -> Option<Vec<String>> {
        Some(read_header(self.output_feature_header_file.as_ref()?))
    }

    pub fn sample_names(&self) -> Option<Vec<String>> {
        Some(read_header(self.sample_header_file.as_ref()?))
    }

}

// Various modes that are included in Parameters, serving as control elements for program internals. Each mode can parse strings that represent alternative options for that mode. Enums were chosen because they compile down to extremely small memory footprint.


#[derive(Clone,Copy,Debug)]
pub enum BoostMode {
    Additive,
    Subsampling,
}

impl BoostMode {
    pub fn read(input: &str) -> BoostMode {
        match input {
            "additive" | "a" | "add" => BoostMode::Additive,
            "s" | "subsampling" | "subsample" => BoostMode::Subsampling,
            _ => {
                eprintln!("Not a valid boost mode, choose sub or add (defaulting to add)");
                BoostMode::Additive
            }
        }
    }

}


#[derive(Debug,Clone)]
pub enum Command {
    Combined,
    Construct,
    Predict,
    Analyze
    // Gradient,
}

impl Command {

    pub fn parse(command: &str) -> Command {

        match &command[..] {
            "construct" | "generate" => {
                Command::Construct
            },
            "predict" => {
                Command::Predict
            },
            "construct_predict" | "conpred" | "combined" => {
                Command::Combined
            }
            "analyze" => {
                Command::Analyze
            }
            // "gradient" => {
            //     Command::Gradient
            // }
            _ =>{
                println!("Not a valid top-level command, please choose from \"construct\",\"predict\",\"analyze\", or \"construct_predict\". Exiting");
                panic!()
            }
        }
    }


}
//
// pub fn interpret(literal:&str, arg_iter:&mut std::env::Args) {
//
//     let command = Command::parse(literal);
//
//     let mut parameters = Parameters::read(arg_iter);
//
//     parameters.command = command;
//
//     match parameters.command {
//         Command::Construct => construct(parameters),
//         Command::Predict => predict(parameters),
//         Command::Combined => combined(parameters),
//         Command::Analyze => unimplemented!(),
//     }
//
// }

impl PredictionMode {
    pub fn read(input:&str) -> PredictionMode {
        match input {
            "branch" | "branching" | "b" => PredictionMode::Branch,
            "truncate" | "truncating" | "t" => PredictionMode::Truncate,
            "abort" | "ab" => PredictionMode::Abort,
            "auto" | "a" => PredictionMode::Auto,
            _ => panic!("Not a valid prediction mode, choose branch, truncate, or abort.")
        }
    }
}

#[derive(Serialize,Deserialize,Debug,Clone,Copy)]
pub enum PredictionMode {
    Branch,
    Truncate,
    Abort,
    Auto
}

#[derive(Serialize,Deserialize,Debug,Clone,Copy)]
pub enum AveragingMode {
    Arithmetic,
    Stacking
}

impl AveragingMode {
    pub fn read(input:&str) -> AveragingMode {
        match input {
            "a" | "arithmetic" | "average" => AveragingMode::Arithmetic,
            "s" | "stacking" => AveragingMode::Stacking,
            _ => panic!("Not a valid averaging mode, choose arithmetic or stacking.")
        }
    }
}

#[derive(Serialize,Deserialize,Debug,Clone,Copy)]
pub enum WeighingMode {
    AbsoluteGain,
    AbsGainSquared,
    AbsoluteDispersion,
    AbsDispSquared,
    Flat,
}

impl WeighingMode {
    pub fn read(input:&str) -> WeighingMode {
        match input {
            "gain" | "absolute_gain" | "g" => WeighingMode::AbsoluteGain,
            "gain_squared" | "gs" => WeighingMode::AbsGainSquared,
            "dispersion" | "d" => WeighingMode::AbsoluteDispersion,
            "dispersion_squared" | "ds" => WeighingMode::AbsDispSquared,
            "flat" | "f" => WeighingMode::Flat,
            _ => panic!("Not a valid weighing mode, please pick from gain, gain_squared, dispersion, dispersion_squared")
        }
    }
}

#[derive(Serialize,Deserialize,Debug,Clone,Copy)]
pub enum DispersionMode {
    MAD,
    Variance,
    SSE,
    SME,
    SSME,
    Entropy,
    Mixed,
}

impl DispersionMode {
    pub fn read(input: &str) -> DispersionMode {
        match input {
            "var" | "variance" => DispersionMode::Variance,
            "sse" => DispersionMode::SSE,
            "mad"  => DispersionMode::MAD,
            "mix" | "mixed" => DispersionMode::Mixed,
            "ssme" => DispersionMode::SSME,
            "sme" => DispersionMode::SME,
            "entropy" => DispersionMode::Entropy,
            _ => panic!("Not a valid dispersion mode, choose var, mad, or mixed")

        }
    }
}

#[derive(Serialize,Deserialize,Debug,Clone,Copy)]
pub enum NormMode {
    L1,
    L2,
}

impl NormMode {
    pub fn read(input: &str) -> NormMode {
        match input {
            "1" | "L1" | "l1" => NormMode::L1,
            "2" | "L2" | "l2" => NormMode::L2,
            _ => panic!("Not a valid norm, choose l1 or l2")
        }
    }
}

impl DropMode {
    pub fn read(input: &str) -> DropMode {
        match input {
            "zeros" | "zero" | "z" => DropMode::Zeros,
            "nans" | "nan" | "NaN" => DropMode::NaNs,
            "none" | "no" => DropMode::No,
            _ => panic!("Not a valid drop mode, choose zero, nan, or none")
        }
    }

    pub fn cmp(&self) -> f64 {
        match self {
            &DropMode::Zeros => 0.,
            &DropMode::NaNs => f64::NAN,
            &DropMode::No => f64::INFINITY,
        }
    }

    pub fn bool(&self) -> bool {
        match self {
            &DropMode::Zeros => true,
            &DropMode::NaNs => true,
            &DropMode::No => false,
        }
    }
}

#[derive(Debug,Clone,Copy,Serialize,Deserialize,PartialEq,Eq)]
pub enum DropMode {
    Zeros,
    NaNs,
    No,
}


pub fn read_matrix(location:&str) -> Vec<Vec<f64>> {

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

    count_array

}

pub fn read_standard_in() -> Vec<Vec<f64>> {

    let stdin = io::stdin();
    let count_array_pipe_guard = stdin.lock();

    let mut count_array: Vec<Vec<f64>> = Vec::new();
    // let mut samples = 0;

    for (_i,line) in count_array_pipe_guard.lines().enumerate() {

        // samples += 1;
        let mut gene_vector = Vec::new();

        for (_j,gene) in line.as_ref().expect("readline error").split_whitespace().enumerate() {

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

    };

    // eprintln!("Counts read:");
    // eprintln!("{:?}", counts);

    matrix_flip(&count_array)
}

pub fn read_header(location: &str) -> Vec<String> {

    println!("Reading header: {}", location);

    let mut header_map = HashMap::new();

    let header_file = File::open(location).expect("Header file error!");
    let mut header_file_iterator = io::BufReader::new(&header_file).lines();

    for (i,line) in header_file_iterator.by_ref().enumerate() {
        let feature = line.unwrap_or("error".to_string());
        let mut renamed = feature.clone();
        let mut j = 1;
        while header_map.contains_key(&renamed) {
            renamed = [feature.clone(),j.to_string()].join("");
            eprintln!("WARNING: Two individual features were named the same thing: {}",feature);
            j += 1;
        }
        header_map.insert(renamed,i);
    };

    let mut header_inter: Vec<(String,usize)> = header_map.iter().map(|x| (x.0.clone().clone(),x.1.clone())).collect();
    header_inter.sort_unstable_by_key(|x| x.1);
    let header_vector: Vec<String> = header_inter.into_iter().map(|x| x.0).collect();

    println!("Read {} lines", header_vector.len());

    header_vector
}

pub fn read_sample_names(location: &str) -> Vec<String> {

    let mut header_vector = Vec::new();

    let sample_name_file = File::open(location).expect("Sample name file error!");
    let mut sample_name_lines = io::BufReader::new(&sample_name_file).lines();

    for line in sample_name_lines.by_ref() {
        header_vector.push(line.expect("Error reading header line!").trim().to_string())
    }

    header_vector
}



mod manual_testing {

    use super::*;

    pub fn test_command_predict_full() {
        let mut args = vec!["predict", "-m","branching","-b","tree.txt","-tg","tree.0","tree.1","tree.2","-c","counts.txt","-p","3","-o","./elsewhere/","-f","header_backup.txt"].into_iter().map(|x| x.to_string());

        let command = Command::parse(&args.next().unwrap());

        println!("{:?}",command);

        // panic!();

    }

}

#[cfg(test)]
pub mod primary_testing {

    use super::*;
    use crate::utils::*;

    #[test]
    fn test_command_trivial() {

        match Command::parse("construct") {
            Command::Construct => {},
            _ => panic!("Failed prediction parse")
        };

        match Command::parse("predict") {
            Command::Predict => {},
            _ => panic!("Failed prediction parse")
        };

        match Command::parse("combined") {
            Command::Combined => {},
            _ => panic!("Failed prediction parse")
        };

    }

    #[test]
    #[should_panic]
    fn test_command_wrong() {
        Command::parse("abc");
    }

    #[test]
    fn test_matrix_flip() {
        let mtx1 = vec![
            vec![0,1,2],
            vec![3,4,5],
            vec![6,7,8]
        ];

        let mtx2 = vec![
            vec![0,3,6],
            vec![1,4,7],
            vec![2,5,8]
        ];

        assert_eq!(matrix_flip(&mtx1),mtx2);

    }

    #[test]
    fn test_pearsonr() {
        let vec1 = vec![1.,2.,3.,4.,5.];
        let vec2 = vec![2.,3.,4.,5.,6.];

        println!("{:?}",pearsonr(&vec1,&vec2));

        if (pearsonr(&vec1,&vec2)-1.) > 0.00001 {
            panic!("Correlation error")
        }
    }

    #[test]
    fn test_parameters_args() {
        let mut args_iter = vec![ "-c","testing/iris.tsv","-p","3","-o","./elsewhere/","-ifh","header_backup.txt"].into_iter().map(|x| x.to_string());

        let args = Parameters::read(&mut args_iter);

        assert_eq!(args.input_count_array_file, "testing/iris.tsv".to_string());
        assert_eq!(args.input_feature_header_file.unwrap(), "header_backup.txt".to_string());
        // assert_eq!(args.output_feature_header_file.unwrap(), "header_backup.txt".to_string());
        assert_eq!(args.sample_header_file, None);
        assert_eq!(args.report_address, "./elsewhere/".to_string());

        assert_eq!(args.processor_limit, 3);

    }


    #[test]
    fn test_read_counts_trivial() {
        assert_eq!(read_matrix("./testing/trivial.txt"),Vec::<Vec<f64>>::with_capacity(0))
    }

    #[test]
    fn test_read_counts_simple() {
        assert_eq!(read_matrix("./testing/simple.txt"), vec![vec![10.,5.,-1.,0.,-2.,10.,-3.,20.]])
    }

    #[test]
    fn test_read_header_trivial() {
        assert_eq!(read_header("./testing/trivial.txt"),Vec::<String>::with_capacity(0))
    }

    #[test]
    fn test_read_header_simple() {
        assert_eq!(read_header("./testing/iris.features"),vec!["petal_length","petal_width","sepal_length","sepal_width"])
    }




}
