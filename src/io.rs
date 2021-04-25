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
use crate::utils::{matrix_flip,mtx_dim};


//::/ Author: Boris Brenerman
//::/ Created: 2017 Academic Year, Johns Hopkins University, Department of Biology, Taylor Lab

//::/ This is a forest-based regression/classification software package designed with single-cell RNAseq data in mind.
//::/
//::/ Currently implemented features are to generate Decision Trees that segment large 2-dimensional matrices, and prediction of samples based on these decision trees
//::/
//::/ Features to be implemented include interaction analysis, python-based node clustering and trajectory analysis using minimum spanning trees of clusters, feature correlation analysis, and finally subsampling-based gradient boosting for sequential tree generation.

//::/ The general structure of the program is as follows:
//::/
//::/ The outer-most class is the Random Forest
//::/


//::/ Random Forests:
//::/
//::/ Random Forest contains:
//::/     - The matrix to be analyzed
//::/     - Decision Trees
//::/
//::/     - Important methods:
//::/         - Method that generates decision trees and calls on them to grow branches
//::/         - Method that generates predicted values for a matrix of samples
//::/
//::/



//::/ Trees:
//::/
//::/ Trees contain:
//::/     - Root Node
//::/     - Feature Thread Pool Sender Channel
//::/     - Drop Mode
//::/
//::/ Each tree contains a subsampling of both rows and columns of the original matrix. The subsampled rows and columns are contained in a root node, which is the only node the tree has direct access to.
//::/

//::/ Feature Thread Pool:
//::/
//::/ Feature Thread Pool contains:
//::/     - Worker Threads
//::/     - Reciever Channel for jobs
//::/
//::/     - Important methods:
//::/         - A wrapper method to compute a set of medians and MADs for each job passed to the pool. Core method logic is in Rank Vector
//::/
//::/ Feature Thread Pools are containers of Worker threads. Each pool contains a multiple in, single out channel locked with a Mutex. Each Worker contained in the pool continuously requests jobs from the channel. If the Mutex is unlocked and has a job, a Worker thread receives it.
//::/
//::/     Jobs:
//::/         Jobs in the pool channel consist of a channel to pass back the solution to the underlying problem and a freshly spawned Rank Vector (see below). The job consists of calling a method on the RV that consumes it and produces the medians and Median Absolute Deviations (MAD) from the Median of the vector if a set of samples is removed from it in a given order. This allows us to determine what the Median Absolute Deviation from the Median would be given the split of that feature by some draw order. The draw orders given to each job are usually denoting that the underlying matrix was sorted by another feature.
//::/
//::/ Worker threads are simple anonymous threads kept in a vector in the pool, requesting jobs on loop from the channel.

//
// pub fn construct(mut args: Parameters) {
//
//     let input_array = args.input_array.take().unwrap();
//     let output_array = args.output_array.take().unwrap();
//
//     let mut loc_args = args.clone();
//     // loc_args.input_feature_names = vec![];
//     // loc_args.output_feature_names = vec![];
//     // loc_args.sample_names = vec![];
//     println!("Parsed parameters (Otherwise /function/ default)");
//     println!("{:?}", loc_args);
//
//     let mut arc_params = Arc::new(args);
//
//     // println!("Argumnets parsed: {:?}", arc_params);
//
//     println!("Reading data");
//
//
//     let report_address = &arc_params.report_address;
//
//     println!("##############################################################################################################");
//     println!("##############################################################################################################");
//     println!("##############################################################################################################");
//
//     let mut rnd_forest = Forest::initialize(&input_array,&output_array, arc_params.clone(), report_address);
//
//     rnd_forest.generate(arc_params.clone(),false).unwrap();
//
//
//
// }

pub fn predict(args: Parameters) {
    //
    // let mut loc_args = args.clone();
    // loc_args.counts = None;
    // println!("Parsed parameters (Otherwise /function/ default)");
    // println!("{:?}", loc_args);
    //
    // let arc_params = Arc::new(args.clone());
    // let tree_backups: TreeBackups;
    //
    // if args.backup_vec.as_ref().unwrap_or(&vec![]).len() > 0 {
    //     tree_backups = TreeBackups::Vector(args.backup_vec.unwrap());
    // }
    // else {
    //     tree_backups = TreeBackups::File(args.backups.expect("Backup trees not provided"));
    // }
    //
    // let counts = args.counts.as_ref().expect("Problem opening the matrix file (eg counts)");
    //
    //
    // let dimensions = (counts.get(0).unwrap_or(&vec![]).len(),counts.len());
    //
    // let features: Vec<String> = args.feature_names.unwrap_or((0..dimensions.1).map(|x| x.to_string()).collect());
    // let feature_map: HashMap<String,usize> = features.iter().cloned().enumerate().map(|x| (x.1,x.0)).collect();
    //
    // let forest = Forest::compact_reconstitute(tree_backups, Some(features), None ,None, "./").expect("Forest reconstitution failed");
    //
    // let predictions = forest.compact_predict(&counts,&feature_map,arc_params, &args.report_address);


}

pub fn combined(mut args:Parameters) {

    // let mut loc_args = args.clone();
    // loc_args.counts = None;
    // println!("Parsed parameters (Otherwise /function/ default)");
    // println!("{:?}", loc_args);
    //
    // let arc_params = Arc::new(args);
    //
    // let counts = arc_params.counts.as_ref().unwrap();
    //
    // let report_address = &arc_params.report_address;
    //
    // println!("##############################################################################################################");
    // println!("##############################################################################################################");
    // println!("##############################################################################################################");
    //
    // let mut rnd_forest = random_forest::Forest::initialize(counts, arc_params.clone(), &report_address);
    //
    // rnd_forest.generate(arc_params.clone(),true);
    //
    // let predictions = rnd_forest.compact_predict(&counts, &rnd_forest.feature_map().unwrap(), arc_params.clone(), &report_address);

}






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

    // pub input_array: Option<Vec<Vec<f64>>>,
    // pub output_array: Option<Vec<Vec<f64>>>,
    // pub input_feature_names: Vec<String>,
    // pub output_feature_names: Vec<String>,
    // pub sample_names: Vec<String>,

    pub processor_limit: usize,
    pub tree_limit: usize,
    pub leaf_size_cutoff: usize,
    pub depth_cutoff: usize,

    pub components: usize,

    pub feature_similarity: Option<Vec<Vec<f64>>>,
    pub sample_similarity: Option<Vec<Vec<f64>>>,

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

            feature_similarity: None,
            sample_similarity: None,

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
                // "-auto" | "-a"=> {
                //     arg_struct.auto = true;
                //     arg_struct.auto()
                // },
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
                "-feature_similarity" | "-f_sim" | "-feature_sim" => {
                    arg_struct.feature_similarity = Some(read_matrix(&args.next().expect("Error parsing feature_similarity count location!")));
                }
                "-sample_similarity" | "-s_sim" | "-sample_sim" => {
                    arg_struct.sample_similarity = Some(read_matrix(&args.next().expect("Error parsing sample_similarity count location!")));
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

    pub fn input_array(&self) -> Vec<Vec<f64>>{
        read_matrix(&self.input_count_array_file)
    }

    pub fn output_array(&self) -> Vec<Vec<f64>>{
        read_matrix(&self.output_count_array_file)
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
    SME,
    SSME,
    Entropy,
    Mixed,
}

impl DispersionMode {
    pub fn read(input: &str) -> DispersionMode {
        match input {
            "var" | "variance" => DispersionMode::Variance,
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

    matrix_flip(&count_array)

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
/////TESTING CODE///////

// let names: Vec<String> = (0..count_array[0].len()).map(|x| x.to_string()).collect();
// let samples: Vec<String> = (0..count_array.len()).map(|x| x.to_string()).collect();

// let medium_case = vec![vec![-1.,0.,-2.,10.,-3.,-4.,-20.,15.,20.,25.,100.]];
//
// let simple_case = vec![vec![0.,-1.,0.,-2.,10.,-3.,15.,20.]];
//

// let mut rng = rand::thread_rng();
// let input_features = rand::seq::sample_iter(&mut rng, names.clone(), 1000).expect("Couldn't generate input features");

// let mut tree = Tree::plant_tree(&matrix_flip(&count_array),&names.clone(),&samples.clone(),names.clone(),names.clone(), 20);
// let mut parallel_tree = Tree::plant_tree(&matrix_flip(&count_array),&names.clone(),&samples.clone(),input_features,names.clone(), 100);
//
// parallel_tree.grow_branches();



// axis_sum_test.push(vec![1.,2.,3.]);
// axis_sum_test.push(vec![4.,5.,6.]);
// axis_sum_test.push(vec![0.,1.,0.]);
// let temp: [f64;7] = [-3.,-2.,-1.,0.,10.,15.,20.];
// let temp2 = temp.into_iter().cloned().collect();
// let temp3 = vec![temp2];
// let temp4 = matrix_flip(&temp3);
//
//

// let mut thr_rng = rand::thread_rng();
// let rng = thr_rng.gen_iter::<f64>();
// let temp5: Vec<f64> = rng.take(49).collect();
// let temp6 = matrix_flip(&(vec![temp5.clone()]));

// axis_sum_test.push(vec![1.,2.,3.]);
// axis_sum_test.push(vec![4.,5.,6.]);
// axis_sum_test.push(vec![7.,8.,9.]);

// println!("Source floats: {:?}", matrix_flip(&counts));

// println!("{:?}", count_array);
//
// let mut raw = RawVector::raw_vector(&matrix_flip(&count_array)[0]);
//
// println!("{:?}",raw);
//
// println!("{:?}", raw.iter_full().cloned().collect::<Vec<(usize,usize,usize,f64,usize)>>());
//
// println!("Crawlers:");
//
// println!("{:?}", raw.crawl_right(raw.first).cloned().collect::<Vec<(usize,usize,usize,f64,usize)>>());
//
// println!("{:?}", raw.crawl_left(raw.first).cloned().collect::<Vec<(usize,usize,usize,f64,usize)>>());
//
// println!("Dropping zeroes:");
//
// raw.drop_zeroes();
//
// println!("Crawling dropped list:");
//
// println!("{:?}", raw.crawl_right(raw.first).cloned().collect::<Vec<(usize,usize,usize,f64,usize)>>());
//
// println!("Skipping dropped items:");
//
// println!("{:?}", raw.drop_skip().cloned().collect::<Vec<(usize,usize,usize,f64,usize)>>());
//
// println!("Printing non-zero values");
//
// println!("{:?}", raw.drop_skip().cloned().map(|x| x.3).collect::<Vec<f64>>());
//
// println!("Printing non-zero indecies");
//
// println!("{:?}", raw.drop_skip().cloned().map(|x| x.1).collect::<Vec<usize>>());
//
// println!("Printing noned-out drops");
// for i in raw.drop_none() {
//     println!("{:?}",i);
// }
//
// println!("Skipping drops");
// for i in raw.drop_skip() {
//     println!("{:?}",i);
// }
//
// println!("{:?}",raw.left_to_right().cloned().collect::<Vec<(usize,usize,usize,f64,usize)>>());
//
// println!("Finding dead center:");
//
// let dead_center = rank_vector::DeadCenter::center(&raw);
//
// println!("{:?}", dead_center);
//
// println!("{:?}", dead_center.median());

// println!("=================================================================");

// println!("Indecies: {:?}", matrix_flip(&count_array)[0]);
//
// println!("Testing Ranked Vector!");
//
// let degenerate_case = vec![0.;10];
//
// let mut ranked: RankVector<String,usize> = RankVector::new(&matrix_flip(&count_array)[0],String::from("test"), Vec::new());
//
// ranked.drop_zeroes();
//
// ranked.initialize();
//
// println!("Dropped values, ranked vector");
//
// println!("{:?}", ranked.vector.drop_skip().cloned().collect::<Vec<(usize,usize,usize,f64,usize)>>());
//
// println!("{:?}", ranked.clone());
//
// ranked.set_boundaries();
//
// println!("{:?}", ranked.clone());
//
// println!("{:?},{:?},{:?},{:?},{:?},{:?}", ranked.left_zone.size,ranked.left_zone.index_set.len(),ranked.median_zone.size,ranked.median_zone.index_set.len(),ranked.right_zone.size,ranked.right_zone.index_set.len());
//
// let ranked_clone = ranked.clone();
//
// {
//     let ordered_draw = OrderedDraw::new(&mut ranked);
//
//     println!("{:?}", ordered_draw.vector.vector.left_to_right().cloned().collect::<Vec<(usize,usize,usize,f64,usize)>>());
//
//     for sample in ordered_draw {
//         println!("Drawing: {:?}", sample);
//     }
// }
//
// println!("Dumping ranked vector:");
//
// let mut backup_debug_file = File::create("ranked_vec_debug.txt").unwrap();
// backup_debug_file.write_fmt(format_args!("{:?}", ranked));
//
// println!("Dumping ranked clone:");
//
// let mut backup_debug_file = File::create("ranked_vec_clone.txt").unwrap();
// backup_debug_file.write_fmt(format_args!("{:?}", ranked_clone));

// println!("Trying to make a rank table:");
//
// // let mut table = RankTable::new(simple_case,&names,&samples);
//
// println!("{},{}",count_array.len(),count_array[0].len());
//
// let mut table = RankTable::new(matrix_flip(&count_array),&names,&samples);
//
// println!("Finished making a rank table, trying to iterate:");
//
// let mut variance_table = Vec::new();
//
// for (j,i) in table.split(String::from("Test")).0.enumerate() {
//     // variance_table.push(vec![i[0].1/i[0].0,i[1].1/i[1].0,i[2].1/i[2].0,i[3].1/i[3].0]);
//     variance_table.push(vec![i[0].1/i[0].0]);
//     println!("{},{:?}",j,i)
// }
//
// println!("Variance table:");



// let minimal = variance_table.iter().map(|x| x.clone().sum()/(x.len() as f64)).enumerate().min_by(|a,b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Greater));
//
// println!("Minimal split is: {:?}", minimal);

// let mut node = Node::root(&vec![matrix_flip(&count_array)[1].clone()],&names,&samples,names[..1].iter().cloned().collect(),names[1..].iter().cloned().collect());

// let mut node = Node::root(&matrix_flip(&count_array),&names,&samples,names[..1].iter().cloned().collect(),names[1..].iter().cloned().collect());

// let mut node = Node::root(&matrix_flip(&count_array),&names.clone(),&samples.clone(),names.clone(),names.clone());

// let mut node = Node::root(&simple_case,&names,&samples,names[..1].iter().cloned().collect(),names[1..].iter().cloned().collect());


//
// println!("{:?}",node.rank_table.sort_by_feature(0));

// node.parallel_derive();
//
// for child in node.children.iter_mut() {
//     child.derive_children();
// }

// tree.test_splits();
// parallel_tree.test_parallel_splits();



// let mut forest = Forest::grow_forest(count_array, 1, 4, true);
// forest.test();

// println!("Inner axist test! {:?}", inner_axis_mean(&axis_sum_test));
// println!("Matrix flip test! {:?}", matrix_flip(&axis_sum_test));

// slow_description_test();
// slow_vs_fast();


// let mut ranked1: RankVector<String,usize> = RankVector::new(&matrix_flip(&count_array)[0],String::from("test"), Vec::new());
// let mut ranked2: RankVector<String,usize> = RankVector::new(&matrix_flip(&count_array)[1],String::from("test"), Vec::new());
// let mut ranked3: RankVector<String,usize> = RankVector::new(&matrix_flip(&count_array)[2],String::from("test"), Vec::new());
// let mut ranked4: RankVector<String,usize> = RankVector::new(&matrix_flip(&count_array)[3],String::from("test"), Vec::new());
//
// ranked1.drop_zeroes();
// ranked2.drop_zeroes();
// ranked3.drop_zeroes();
// ranked4.drop_zeroes();
//
//
// ranked1.initialize();
// ranked2.initialize();
// ranked3.initialize();
// ranked4.initialize();
//
// ranked1.set_boundaries();
// ranked2.set_boundaries();
// ranked3.set_boundaries();
// ranked4.set_boundaries();
//
// {
//     let ordered_draw = OrderedDraw::new(&mut ranked1);
//
//     println!("{:?}", ordered_draw.vector.vector.left_to_right().cloned().collect::<Vec<(usize,usize,usize,f64,usize)>>());
//
//     for sample in ordered_draw {
//         println!("Drawing: {:?}", sample);
//     }
// }
// fn inner_axis_sum(in_mat: &Vec<Vec<f64>>) -> Vec<f64> {
//
//     let mut s = vec![0f64;in_mat[0].len()];
//
//     for i in in_mat {
//         for j in i.iter().enumerate() {
//             s[j.0] += *j.1;
//         }
//     }
//     // println!("Inner axis being computed: {:?}", in_mat.iter().map(|x| x[0]).take(10).collect::<Vec<_>>());
//     // println!("{}", in_mat[0].len());
//     // println!("Inner axis sum: {}", s[0]);
//     s
// }
//
// fn inner_axis_mean(in_mat: &Vec<Vec<f64>>) -> Vec<f64> {
//
//     let mut s = vec![0f64;in_mat[0].len()];
//
//     for i in in_mat {
//         for j in i.iter().enumerate() {
//             s[j.0] += *j.1/(in_mat.len() as f64);
//         }
//     }
//
//     s
// }
//
// fn inner_axis_variance_sum(in_mat: &Vec<Vec<f64>>, in_means: Option<Vec<f64>>) -> Vec<f64> {
//
//     let m: Vec<f64>;
//
//     match in_means {
//         Option::Some(input) => m = input,
//         Option::None => m = inner_axis_mean(in_mat)
//     }
//
//     println!("Inner axis mean: {:?}", m);
//
//     let mut vs = vec![0f64;in_mat[0].len()];
//
//     for i in in_mat {
//         for j in i.iter().enumerate() {
//             // println!("Variance sum compute");
//             // println!("{}",*j.1);
//             // println!("{}", m[j.0]);
//             vs[j.0] += (*j.1 - m[j.0]).powi(2);
//             // println!("{}", vs[j.0]);
//         }
//     }
//     // println!("Inner_axis being computed: {:?}", in_mat.iter().map(|x| x[0]).take(10).collect::<Vec<_>>());
//     // println!("{}", in_mat.len());
//     // println!("Inner axis variance sum: {}", vs[0]);
//     vs
// }
//
// fn inner_axis_stats(in_mat: &Vec<Vec<f64>>) -> (Vec<f64>,Vec<f64>) {
//
//     let m = inner_axis_mean(in_mat);
//
//     let mut v = vec![0f64;in_mat[0].len()];
//
//     for i in in_mat {
//         for j in i.iter().enumerate() {
//             v[j.0] += (*j.1 - m[j.0]).powi(2)/(v.len() as f64);
//         }
//     }
//
//     (m,v)
// }
