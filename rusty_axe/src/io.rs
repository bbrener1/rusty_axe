// #![feature(test)]



use std::io;
use std::f64;
use std::collections::HashMap;
use std::fs::File;
use std::io::prelude::*;
use std::fmt::Debug;
use crate::utils::{arr_from_vec2};
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


#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct Parameters {
    auto: bool,
    pub unsupervised: bool,
    // pub stdin: bool,
    pub input_count_array_file: String,
    pub output_count_array_file: String,
    pub input_feature_header_file: Option<String>,
    pub output_feature_header_file: Option<String>,
    pub sample_header_file: Option<String>,
    pub report_address: String,

    pub processor_limit: usize,
    pub parallel_trees: bool,
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


    pub norm_mode: NormMode,
    pub standardize: bool,
    pub dispersion_mode: DispersionMode,
    pub split_fraction_regularization: f64,

}

impl Parameters {

    pub fn empty() -> Parameters {
        let arg_struct = Parameters {
            auto: false,
            unsupervised: false,
            input_count_array_file: "".to_string(),
            output_count_array_file: "".to_string(),
            input_feature_header_file: None,
            output_feature_header_file: None,
            sample_header_file: None,
            report_address: "./".to_string(),


            processor_limit: 1,
            parallel_trees: true,
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

            norm_mode: NormMode::L2,
            standardize: false,
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

        let _ = args.next().unwrap();
        // eprintln!("Rust observes: {}",blank);

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
                },
                "-c" | "-counts" => {
                    let single_count_array_file = args.next().expect("Error parsing count location!");
                    arg_struct.unsupervised = true;
                    arg_struct.input_count_array_file = single_count_array_file.clone();
                    arg_struct.output_count_array_file = single_count_array_file;
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
                "-oc" | "-output_counts" => {
                    arg_struct.output_count_array_file = args.next().expect("Failed to retrieve output location");
                    // arg_struct.output_array = Some(read_matrix(&args.next().expect("Error parsing output count location!")));
                }
                "-dm" | "-dispersion_mode" => {
                    arg_struct.dispersion_mode = DispersionMode::read(&args.next().expect("Failed to read split mode"));
                },
                "-split_fraction_regularization" | "-sfr" => {
                    arg_struct.split_fraction_regularization = args.next().expect("Error processing SFR").parse::<f64>().expect("Error parsing SFR");
                }
                "-n" | "-norm" | "-norm_mode" => {
                    arg_struct.norm_mode = NormMode::read(&args.next().expect("Failed to read norm mode"));
                },
                "-std" | "-standardize" | "-standardized" => {
                    arg_struct.standardize = args.next().unwrap().parse::<bool>().expect("Error parsing std argument");
                    // arg_struct.standardize = true;
                }
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
                "-parallel_trees" => {
                    arg_struct.parallel_trees = args.next().expect("Argument error").parse::<bool>().unwrap();
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
                    arg_struct.input_feature_header_file = Some(header_file.clone());
                    arg_struct.output_feature_header_file = Some(header_file);
                },
                "-s" | "-samples" => {
                    arg_struct.sample_header_file = Some(args.next().expect("Error processing feature file"));
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
                        panic!("Unexpected argument:{}",arg);
                    }
                }

            }
        }

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



pub fn read_matrix(location:&str) -> Vec<Vec<f64>> {

    let count_array_file = File::open(location).expect("Count file error!");
    let mut count_array_lines = io::BufReader::new(&count_array_file).lines();

    let mut count_array: Vec<Vec<f64>> = Vec::new();

    for (i,line) in count_array_lines.by_ref().enumerate() {

        let mut gene_vector = Vec::new();
        let gene_line = line.expect("Readline error");
        for gene in gene_line.split_whitespace() {
            match gene.parse::<f64>() {
                Ok(exp_val) => {
                    if exp_val != f64::NAN {
                        gene_vector.push(exp_val);
                    }
                    else { panic!("Nan in input. Please sanitize matrix!") };
                },
                Err(msg) => {

                    println!("Couldn't parse a cell in the text file, Rust sez: {:?}",msg);
                    println!("Cell content: {:?}", gene);
                    panic!();
                }
            }
        }

        count_array.push(gene_vector);

        if i % 100 == 0 {
            print!("Ingesting {}\r", i);
        }


    };

    print!("Ingested {},{}\r", count_array.len(),count_array.get(0).unwrap_or(&vec![]).len());
    print!("                                          ");

    count_array

}

pub fn read_header(location: &str) -> Vec<String> {

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

    header_vector
}


#[cfg(test)]
pub mod primary_testing {

    use super::*;
    use crate::utils::*;

    static TEST_LOCATION: &str = "./rusty_axe/src/testing/";




    #[test]
    fn test_parameters_args() {
        let mut args_iter = vec![
            "blank",
            "-c",
            "testing/iris.tsv",
            "-p",
            "3",
            "-o",
            "./elsewhere/",
            "-ifh",
            "header_backup.txt"
        ].into_iter().map(|x| x.to_string());

        let args = Parameters::read(&mut args_iter);

        assert_eq!(args.input_count_array_file, "testing/iris.tsv".to_string());
        assert_eq!(args.input_feature_header_file.unwrap(), "header_backup.txt".to_string());
        assert_eq!(args.sample_header_file, None);
        assert_eq!(args.report_address, "./elsewhere/".to_string());

        assert_eq!(args.processor_limit, 3);

    }

    #[test]
    fn test_read_counts_trivial() {
        assert_eq!(
            read_matrix(&format!("{}/trivial.txt",TEST_LOCATION)),
            Vec::<Vec<f64>>::with_capacity(0)
        );
    }

    #[test]
    fn test_read_counts_simple() {
        assert_eq!(
            read_matrix(&format!("{}/simple.txt",TEST_LOCATION)),
            vec![vec![10.,5.,-1.,0.,-2.,10.,-3.,20.]]
        );
    }

    #[test]
    fn test_read_header_trivial() {
        assert_eq!(
            read_header(&format!("{}/trivial.txt",TEST_LOCATION)),
            Vec::<String>::with_capacity(0)
        )
    }

    #[test]
    fn test_read_header_simple() {
        assert_eq!(
            read_header(&format!("{}/iris.features",TEST_LOCATION)),
            vec!["petal_length","petal_width","sepal_length","sepal_width"]
        )
    }




}
