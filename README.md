# Rusty Axe

## Analyzing nested structure in data using unsupervised random forests.

This package is intended to interface with one or two numpy matrices of a large size (>100 samples, >10 features), and decomposes said matrices into random forest factors (RFFs) that describe different effects at different levels of nesting and non-linear dependency. It generates HTML reports that describe the underlying data, and can also generate other kinds of feedback. This package additionally can train on one dataset and compare that dataset to another. 

This package is currently intended to be run on linux or osx. This package may funciton on windows but no guarantees are made. 

This tool is very much a work in progress, please send feedback, positive or negative, to boris.brenerman@gmail.com or by opening an issue here! I want to hear how to make this tool more useable and intuitive, and also which parts are helpful.

## Publlication

A more complete description of this approach to understanding data is available in the form of a publication: https://www.biorxiv.org/content/10.1101/2021.09.13.460168v1

## Warning, Please Install By Cloning, Pip Installation Is Out Of Date
// Install by invoking 
// `pip install rusty_axe_bbrener1`

## Tutorial

A tutorial is available within this repo in the form of an ipython notebook. 

Please consult this tutorial or any of the notebooks used to generate the figures of the accompanying paper. 

## Building From Source 

Optionally, you may wish to build this package from source (although installing via pypi is probably preferred) In order to build this package from source you will need the rust compiler. It is easily obtained

### Obtaining Rust

A rust compiler can be obtained and silently installed by executing 

`curl https://sh.rustup.rs -sSf | sh -s -- -y`

If you wish to alter any aspect of the rust compiler defaults, you can execute 

`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`

or simply check the current recommendations at https://www.rust-lang.org/tools/install

### Building the package and inserting it into the python path

After you've obtained the rust compiler I recommend you build using 

`python -m build` 
in a cloned repo and then install via 

`pip install dist/<tarball>`

