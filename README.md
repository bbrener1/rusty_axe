# Rusty Axe

## Analyzing nested structure in data using unsupervised random forests.

This package is intended to interface with one or two numpy matrices of a large size (>100 samples, >10 features), and decomposes said matrices into random forest factors (RFFs) that describe different effects at different levels of nesting and non-linear dependency. It generates HTML reports that describe the underlying data, and can also generate other kinds of feedback. This package additionally can train on one dataset and compare that dataset to another. 

For a more complete description of available functions please see the tutorial under ./tutorial

This package is currently intended to be run on linux or osx. This package may funciton on windows but no guarantees are made. 

# Publlication

A more complete description of this approach to understanding data is pending publication. A bioarxiv preprint will be available shortly. 

## Installation

Install by invoking 

`pip install rusty_axe_bbrener1`

## Tutorial

A tutorial is available within this repo in the form of an ipython notebook. 

Please consult this tutorial or any of the notebooks used to generate the figures of the accompanying paper. 

## Building From Source 

In order to build this package from source you will need the rust compiler. It is easily obtained

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


