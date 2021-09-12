# Rusty Axe

## Analyzing nested structure in data using unsupervised random forests.

This package is intended to interface with one or two numpy matrices of a large size (>100 samples, >10 features), and decomposes said matrices into random forest factors (RFFs) that describe different effects at different levels of nesting and non-linear dependency. It generates HTML reports that describe the underlying data, and can also generate other kinds of feedback. This package additionally can train on one dataset and compare that dataset to another. 

For a more complete description of available functions please see the tutorial under ./tutorial

This package is currently intended to be run on linux or osx. This package may funciton on windows but no guarantees are made. 

## Installation

# COMPATIBILITY WARNING
Statsmodels, which is unfortunately necessary to run several core components of this package, has bad support for OSX Python 3.9. Therefore if you are on OSX, you will either need an existing instal of statsmodels, or python<=3.8 

### Obtaining Rust

Please note, before instlling this package you should obtain the rust compiler. 

A rust compiler can be obtained and silently installed by executing 

`curl https://sh.rustup.rs -sSf | sh -s -- -y`

If you wish to alter any aspect of the rust compiler defaults, you can execute 

`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`

or simply check the current recommendations at https://www.rust-lang.org/tools/install

## Installation

Install by invoking 

`pip install rusty_axe_bbrener1`
