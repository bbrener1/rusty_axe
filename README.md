# Rusty Axe

## Analyzing nested structure in data using unsupervised random forests.

This package is intended to interface with one or two numpy matrices of a large size (>100 samples, >10 features), and decomposes said matrices into random forest factors (RFFs) that describe different effects at different levels of nesting and non-linear dependency. It generates HTML reports that describe the underlying data, and can also generate other kinds of feedback. This package additionally can train on one dataset and compare that dataset to another. 

It is currently intended to be run on linux or osx, but can limp on windows. 

## Installation

Installation is currently by pip -install of the sdist file located in this repo under /dist. 

Alternatively, if you wish to clone this repo, python setup.py should also do the trick. 

A conda installer is forthcoming, as is a pypi upload. 
