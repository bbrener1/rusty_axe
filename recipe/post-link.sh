#!/bin/bash

# Fetch the actual repo
git clone https://github.com/bbrener1/rf_5
cd ./rf_5

# Setup the html directory for reports/consensus trees
mkdir -p $PREFIX/rf_5/html
mkdir -p $PREFIX/rf_5/work

# Build
cargo build --release 2>/dev/null
