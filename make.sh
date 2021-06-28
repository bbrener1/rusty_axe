#!/bin/bash

set -e

# First check for git:

git || echo "Git is a prerequisite for this tool, I don't want to put it in the wrong place for you. Please install yourself."

# Check to see if the rust configuration is weird:
if rustc && [ ! cargo ]
then {
  echo "I see rustc but not cargo. If you are an advanced user please clone and compile the code directly from https://github.com/bbrener1/rf_5."
  exit 1
}
fi

# Check for cargo

cargo || {
  # If cargo isn't present check for rustup
  rustup || {
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  }
  # Add cargo when rustup is present
  rustup component add cargo
}

# Setup the html directory for reports/consensus trees
mkdir html

# Build
cargo build --release

if [[ -f ./rf_5/target/release/rf_5]]
then
  echo "Installed successfully"
  exit 0
else
  echo "Problem building"
  exit 1
fi
