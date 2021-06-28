#!/bin/bash

set -e

# First check for git:

git || echo "Git is a prerequisite for this tool, I don't want to put it in the wrong place for you. Please install yourself."

# Check to see if the rust configuration is weird:
rustc && !(cargo) {
  echo "I see rustc but not cargo. If you are an advanced user please clone and compile the code directly."
  exit 1
}

# Check for cargo

cargo || {
  # If cargo isn't present check for rustup
  rustup || {
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  }
  # Add cargo when rustup is present
  rustup component add cargo
}

# Cargo now exists, clone the repo
git clone https://github.com/bbrener1/rf_5

mkdir ./rf_5/html

cargo build --release
