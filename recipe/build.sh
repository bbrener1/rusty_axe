#!/bin/bash

cargo build --release

if [[ -f ./target/release/rf_5 ]];
then
  cp ./target/release/rf_5 .
else
  exit 1
fi
