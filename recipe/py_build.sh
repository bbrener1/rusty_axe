#!/bin/bash
# This script needs the intermediate location of the library passed to it to compile
# the source and copy the binary to the appropriate location.

mkdir $1/rusty_axe/bin

cargo build --release

if [[ -f $1/target/release/rf_5 ]];
then
  cp $1/target/release/rf_5 $1/rusty_axe/bin/rf_5
  chmod +x $1/rusty_axe/bin/rf_5
  echo $(ls $1)
  echo $(ls -lh $1/rusty_axe/bin)
else
  exit 1
fi
