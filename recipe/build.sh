#!/bin/bash
# This is the conda build script. It calls out to the script that would be
# used by the setuptools install process and alerts it to the location
# of the library.

echo $(ls)
echo $(ls ./*)
echo $(pwd)

python -m build

echo $(ls)
echo $(ls ./*)

python -m install ./dist/rusty-axe-bbrener1-0.66.tar.gz

if [[ -f ./target/release/rf_5 ]];
then
  cp ./target/release/rf_5 ./rusty_axe/bin/rf_5
  chmod +x ./rusty_axe/bin/rf_5
else
  exit 1
fi
