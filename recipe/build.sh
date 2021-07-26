#!/bin/bash
# This is the conda build script. It calls out to the script that would be
# used by the setuptools install process and alerts it to the location
# of the library. 

py_build.sh $PREFIX
