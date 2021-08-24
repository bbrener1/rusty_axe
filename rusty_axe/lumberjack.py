#!/usr/bin/env python

import numpy as np
import glob
import sys
import os
import tempfile as tmp
from pathlib import Path
import io
import argparse
from time import sleep
import subprocess as sp
import rusty_axe.tree_reader as tr


bin_path = os.path.join("bin", "rf_5")
RUST_PATH = str((Path(__file__).parent /
                 bin_path).resolve())


def main(location, input, output=None, ifh=None, ofh=None, **kwargs):
    if output is None:
        output = input
    print("Running main")
    print("Trying to load")
    print(input)
    print(output)
    input_counts = np.loadtxt(input)
    output_counts = np.loadtxt(output)
    if ifh is not None:
        ifh = np.loadtxt(ifh, dtype=str)
    if ofh is not None:
        ofh = np.loadtxt(ofh, dtype=str)
    print("Loaded counts")
    print(input)
    fit_return = save_trees(
        location, input_counts, output_counts=output_counts, ifh=ifh, ofh=ofh, **kwargs)
    print(fit_return)


def save_trees(location, input_counts, output_counts=None, ifh=None, ofh=None, header=None, lrg_mem=None, **kwargs):

    if output_counts is None:
        output_counts = input_counts

    if header is not None:
        ifh = header
        ofh = header

    np.savetxt(location + "input.counts", input_counts)
    np.savetxt(location + "output.counts", output_counts)

    if ifh is None:
        np.savetxt(location + "tmp.ifh",
                   np.arange(input_counts.shape[1], dtype=int), fmt='%u')
    else:
        np.savetxt(location + "tmp.ifh", ifh, fmt="%s")

    if ofh is None:
        np.savetxt(location + "tmp.ofh",
                   np.arange(output_counts.shape[1], dtype=int), fmt='%u')
    else:
        np.savetxt(location + "tmp.ofh", ofh, fmt="%s")

    print("Generating trees")

    return inner_fit(input_counts, output_counts, location, ifh=(location + "tmp.ifh"), ofh=(location + "tmp.ofh"), lrg_mem=lrg_mem, **kwargs)


def load(location):
    # Alias to the tree reader load
    return tr.Forest.load(location)


def fit(input_counts, cache=True, output_counts=None, ifh=None, ofh=None, header=None, backtrace=False, lrg_mem=None, location=None, **kwargs):

    if output_counts is None:
        output_counts = input_counts

    tmp_dir = None
    if location is None:

        print("Input:" + str(input_counts.shape))
        print("Output:" + str(output_counts.shape))

        tmp_dir = tmp.TemporaryDirectory()
        location = tmp_dir.name + "/"

    arguments = save_trees(tmp_dir.name + "/", input_counts=input_counts, output_counts=output_counts,
                           ifh=ifh, ofh=ofh, header=header, lrg_mem=lrg_mem, **kwargs)

    forest = tr.Forest.load_from_rust(location, prefix="tmp", ifh="tmp.ifh", ofh="tmp.ofh",
                                      clusters="tmp.clusters", input="input.counts", output="output.counts")

    forest.set_cache(cache)

    forest.arguments = arguments

    if tmp_dir is not None:
        tmp_dir.cleanup()

    return forest


def inner_fit(input_counts, output_counts, location, backtrace=False, lrg_mem=None, **kwargs):

    print("Running " + RUST_PATH)

    arg_list = []

    arg_list.extend([RUST_PATH, "-ic", location + "input.counts",
                     "-oc", location + "output.counts", "-o", location + "tmp", "-auto"])

    for arg in kwargs.keys():
        arg_list.append("-" + str(arg))
        arg_list.append(str(kwargs[arg]))

    if lrg_mem is not None:
        arg_list.append("-lrg_mem")

    print("Command: " + " ".join(arg_list))

    with sp.Popen(arg_list, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE, universal_newlines=True) as cp:
        tree_count = 0
        while True:
            rc = cp.poll()
            if rc is not None:
                print(cp.stdout.read(), end='')
                print(cp.stderr.read(), end='')
                break
            output = cp.stdout.readline()
            if output[:6] == "Ingest":
                print(output.rstrip(), end="\r")
            elif output[:9] == "Computing":
                tree_count += 1
                print(f"Computing tree {tree_count}", end='\r')
            else:
                print(output)

    return arg_list


if __name__ == "__main__":
    kwargs = {x.split("=")[0]: x.split("=")[1] for x in sys.argv[3:]}
    main(sys.argv[1], sys.argv[2], **kwargs)
