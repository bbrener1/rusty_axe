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
# import asyncio as aio
# import multiprocessing as mp
import subprocess as sp
# path_to_rust = (Path(__file__).parent / "target/release/lumberjack_1").resolve()
# path_to_tree_reader = Path(__file__).resolve()
# sys.path.append(path_to_tree_reader)
# import tree_reader_sc as tr
import rusty_axe.tree_reader as tr

import numpy as np
# import matplotlib.pyplot as plt

RUST_PATH = str((Path(__file__).parent /
                    "../../target/release/rf_5").resolve())

def main(location, input, output=None, ifh=None, ofh=None, **kwargs):
    # print("Tree reader?")
    # print(path_to_tree_reader)
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


def fit(input_counts, output_counts=None, ifh=None, ofh=None, header=None, backtrace=False, lrg_mem=None, location=None, **kwargs):

    if output_counts is None:
        output_counts = input_counts

    tmp_dir = None
    if location is None:

        print("Setting context")

        print("Input:" + str(input_counts.shape))
        print("Output:" + str(output_counts.shape))

        tmp_dir = tmp.TemporaryDirectory()
        location = tmp_dir.name + "/"

    arguments = save_trees(tmp_dir.name + "/", input_counts=input_counts, output_counts=output_counts,
                           ifh=ifh, ofh=ofh, header=header, lrg_mem=lrg_mem, **kwargs)

    print("CHECK TRUTH")
    print(tmp_dir.name)
    print(os.listdir(tmp_dir.name))

    print("Generating trees")

    # inner_fit(input_counts,output_counts,location,ifh=location + 'tmp.ifh',ofh=location + 'tmp.ofh',backtrace=backtrace,**kwargs)
    # ihmm_fit(location)

    print("CHECK OUTPUT")
    print(os.listdir(tmp_dir.name))

    forest = tr.Forest.load_from_rust(location, prefix="tmp", ifh="tmp.ifh", ofh="tmp.ofh",
                                      clusters="tmp.clusters", input="input.counts", output="output.counts")

    forest.arguments = arguments

    if tmp_dir is not None:
        tmp_dir.cleanup()

    return forest


def inner_fit(input_counts, output_counts, location, backtrace=False, lrg_mem=None, **kwargs):

    # targets = "\n".join(["\t".join([str(y) for y in x]) for x in targets]) + "\n"


    print("Running " + RUST_PATH)

    arg_list = []

    # if backtrace:
    #     arg_list.append("RUST_BACKTRACE=1")

    arg_list.extend([RUST_PATH, "-ic", location + "input.counts",
                     "-oc", location + "output.counts", "-o", location + "tmp", "-auto"])

    for arg in kwargs.keys():
        arg_list.append("-" + str(arg))
        arg_list.append(str(kwargs[arg]))

    if lrg_mem is not None:
        arg_list.append("-lrg_mem")

    print("Command: " + " ".join(arg_list))

    # cp = sp.run(arg_list,stdout=sp.PIPE,stderr=sp.PIPE,universal_newlines=True)

    # cp = sp.Popen(arg_list,stdin=sp.PIPE,stdout=sp.PIPE,stderr=sp.PIPE,universal_newlines=True)
    # try:
    #     output,error = cp.communicate(input=targets,timeout=1)
    # except:
    #     print("Communicated input")
    #
    with sp.Popen(arg_list, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE, universal_newlines=True) as cp:
        # try:
        #     cp.communicate(input=targets,timeout=1)
        # except:
        #     pass
        while True:
            # sleep(0.1)
            rc = cp.poll()
            if rc is not None:
                print(cp.stdout.read())
                print(cp.stderr.read())
                break
            output = cp.stdout.readline()
            # print("Read line")
            print(output.strip())

    return arg_list
    # while cp.poll() is None:
    #     sys.stdout.flush()
    #     sys.stdout.write("Constructing trees: %s" % str(len(glob.glob(location + "tmp.*.compact"))) + "\r")
    #     # sys.stdout.write(str(os.listdir(location)))
    #     sleep(1)

    # print(cp.stdout.read())
    #
    # print(cp.stderr.read())


if __name__ == "__main__":
    kwargs = {x.split("=")[0]: x.split("=")[1] for x in sys.argv[3:]}
    main(sys.argv[1], sys.argv[2], **kwargs)


# ,feature_sub=None,distance=None,sample_sub=None,scaling=None,merge_distance=None,refining=False,error_dump=None,convergence_factor=None,smoothing=None,locality=None)
