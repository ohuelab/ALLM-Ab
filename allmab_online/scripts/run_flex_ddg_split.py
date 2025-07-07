#!/usr/bin/python

from __future__ import print_function

import socket
import sys
import os
import subprocess
import argparse

use_multiprocessing = True
if use_multiprocessing:
    import multiprocessing

    max_cpus = 35  # We might want to not run on the full number of cores, as Rosetta take about 2 Gb of memory per instance

###################################################################################################################################################################
# Important: The variables below are set to values that will make the run complete faster (as a tutorial example), but will not give scientifically valid results.
#            Please change them to the "normal" default values before a real run.
###################################################################################################################################################################

rosetta_scripts_path = os.path.expanduser(
    "rosetta_scripts.linuxgccrelease"
)
path_to_script = "ddG-backrub.xml"

if not os.path.isfile(rosetta_scripts_path):
    print(
        'ERROR: "rosetta_scripts_path" variable must be set to the location of the "rosetta_scripts" binary executable'
    )
    print('This file might look something like: "rosetta_scripts.linuxgccrelease"')
    raise Exception("Rosetta scripts missing")


def run_flex_ddg(
    name,
    input_path,
    input_pdb_path,
    output_path,
    chains_to_move,
    nstruct_i,
    nstruct,
    max_minimization_iter,
    abs_score_convergence_thresh,
    number_backrub_trials,
    backrub_trajectory_stride,
):
    output_directory = os.path.join(output_path, "%02d" % nstruct_i)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)

    flex_ddg_args = [
        os.path.abspath(rosetta_scripts_path),
        "-s %s" % os.path.abspath(input_pdb_path),
        "-parser:protocol",
        os.path.abspath(path_to_script),
        "-parser:script_vars",
        "chainstomove=" + chains_to_move,
        "mutate_resfile_relpath="
        + os.path.abspath(os.path.join(input_path, "nataa_mutations.resfile")),
        "number_backrub_trials=%d" % number_backrub_trials,
        "max_minimization_iter=%d" % max_minimization_iter,
        "abs_score_convergence_thresh=%.1f" % abs_score_convergence_thresh,
        "backrub_trajectory_stride=%d" % backrub_trajectory_stride,
        "-restore_talaris_behavior",
        "-in:file:fullatom",
        "-ignore_unrecognized_res",
        "-ignore_zero_occupancy false",
        "-ex1",
        "-ex2",
    ]

    log_path = os.path.join(output_directory, "rosetta.out")

    print("Running Rosetta with args:")
    print(" ".join(flex_ddg_args))
    print("Output logged to:", os.path.abspath(log_path))
    print()

    outfile = open(log_path, "w")
    process = subprocess.Popen(
        flex_ddg_args,
        stdout=outfile,
        stderr=subprocess.STDOUT,
        close_fds=True,
        cwd=output_directory,
    )
    returncode = process.wait()
    outfile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Rosetta flex ddG")
    parser.add_argument(
        "--input_path", type=str, default="inputs", help="Path to input files (default: inputs)"
    )
    parser.add_argument(
        "--output_path", type=str, default="output", help="Path to output files (default: output)"
    )
    parser.add_argument(
        "--nstruct", type=int, default=35, help="Number of structures (default: 35)"
    )
    parser.add_argument(
        "--max_min_iter",
        type=int,
        default=5000,
        help="Max minimization iterations (default: 5000)",
    )
    parser.add_argument(
        "--abs_score_conv",
        type=float,
        default=1.0,
        help="Absolute score convergence threshold (default: 1.0)",
    )
    parser.add_argument(
        "--backrub_trials",
        type=int,
        default=35000,
        help="Number of backrub trials (default: 35000)",
    )
    parser.add_argument(
        "--backrub_stride",
        type=int,
        default=7000,
        help="Backrub trajectory stride (default: 7000)",
    )
    parser.add_argument(
        "--max_cpus",
        type=int,
        default=-1,
        help="Maximum number of CPUs to use (default: -1, use all available)",
    )
    parser.add_argument(
        "--index", type=int, default=0, help="Index of the first case to run (default: 0)"
    )
    parser.add_argument(
        "--offset", type=int, default=1000, help="Number of cases to run (default: 1)"
    )
    args = parser.parse_args()

    cases = []
    case_names = sorted(os.listdir(args.input_path))[args.index:args.index*args.offset+args.offset]
    for nstruct_i in range(1, args.nstruct + 1):
        for case_name in case_names:
            case_path = os.path.join(args.input_path, case_name)
            output_path = os.path.join(args.output_path, case_name)
            for f in os.listdir(case_path):
                if f .endswith(".pdb"):
                    input_pdb_path = os.path.join(case_path, f)
                    break

            with open(os.path.join(case_path, "chains_to_move.txt"), "r") as f:
                chains_to_move = f.readlines()[0].strip()

            cases.append(
                (
                    case_name,
                    case_path,
                    input_pdb_path,
                    output_path,
                    chains_to_move,
                    nstruct_i,
                    args.nstruct,
                    args.max_min_iter,
                    args.abs_score_conv,
                    args.backrub_trials,
                    args.backrub_stride,
                )
            )

    if use_multiprocessing:
        max_cpus = (
            min(args.max_cpus, multiprocessing.cpu_count())
            if args.max_cpus > 0
            else min(max_cpus, multiprocessing.cpu_count())
        )
        pool = multiprocessing.Pool(processes=max_cpus)

    for case_args in cases:
        print(case_args[0], case_args[5])
        if use_multiprocessing:
            pool.apply_async(run_flex_ddg, args=case_args)
        else:
            run_flex_ddg(*case_args)

    if use_multiprocessing:
        pool.close()
        pool.join()
