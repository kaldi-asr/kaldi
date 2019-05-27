#!/usr/bin/env python3

# Copyright 2018 Tilde
# License: Apache 2.0

import sys

import argparse
import os
import re
import glob

script_name = sys.argv[0]

parser = argparse.ArgumentParser(description="Removes models from past training iterations of "
                                             "RNNLM. Can use either 'keep_latest' (default) or "
                                             "'keep_best' cleanup strategy, where former keeps "
                                             "the models that are freshest, while latter keeps "
                                             "the models with best training objective score on "
                                             "dev set.",
                                 epilog="E.g. " + script_name + " exp/rnnlm_a --keep_best",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("rnnlm_dir",
                    help="Directory where the RNNLM has been trained")
parser.add_argument("--iters_to_keep",
                    help="Max number of iterations to keep",
                    type=int,
                    default=3)
parser.add_argument("--keep_latest",
                    help="Keeps the training iterations that are latest by age",
                    action="store_const",
                    const=True,
                    default=False)
parser.add_argument("--keep_best",
                    help="Keeps the training iterations that have the best objf",
                    action="store_const",
                    const=True,
                    default=False)

args = parser.parse_args()

# validate arguments
if args.keep_latest and args.keep_best:
    sys.exit(script_name + ": can only use one of 'keep_latest' or 'keep_best', but not both")
elif not args.keep_latest and not args.keep_best:
    sys.exit(script_name + ": no cleanup strategy specified: use 'keep_latest' or 'keep_best'")


class IterationInfo:
    def __init__(self, model_files, objf, compute_prob_done):
        self.model_files = model_files
        self.objf = objf
        self.compute_prob_done = compute_prob_done

    def __str__(self):
        return "{model_files: %s, compute_prob: %s, objf: %2.3f}" % (self.model_files,
                                                                     self.compute_prob_done,
                                                                     self.objf)

    def __repr__(self):
        return self.__str__()


def get_compute_prob_info(log_file):
    # we want to know 3 things: iteration number, objf and whether compute prob is done
    iteration = int(log_file.split(".")[-2])
    objf = -2000
    compute_prob_done = False
    # roughly based on code in get_best_model.py
    try:
        f = open(log_file, "r", encoding="utf-8")
    except:
        print(script_name + ": warning: compute_prob log not found for iteration " +
              str(iter) + ". Skipping",
              file=sys.stderr)
        return iteration, objf, compute_prob_done
    for line in f:
        objf_m = re.search('Overall objf .* (\S+)$', str(line))
        if objf_m is not None:
            try:
                objf = float(objf_m.group(1))
            except Exception as e:
                sys.exit(script_name + ": line in file {0} could not be parsed: {1}, error is: {2}".format(
                    log_file, line, str(e)))
        if "# Ended" in line:
            compute_prob_done = True
    if objf == -2000:
        print(script_name + ": warning: could not parse objective function from " + log_file, file=sys.stderr)
    return iteration, objf, compute_prob_done


def get_iteration_files(exp_dir):
    iterations = dict()
    compute_prob_logs = glob.glob(exp_dir + "/log/compute_prob.[0-9]*.log")
    for log in compute_prob_logs:
        iteration, objf, compute_prob_done = get_compute_prob_info(log)
        if iteration == 0:
            # iteration 0 is special, never consider it for cleanup
            continue
        if compute_prob_done:
            # this iteration can be safely considered for cleanup
            # gather all model files belonging to it
            model_files = []
            # when there are multiple jobs per iteration, there can be several model files
            # we need to potentially clean them all up without mixing them up
            model_files.extend(glob.glob("{0}/word_embedding.{1}.mat".format(exp_dir, iteration)))
            model_files.extend(glob.glob("{0}/word_embedding.{1}.[0-9]*.mat".format(exp_dir, iteration)))
            model_files.extend(glob.glob("{0}/feat_embedding.{1}.mat".format(exp_dir, iteration)))
            model_files.extend(glob.glob("{0}/feat_embedding.{1}.[0-9]*.mat".format(exp_dir, iteration)))
            model_files.extend(glob.glob("{0}/{1}.raw".format(exp_dir, iteration)))
            model_files.extend(glob.glob("{0}/{1}.[0-9]*.raw".format(exp_dir, iteration)))
            # compute_prob logs outlive model files, only consider iterations that do still have model files
            if len(model_files) > 0:
                iterations[iteration] = IterationInfo(model_files, objf, compute_prob_done)
    return iterations


def remove_model_files_for_iter(iter_info):
    for f in iter_info.model_files:
        os.remove(f)


def keep_latest(iteration_dict):
    max_to_keep = args.iters_to_keep
    kept = 0
    iterations_in_reverse_order = reversed(sorted(iteration_dict))
    for iter in iterations_in_reverse_order:
        if kept < max_to_keep:
            kept += 1
        else:
            remove_model_files_for_iter(iteration_dict[iter])


def keep_best(iteration_dict):
    iters_to_keep = args.iters_to_keep
    best = []
    for iter, iter_info in iteration_dict.items():
        objf = iter_info.objf
        if objf == -2000:
            print(script_name + ": warning: objf unavailable for iter " + str(iter), file=sys.stderr)
            continue
        # add potential best, sort by objf, trim to iters_to_keep size
        best.append((iter, objf))
        best = sorted(best, key=lambda x: -x[1])
        if len(best) > iters_to_keep:
            throwaway = best[iters_to_keep:]
            best = best[:iters_to_keep]
            # remove iters that we know are not the best
            for (iter, _) in throwaway:
                remove_model_files_for_iter(iteration_dict[iter])


# grab all the iterations mapped to their model files, objf score and compute_prob status
iterations = get_iteration_files(args.rnnlm_dir)
# apply chosen cleanup strategy
if args.keep_latest:
    keep_latest(iterations)
else:
    keep_best(iterations)
