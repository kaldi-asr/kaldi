#!/usr/bin/env python3

import argparse
import sys
import os

script_name = sys.argv[0]

# TODO decent description
parser = argparse.ArgumentParser(description="Removes models from past training iterations of "
                                             "RNNLM. Several strategies for picking which iterations "
                                             "to keep are available.",
                                 epilog="E.g. " + script_name + " exp/rnnlm_a",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("rnnlm_dir",
                    help="Directory where the RNNLM has been trained")
parser.add_argument("last_iteration",
                    help="Number of the last iteration",
                    type=int)
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
    sys.exit(script_name + ": can only use either 'keep_latest' or 'keep_best', but not both")
elif not args.keep_latest and not args.keep_best:
    sys.exit(script_name + ": no cleanup strategy specified: use 'keep_latest' or 'keep_best'")

# TODO now for some actual logic............
# check exp dir for model files
# list all files there, look for word_embedding.%d.mat and %d.raw files
# if keep_best, check compute_prob logs for best eval scores or adapt/use get_best_model.py
# if keep_latest, find the latest iteration that is not used or rely on last_iteration arg?


def get_iteration_files(exp_dir):
    iterations = dict()
    for f in os.listdir(exp_dir):
        joined_f = os.path.join(exp_dir, f)
        if os.path.isfile(joined_f) and (f.startswith("word_embedding") or f.endswith(".raw")):
            split = f.split(".")
            ext = split[-1]
            iter = int(split[-2])
            if iter in iterations:
                if ext == "raw":
                    iterations[iter] = (iterations[iter][0], joined_f)
                else:
                    iterations[iter] = (joined_f, iterations[iter][1])
            else:
                if ext == "raw":
                    iterations[iter] = (None, joined_f)
                else:
                    iterations[iter] = (joined_f, None)
    return iterations


def keep_latest(iteration_dict):
    max_to_keep = args.iters_to_keep
    kept = 0
    iterations_in_reverse_order = reversed(sorted(iteration_dict))
    for iter in iterations_in_reverse_order:
        if kept < max_to_keep:
            kept += 1
        else:
            iter_files = iteration_dict[iter]
            os.remove(iter_files[0])
            os.remove(iter_files[1])


# TODO just testing
iterations = get_iteration_files(args.rnnlm_dir)
print(iterations)
keep_latest(iterations)
print(get_iteration_files(args.rnnlm_dir))


# TODO implement rest of the bookkeeping