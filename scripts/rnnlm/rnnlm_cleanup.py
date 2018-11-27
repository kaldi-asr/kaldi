#!/usr/bin/env python3

import argparse
import sys
import os
import re

script_name = sys.argv[0]

# TODO decent description
parser = argparse.ArgumentParser(description="Removes models from past training iterations of "
                                             "RNNLM. Several strategies for picking which iterations "
                                             "to keep are available.",
                                 epilog="E.g. " + script_name + " exp/rnnlm_a",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("rnnlm_dir",
                    help="Directory where the RNNLM has been trained")
# parser.add_argument("last_iteration",
#                     help="Number of the last iteration",
#                     type=int)
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


def keep_best(iteration_dict, exp_dir):
    iters_to_keep = args.iters_to_keep
    best = []
    for iter, iter_files in iteration_dict.items():
        # this is roughly taken from get_best_model.py
        logfile = "{0}/log/compute_prob.{1}.log".format(exp_dir, iter)
        try:
            f = open(logfile, "r", encoding="latin-1")
        except:
            sys.exit(script_name + ": could not open log-file " + logfile)
        objf = -2000
        for line in f:
            m = re.search('Overall objf .* (\S+)$', str(line))
            if m is not None:
                try:
                    objf = float(m.group(1))
                except Exception as e:
                    sys.exit(script_name + ": line in file {0} could not be parsed: {1}, error is: {2}".format(
                        logfile, line, str(e)))
        if objf == -2000:
            print(script_name + ": warning: could not parse objective function from " + logfile, file=sys.stderr)
            continue
        # add potential best, sort by objf, trim to iters_to_keep size
        best.append((iter, objf))
        best = sorted(best, key=lambda x: -x[1])
        if len(best) > iters_to_keep:
            throwaway = best[iters_to_keep:]
            best = best[:iters_to_keep]
            # remove iters that we know are not the best
            for (iter, _) in throwaway:
                iter_files = iteration_dict[iter]
                os.remove(iter_files[0])
                os.remove(iter_files[1])


# grab all the iterations mapped to their word_embedding and .raw files
iterations = get_iteration_files(args.rnnlm_dir)
print(iterations)
# apply chosen cleanup strategy
if args.keep_latest:
    keep_latest(iterations)
else:
    keep_best(iterations, args.rnnlm_dir)
print(get_iteration_files(args.rnnlm_dir))
