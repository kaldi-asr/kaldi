#!/usr/bin/env python3

import sys

import argparse
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


class IterationInfo:
    def __init__(self, word_embedding_file, raw_file, objf, compute_prob_done):
        self.word_embedding_file = word_embedding_file
        self.raw_file = raw_file
        self.objf = objf
        self.compute_prob_done = compute_prob_done

    def __str__(self):
        return "{word_embedding: %s, raw: %s, compute_prob: %s, objf: %2.3f}" % (self.word_embedding_file,
                                                                                 self.raw_file,
                                                                                 self.compute_prob_done,
                                                                                 self.objf)

    def __repr__(self):
        return self.__str__()


def get_compute_prob_info(exp_dir, iter):
    # roughly based on code in get_best_model.py
    log_file = "{0}/log/compute_prob.{1}.log".format(exp_dir, iter)
    try:
        f = open(log_file, "r", encoding="latin-1")
    except:
        sys.exit(script_name + ": could not open log-file " + log_file)
    # we now want 2 things: objf and whether compute prob is done
    objf = -2000
    compute_prob_done = False
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
    return objf, compute_prob_done


def get_iteration_files(exp_dir):
    iterations = dict()
    for f in os.listdir(exp_dir):
        joined_f = os.path.join(exp_dir, f)
        if os.path.isfile(joined_f) and (f.startswith("word_embedding") or f.endswith(".raw")):
            split = f.split(".")
            ext = split[-1]
            iter = int(split[-2])
            objf, compute_prob_done = get_compute_prob_info(exp_dir, iter)
            if iter in iterations:
                iter_info = iterations[iter]
                if ext == "raw":
                    iter_info.raw_file = joined_f
                else:
                    iter_info.word_embedding_file = joined_f
                iter_info.objf = objf
                iter_info.compute_prob_done = compute_prob_done
            else:
                if ext == "raw":
                    iterations[iter] = IterationInfo(None, joined_f, objf, compute_prob_done)
                else:
                    iterations[iter] = IterationInfo(joined_f, None, objf, compute_prob_done)
    return iterations


def remove_model_files_for_iter(iter_info):
    os.remove(iter_info.word_embedding_file)
    os.remove(iter_info.raw_file)


def keep_latest(iteration_dict):
    max_to_keep = args.iters_to_keep
    kept = 0
    iterations_in_reverse_order = reversed(sorted(iteration_dict))
    for iter in iterations_in_reverse_order:
        # check if compute prob is done for this iteration, if not, leave it for future cleanups...
        if iteration_dict[iter].compute_prob_done:
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
        if not iter_info.compute_prob_done:
            # if compute_prob is not done, yet, we leave it for future cleanups
            print(script_name + ": warning: compute_prob not done yet for iter " + str(iter), file=sys.stderr)
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


# grab all the iterations mapped to their word_embedding and .raw files
iterations = get_iteration_files(args.rnnlm_dir)
# print(iterations)  # FIXME remove
# apply chosen cleanup strategy
if args.keep_latest:
    keep_latest(iterations)
else:
    keep_best(iterations)
# print(get_iteration_files(args.rnnlm_dir))  # FIXME remove
