#!/usr/bin/env python

from __future__ import print_function
import re
import os
import argparse
import sys
import warnings
import copy
import glob


if __name__ == "__main__":
    # we add compulsory arguments as named arguments for readability
    parser = argparse.ArgumentParser(description="Create a list of models suitable for averaging "
                                                 "based on their train objf values.",
                                     epilog="See steps/nnet3/lstm/train.sh for example.")

    parser.add_argument("--difference-threshold", type=float,
                        help="The threshold for discarding models, "
                        "when objf of the model differs more than this value from the best model "
                        "it is discarded.",
                        default=1.0)

    parser.add_argument("num_models", type=int,
                        help="Number of models.")

    parser.add_argument("logfile_pattern", type=str,
                        help="Pattern for identifying the log-file names. "
                        "It specifies the entire log file name, except for the job number, "
                        "which is replaced with '%'. e.g. exp/nneet3/tdnn_sp/log/train.4.%.log")


    args = parser.parse_args()

    assert(args.num_models > 0)

    parse_regex = re.compile("LOG .* Overall average objective function for 'output' is ([0-9e.\-+]+) over ([0-9e.\-+]+) frames")
    loss = []
    for i in range(args.num_models):
        model_num = i + 1
        logfile = re.sub('%', str(model_num), args.logfile_pattern)
        lines = open(logfile, 'r').readlines()
        this_loss = -100000
        for line_num in range(1, len(lines) + 1):
            # we search from the end as this would result in
            # lesser number of regex searches. Python regex is slow !
            mat_obj = parse_regex.search(lines[-1*line_num])
            if mat_obj is not None:
                this_loss = float(mat_obj.groups()[0])
                break;
        loss.append(this_loss);
    max_index = loss.index(max(loss))
    accepted_models = []
    for i in range(args.num_models):
        if (loss[max_index] - loss[i]) <= args.difference_threshold:
            accepted_models.append(i+1)

    model_list = " ".join(map(lambda x: str(x), accepted_models))
    print(model_list)

    if len(accepted_models) != args.num_models:
        print("WARNING: Only {0}/{1} of the models have been accepted for averaging, based on log files {2}.".format(len(accepted_models), args.num_models, args.logfile_pattern), file=sys.stderr)
        print("         Using models {0}".format(model_list), file=sys.stderr)
