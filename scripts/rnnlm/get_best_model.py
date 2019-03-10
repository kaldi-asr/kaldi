#!/usr/bin/env python3

# Copyright  2017  Johns Hopkins University (author: Daniel Povey)
# License: Apache 2.0.

import argparse
import glob
import re
import sys

parser = argparse.ArgumentParser(description="Works out the best iteration of RNNLM training "
                                             "based on dev-set perplexity, and prints the number corresponding "
                                             "to that iteration",
                                 epilog="E.g. " + sys.argv[0] + " exp/rnnlm_a",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("rnnlm_dir",
                    help="Directory where the RNNLM has been trained")

args = parser.parse_args()

num_iters = None
try:
    with open(args.rnnlm_dir + "/info.txt", encoding="latin-1") as f:
        for line in f:
            a = line.split("=")
            if a[0] == "num_iters":
                num_iters = int(a[1])
                break
except:
    sys.exit(sys.argv[0] + ": could not find {0}/info.txt (or wrong format)".format(
        args.rnnlm_dir))

if num_iters is None:
    sys.exit(sys.argv[0] + ": could not get num_iters from {0}/info.txt".format(
        args.rnnlm_dir))

best_objf = -2000
best_iter = -1
for i in range(1, num_iters):
    this_logfile = "{0}/log/compute_prob.{1}.log".format(args.rnnlm_dir, i)
    try:
        f = open(this_logfile, 'r', encoding='latin-1')
    except:
        sys.exit(sys.argv[0] + ": could not open log-file {0}".format(this_logfile))
    this_objf = -1000
    for line in f:
        m = re.search('Overall objf .* (\S+)$', str(line))
        if m is not None:
            try:
                this_objf = float(m.group(1))
            except Exception as e:
                sys.exit(sys.argv[0] + ": line in file {0} could not be parsed: {1}, error is: {2}".format(
                    this_logfile, line, str(e)))
    # verify this iteration still has model files present
    if len(glob.glob("{0}/{1}.raw".format(args.rnnlm_dir, i))) == 0:
        # this iteration has log files, but model files have been cleaned up, skip it
        continue
    if this_objf == -1000:
        print(sys.argv[0] + ": warning: could not parse objective function from {0}".format(
            this_logfile), file=sys.stderr)
    if this_objf > best_objf:
        best_objf = this_objf
        best_iter = i

if best_iter == -1:
    sys.exit(sys.argv[0] + ": error: could not get best iteration.")

print(str(best_iter))
