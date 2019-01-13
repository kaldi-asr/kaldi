#!/usr/bin/env python3

# Copyright 2019    Johns Hopkins University (author: Daniel Povey)


# Apache 2.0.

""" This script outputs information about a neural net training schedule,
    to be used by ../train.py.
"""

import argparse
import sys

sys.path.insert(0, 'steps')
import libs.nnet3.train.common as common_train_lib
import libs.common as common_lib



def get_args():
    parser = argparse.ArgumentParser(
        description="Output training schedule information to be consumed by ../train.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--frame-subsampling-factor", type=int, default=3,
                        help="""Frame subsampling factor for the combined model
                        (bottom+top), will normally be 3.  Required here in order
                        to deal with frame-shifted versions of the input.""")
    parser.add_argument("--initial-effective-lrate",
                        type=float,
                        dest='initial_effective_lrate', default=0.001,
                        help="""Effective learning rate used on the first iteration,
                        determines schedule via geometric interpolation with
                        --final-effective-lrate.   Actual learning rate is
                        this times the num-jobs on that iteration.""")
    parser.add_argument("--final-effective-lrate", type=float,
                        dest='final_effective_lrate', default=0.0001,
                        help="""Learning rate used on the final iteration, see
                        --initial-effective-lrate for more documentation.""")
    parser.add_argument("--num-jobs-initial", type=int, default=1,
                        help="""Number of parallel neural net jobs to use at
                        the start of training""")
    parser.add_argument("--num-jobs-final", type=int, default=1,
                        help="""Number of parallel neural net jobs to use at
                        the end of training.  Would normally
                        be >= --num-jobs-initial""")
    parser.add_argument("--num-epochs", type=float, default=4.0,
                        help="""The number of epochs to train for.
                        Note: the 'real' number of times we see each
                        utterance is this number times --frame-subsampling-factor
                        (to cover frame-shifted copies of the data), times
                        the value of --num-repeats given to process_egs.sh,
                        times any factor arising from data augmentation.""")
    parser.add_argument("--dropout-schedule", type=str,
                        help="""Use this to specify the dropout schedule (how the dropout probability varies
                        with time, 0 == no dropout).  You specify a piecewise
                        linear function on the domain [0,1], where 0 is the
                        start and 1 is the end of training; the
                        function-argument (x) rises linearly with the amount of
                        data you have seen, not iteration number (this improves
                        invariance to num-jobs-{initial-final}).  E.g. '0,0.2,0'
                        means 0 at the start; 0.2 after seeing half the data;
                        and 0 at the end.  You may specify the x-value of
                        selected points, e.g.  '0,0.2@0.25,0' means that the 0.2
                        dropout-proportion is reached a quarter of the way
                        through the data.  The start/end x-values are at
                        x=0/x=1, and other unspecified x-values are interpolated
                        between known x-values.  You may specify different rules
                        for different component-name patterns using
                        'pattern1=func1 pattern2=func2', e.g. 'relu*=0,0.1,0
                        lstm*=0,0.2,0'.  More general should precede less
                        general patterns, as they are applied sequentially.""")

    parser.add_argument("--schedule-out", type=str, required=True,
                        "Output file containing the training schedule.  The output
                        is lines, one per training iteration.  Each line contains
                        tab-separated fields of the form:
                        <iteration-index> <num-jobs> <scp-indexes> <dropout-string> <learning-rate> <frame-shifts>
                        where <iteration-index> is an iteration index starting from 0,
                        <num-jobs> is the number of jobs for this iteration (between
                        num-jobs-initial and num-jobs-final),
                        <scp-indexes> is a space-separated string containing the
                        indexes of the .scp files in the egs dir to use for this
                        iteration (e.g. '1 2 3'), <dropout-string> is either the empty
                        string or something to be passed to the --edits command of
                        nnet3-am-copy or nnet3-copy; <learning-rate> is the
                        actual learning rate on this iteration (the effective learning
                        rate times the num-jobs), and <frame-shifts> is a space-separated
                        string containing the frame shifts for each job.")



def main():
    pass
if __name__ == "__main__":
    main()
