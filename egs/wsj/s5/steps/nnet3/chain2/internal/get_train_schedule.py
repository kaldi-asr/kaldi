#!/usr/bin/env python3

# Copyright 2019    Johns Hopkins University (author: Daniel Povey)
# Copyright         Hossein Hadian
# Copyright   2019  Idiap Research Institute (Author: Srikanth Madikeri).  


# Apache 2.0.

""" This script outputs information about a neural net training schedule,
    to be used by ../train.sh, in the form of lines that can be selected
    and sourced by the shell.
"""

import argparse
import sys

sys.path.insert(0, 'steps')
import libs.nnet3.train.common as common_train_lib
import libs.common as common_lib

def get_args():
    parser = argparse.ArgumentParser(
        description="""Output training schedule information to be consumed by ../train.sh""",
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

    parser.add_argument("--num-scp-files", type=int, default=0, required=True,
                        help="""The number of .scp files in the egs dir.""")
    parser.add_argument("--schedule-out", type=str, required=True,
                        help="""Output file containing the training schedule.  The output
                        is lines, one per training iteration.
                        Each line (one per iteration) is a list of ;-separated commands setting shell
                        variables.  Currently the following variables are set:
                        iter, num_jobs, inv_num_jobs, scp_indexes, frame_shifts, dropout_opt, lrate.
                        """)

    print(sys.argv, file=sys.stderr)
    args = parser.parse_args()

    return args

def get_schedules(args):
    num_scp_files_expanded = args.num_scp_files * args.frame_subsampling_factor
    num_scp_files_to_process = int(args.num_epochs * num_scp_files_expanded)
    num_scp_files_processed = 0
    num_iters = ((num_scp_files_to_process * 2)
                 // (args.num_jobs_initial + args.num_jobs_final))

    with open(args.schedule_out, 'w', encoding='latin-1') as ostream:
        for iter in range(num_iters):
            current_num_jobs = int(0.5 + args.num_jobs_initial
                                   + (args.num_jobs_final - args.num_jobs_initial)
                                   * float(iter) / num_iters)
            # as a special case, for iteration zero we use just one job
            # regardless of the --num-jobs-initial and --num-jobs-final.  This
            # is because the model averaging does not work reliably for a
            # freshly initialized model.
            # if iter == 0:
            #     current_num_jobs = 1

            lrate = common_train_lib.get_learning_rate(iter, current_num_jobs,
                                                       num_iters,
                                                       num_scp_files_processed,
                                                       num_scp_files_to_process,
                                                       args.initial_effective_lrate,
                                                       args.final_effective_lrate)

            if args.dropout_schedule == "":
                args.dropout_schedule = None
            dropout_edit_option = common_train_lib.get_dropout_edit_option(
                args.dropout_schedule,
                float(num_scp_files_processed) / max(1, (num_scp_files_to_process - args.num_jobs_final)),
                iter)

            frame_shifts = []
            egs = []
            for job in range(1, current_num_jobs + 1):
                # k is a zero-based index that we will derive the other indexes from.
                k = num_scp_files_processed + job - 1
                # work out the 1-based scp index.
                scp_index = (k % args.num_scp_files) + 1
                # previous : frame_shift = (k/num_scp_files) % frame_subsampling_factor
                frame_shift = ((scp_index + k // args.num_scp_files)
                               % args.frame_subsampling_factor)

                # Instead of frame shifts like [0, 1, 2], we make them more like
                # [0, 1, -1].  This is clearer in intent, and keeps the
                # supervision starting at frame zero, which IIRC is a
                # requirement somewhere in the 'chaina' code.
#               TODO: delete this section if no longer useful
                # if frame_shift > (args.frame_subsampling_factor // 2):
                #     frame_shift = frame_shift - args.frame_subsampling_factor

                frame_shifts.append(str(frame_shift))
                egs.append(str(scp_index))


            print("""iter={iter}; num_jobs={nj}; inv_num_jobs={nj_inv}; scp_indexes=(pad {indexes}); frame_shifts=(pad {shifts}); dropout_opt="{opt}"; lrate={lrate}""".format(
                iter=iter, nj=current_num_jobs, nj_inv=(1.0 / current_num_jobs),
                indexes = ' '.join(egs), shifts=' '.join(frame_shifts),
                opt=dropout_edit_option, lrate=lrate), file=ostream)
            num_scp_files_processed = num_scp_files_processed + current_num_jobs


def main():
    args = get_args()
    get_schedules(args)

if __name__ == "__main__":
    main()
