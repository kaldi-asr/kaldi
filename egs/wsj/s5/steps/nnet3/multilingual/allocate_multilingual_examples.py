#!/usr/bin/env python3

# Copyright 2017 Pegah Ghahremani
#
# Apache 2.0.

""" This script generates examples for multilingual training of neural network.
    This scripts produces 3 sets of files --
    egs.*.scp, egs.output.*.ark, egs.weight.*.ark

    egs.*.scp are the SCP files of the training examples.
    egs.weight.*.ark map from the key of the example to the language-specific
    weight of that example.
    egs.output.*.ark map from the key of the example to the name of
    the output-node in the neural net for that specific language, e.g.
    'output-2'.

    --egs-prefix option can be used to generate train and diagnostics egs files.
    If --egs-prefix=train_diagnostics. is passed, then the files produced by the
    script will be named with the prefix as "train_diagnostics."
    instead of "egs."
    i.e. the files produced are -- train_diagnostics.*.scp,
    train_diagnostics.output.*.ark, train_diagnostics.weight.*.ark and
    train_diagnostics.ranges.*.txt.
    The other egs-prefix options used in the recipes are "valid_diagnositics."
    for validation examples and "combine." for examples used for model
    combination.

    You can call this script as (e.g.):

    allocate_multilingual_examples.py [opts] example-scp-lists
        multilingual-egs-dir

    allocate_multilingual_examples.py --shuffle-factor 2
        --lang2weight  "0.2,0.8" exp/lang1/egs.scp exp/lang2/egs.scp
        exp/multi/egs

"""

from __future__ import print_function
import os, argparse, sys, random
import logging
import traceback

sys.path.insert(0, 'steps')
#import libs.common as common_lib

logger = logging.getLogger('libs')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(pathname)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('Start generating multilingual examples')


def get_args():

    parser = argparse.ArgumentParser(
        description=""" This script generates examples for multilingual training
        of neural network by producing 3 sets of primary files
        as egs.*.scp, egs.output.*.ark, egs.weight.*.ark.
        egs.*.scp are the SCP files of the training examples.
        egs.weight.*.ark map from the key of the example to the language-specific
        weight of that example.
        egs.output.*.ark map from the key of the example to the name of
        the output-node in the neural net for that specific language, e.g.
        'output-2'.""",
        epilog="Called by steps/nnet3/multilingual/combine_egs.sh")

    parser.add_argument("--samples-per-iter", type=int, default=40000,
                        help="The target number of egs in each archive of egs, "
                        "(prior to merging egs). ")
    parser.add_argument("--num-archives", type=int, default=None,
                        help="number of output archives. If not set, it will be "
                        "determined using --samples-per-iter.")
    parser.add_argument("--seed", type=int, default=1,
                        help="Seed for random number generator")
    parser.add_argument("--shuffle-factor", type=int, default=1,
                        help="shuffle-factor N means for each output scp file, we read exactly "
                        "N block of examples from each input scp file in a round-robin manner."
                        "Larger values mean more shuffling and more random disk access.")
    parser.add_argument("--egs-prefix", type=str, default="egs.",
                        help="option can be used to generated example scp, weight "
                        "and output files for training and diagnostics."
                        "If --egs-prefix=combine. , then files produced "
                        "by the sript will be named with this prefix as "
                        "combine.output.*.ark, combine.weight.*.ark, combine.*.scp, "
                        "combine.ranges.*.ark.")
    parser.add_argument("--lang2weight", type=str,
                        help="comma-separated list of weights, one per language."
                        "The language order is as egs_scp_lists.")
# now the positional arguments
    parser.add_argument("egs_scp_lists", nargs='+',
                        help="list of egs.scp files per input language."
                           "e.g. exp/lang1/egs/egs.scp exp/lang2/egs/egs.scp")
    parser.add_argument("egs_dir",
                        help="Name of egs directory e.g. exp/tdnn_multilingual_sp/egs")


    print(sys.argv, file=sys.stderr)
    args = parser.parse_args()

    return args




def read_lines(file_handle, num_lines):
    n_read = 0
    lines = []
    while n_read < num_lines:
        line = file_handle.readline()
        if not line:
            break
        lines.append(line.strip())
        n_read += 1
    return lines


def process_multilingual_egs(args):
    args = get_args()

    scp_lists = args.egs_scp_lists
    num_langs = len(scp_lists)

    lang_to_num_examples = [0] * num_langs
    for lang in range(num_langs):
        with open(scp_lists[lang]) as fh:
            lang_to_num_examples[lang] = sum([1 for line in fh])
        logger.info("Number of examples for language {0} "
                    "is {1}.".format(lang, lang_to_num_examples[lang]))

    # If weights are not provided, the weights are 1.0.
    if args.lang2weight is None:
        lang2weight = [ 1.0 ] * num_langs
    else:
        lang2weight = args.lang2weight.split(",")
        assert(len(lang2weight) == num_langs)

    if not os.path.exists(os.path.join(args.egs_dir, 'info')):
        os.makedirs(os.path.join(args.egs_dir, 'info'))

    with open("{0}/info/{1}num_tasks".format(args.egs_dir, args.egs_prefix), "w") as fh:
        print("{0}".format(num_langs), file=fh)

    # Total number of egs in all languages
    tot_num_egs = sum(lang_to_num_examples[i] for i in range(num_langs))
    if args.num_archives is not None:
        num_archives = args.num_archives
    else:
        num_archives = tot_num_egs // args.samples_per_iter + 1


    with open("{0}/info/{1}num_archives".format(args.egs_dir, args.egs_prefix), "w") as fh:
        print("{0}".format(num_archives), file=fh)

    egs_per_archive = tot_num_egs // num_archives
    base_block_size = egs_per_archive // args.shuffle_factor

    # The block size for each lang is calculated based on its size
    lang_to_block_size = [int(lang_to_num_examples[i] / tot_num_egs * base_block_size) for i in range(num_langs)]

    logger.info("egs-per-archive for the output scp files is {}".format(egs_per_archive))

    for lang in range(num_langs):
        logger.info("egs per block for lang {} is {}".format(lang, lang_to_block_size[lang]))

    in_scp_file_handles = [open(scp_lists[lang], 'r') for lang in range(num_langs)]


    # For each output scp file, read the examples in a round-robin fashion
    # from the inputs scp files. Each time a block is read, where the size of the
    # block is proportional to the total number of egs in the corresponding lang,
    # so that there is a bit of each lang in each generated scp file.
    for archive_index in range(num_archives):
        logger.info("Generating archive {}...".format(archive_index))

        out_scp_file_handle = open('{0}/{1}{2}.scp'.format(args.egs_dir, args.egs_prefix,
                                                           archive_index + 1), 'w')
        eg_to_output_file_handle = open("{0}/{1}output.{2}.ark".format(args.egs_dir,
                                                                       args.egs_prefix, archive_index + 1), 'w')
        eg_to_weight_file_handle = open("{0}/{1}weight.{2}.ark".format(
            args.egs_dir, args.egs_prefix, archive_index + 1), 'w')

        for round_index in range(args.shuffle_factor):
            # Read 'block_size' examples from each lang and write them to the current output scp file:
            for lang_index in range(num_langs):
                example_lines  = read_lines(in_scp_file_handles[lang_index], lang_to_block_size[lang_index])

                for eg_line in example_lines:
                    eg_id = eg_line.split()[0]

                    print(eg_line, file=out_scp_file_handle)
                    print("{0} output-{1}".format(eg_id, lang_index), file=eg_to_output_file_handle)
                    print("{0} {1}".format(eg_id, lang2weight[lang_index]), file=eg_to_weight_file_handle)

        out_scp_file_handle.close()
        eg_to_output_file_handle.close()
        eg_to_weight_file_handle.close()

    logger.info("Finished generating {0}*.scp, {0}output.*.ark "
                "and {0}weight.*.ark files.".format(args.egs_prefix))


def main():
    try:
        args = get_args()
        process_multilingual_egs(args)
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
  main()
