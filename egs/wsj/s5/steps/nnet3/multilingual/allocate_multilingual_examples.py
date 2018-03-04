#!/usr/bin/env python3

# Copyright      2017 Pegah Ghahremani
#                2018 Hossein Hadian
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

    For chain training egs, the --egs-prefix option should be "cegs."

    You can call this script as (e.g.):

    allocate_multilingual_examples.py [opts] example-scp-lists
        multilingual-egs-dir

    allocate_multilingual_examples.py --block-size 512
        --lang2weight  "0.2,0.8" exp/lang1/egs.scp exp/lang2/egs.scp
        exp/multi/egs

"""

from __future__ import print_function
import os, argparse, sys, random
import logging
import traceback

sys.path.insert(0, 'steps')

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

    parser.add_argument("--num-archives", type=int, default=None,
                        help="Number of archives to split the data into. (Note: in reality they are not "
                        "archives, only scp files, but we use this notation by analogy with the "
                        "conventional egs-creating script).")
    parser.add_argument("--block-size", type=int, default=512,
                        help="This relates to locality of disk access. 'block-size' is"
                        "the average number of examples that are read consecutively"
                        "from each input scp file (and are written in the same order to the output scp files)"
                        "Smaller values lead to more random disk access (during "
                        "the nnet3 training process).")
    parser.add_argument("--egs-prefix", type=str, default="egs.",
                        help="This option can be used to add a prefix to the filenames "
                        "of the output files. For e.g. "
                        "if --egs-prefix=combine. , then the files produced "
                        "by this script will be "
                        "combine.output.*.ark, combine.weight.*.ark, and combine.*.scp")
    parser.add_argument("--lang2weight", type=str,
                        help="Comma-separated list of weights, one per language. "
                        "The language order is as egs_scp_lists.")
# now the positional arguments
    parser.add_argument("egs_scp_lists", nargs='+',
                        help="List of egs.scp files per input language."
                           "e.g. exp/lang1/egs/egs.scp exp/lang2/egs/egs.scp")
    parser.add_argument("egs_dir",
                        help="Name of output egs directory e.g. exp/tdnn_multilingual_sp/egs")


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
        lang2weight = [1.0] * num_langs
    else:
        lang2weight = args.lang2weight.split(",")
        assert(len(lang2weight) == num_langs)

    if not os.path.exists(os.path.join(args.egs_dir, 'info')):
        os.makedirs(os.path.join(args.egs_dir, 'info'))

    with open("{0}/info/{1}num_tasks".format(args.egs_dir, args.egs_prefix), "w") as fh:
        print("{0}".format(num_langs), file=fh)

    # Total number of egs in all languages
    tot_num_egs = sum(lang_to_num_examples[i] for i in range(num_langs))
    num_archives = args.num_archives

    with open("{0}/info/{1}num_archives".format(args.egs_dir, args.egs_prefix), "w") as fh:
        print("{0}".format(num_archives), file=fh)

    logger.info("There are a total of {} examples in the input scp "
                "files.".format(tot_num_egs))
    logger.info("Number of blocks in each output archive will be approximately "
                "{}, and block-size is {}.".format(int(round(tot_num_egs / num_archives / args.block_size)),
                                                   args.block_size))
    for lang in range(num_langs):
        blocks_per_archive_this_lang = lang_to_num_examples[lang] / num_archives / args.block_size
        warning = ""
        if blocks_per_archive_this_lang < 1.0:
            warning = ("Warning: This means some of the output archives might "
                       "not include any examples from this lang.")
        logger.info("The proportion of egs from lang {} is {:.2f}. The number of blocks "
                    "per archive for this lang is approximately {:.2f}. "
                    "{}".format(lang, lang_to_num_examples[lang] / tot_num_egs,
                                blocks_per_archive_this_lang,
                                warning))

    in_scp_file_handles = [open(scp_lists[lang], 'r') for lang in range(num_langs)]

    num_remaining_egs = tot_num_egs
    lang_to_num_remaining_egs = [n for n in lang_to_num_examples]
    for archive_index in range(num_archives + 1):  #  +1 is because we write to the last archive in two rounds
        num_remaining_archives = num_archives - archive_index
        num_remaining_blocks = num_remaining_egs / args.block_size

        last_round = (archive_index == num_archives)
        if not last_round:
            num_blocks_this_archive = int(round(num_remaining_blocks / num_remaining_archives))
            logger.info("Generating archive {} containing {} blocks...".format(archive_index, num_blocks_this_archive))
        else:  # This is the second round for the last archive. Flush all the remaining egs...
            archive_index = num_archives - 1
            num_blocks_this_archive = num_langs
            logger.info("Writing all the {} remaining egs to the last archive...".format(num_remaining_egs))

        out_scp_file_handle = open('{0}/{1}{2}.scp'.format(args.egs_dir, args.egs_prefix, archive_index + 1),
                                   'a' if last_round else 'w')
        eg_to_output_file_handle = open("{0}/{1}output.{2}.ark".format(args.egs_dir, args.egs_prefix, archive_index + 1),
                                        'a' if last_round else 'w')
        eg_to_weight_file_handle = open("{0}/{1}weight.{2}.ark".format(args.egs_dir, args.egs_prefix, archive_index + 1),
                                        'a' if last_round else 'w')


        for block_index in range(num_blocks_this_archive):
            # Find the lang with the highest proportion of remaining examples
            remaining_proportions = [remain / tot for remain, tot in zip(lang_to_num_remaining_egs, lang_to_num_examples)]
            lang_index, max_proportion = max(enumerate(remaining_proportions), key=lambda a: a[1])

            # Read 'block_size' examples from the selected lang and write them to the current output scp file:
            example_lines  = read_lines(in_scp_file_handles[lang_index], args.block_size)
            for eg_line in example_lines:
                eg_id = eg_line.split()[0]
                print(eg_line, file=out_scp_file_handle)
                print("{0} output-{1}".format(eg_id, lang_index), file=eg_to_output_file_handle)
                print("{0} {1}".format(eg_id, lang2weight[lang_index]), file=eg_to_weight_file_handle)

            num_remaining_egs -= len(example_lines)
            lang_to_num_remaining_egs[lang_index] -= len(example_lines)

        out_scp_file_handle.close()
        eg_to_output_file_handle.close()
        eg_to_weight_file_handle.close()

    for handle in in_scp_file_handles:
        handle.close()
    logger.info("Finished generating {0}*.scp, {0}output.*.ark "
                "and {0}weight.*.ark files. Wrote a total of {1} examples "
                "to {2} archives.".format(args.egs_prefix,
                                          tot_num_egs - num_remaining_egs, num_archives))


def main():
    try:
        args = get_args()
        process_multilingual_egs(args)
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
