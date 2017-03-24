#!/usr/bin/env python

# Copyright 2017 Pegah Ghahremani
#
# Apache 2.0.

""" This script generates egs.*.scp used for multilingual setup.
    Also this script generates output.*.ark and weight.*.ark.
    weight.*.ark map from the key of the example to the language-specific
    weight of that example.
    output.*.ark map from the key of the example to the name of the output-node
    in the neural net for that specific language, e.g. 'output-2'.

    ranges.*.txt are temporary files that are used by the script itself,
    each one corresponding to one 'egs' file that will be generated.
    Each line in ranges.*.txt corresponds to ranges of  examples
    selected from one of the input languages's scp files as:
    <lang> <local-scp-line> <num-examples>

    That can be interpreted as selecting <num-example> examples starting from
    <local-scp-line> line from {lang}_th 'egs' file in "egs_scp_list".
    (note that <local-scp-line> is the zero-based line number.)

    Example lines might look like:
    0 0 256
    2 1024 256

    egs.*.scp is generated using ranges.*.txt as following:
    "<num-examples>" consecutive examples starting from line "<local-scp-line>" from
    {lang}_th input "egs" scp-file is copied to egs.*.scp.

    You can call this script as (e.g.):

    allocate_multilingual_examples.py [opts] example-scp-lists
        multilingual-egs-dir

    allocate_multilingual_examples.py --num-jobs 10 --minibatch-size 512
        --lang2weight  "0.2,0.8" exp/lang1/egs.scp exp/lang2/egs.scp
        exp/multi/egs

    To avoid loading whole scp files from all languages in memory,
    input egs.scp files are processed line by line using readline() for input languages
    and to have more randomization across different archives,
    "num_jobs * num_archives" temporary scp.job.archive_index files created in
    egs/temp dir and all "num_jobs" scp.*.archive_index combined
    into egs.archiv_index.scp.
"""

from __future__ import print_function
import os, argparse, sys, random
import logging
import traceback

sys.path.insert(0, 'steps')
import libs.common as common_lib

logger = logging.getLogger('libs')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(pathname)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('Start generating multilingual examples (allocate_multilingual_examples.py)')


def get_args():

    parser = argparse.ArgumentParser(description="Writes ranges.*, output.* and weight.* files "
                                   "in preparation for dumping egs for multilingual training.",
                                   epilog="Called by steps/nnet3/multilingual/get_egs.sh")
    parser.add_argument("--samples-per-iter", type=int, default=40000,
                        help="The target number of egs in each archive of egs, "
                        "(prior to merging egs). ")
    parser.add_argument("--num-jobs", type=int, default=20,
                        help="This can be used for better randomization in distributing "
                        "languages's examples across archives, where egs.job.*.scp are generated "
                        "randomly and examples are combined across all jobs.")
    parser.add_argument("--random-lang", type=str, action=common_lib.StrToBoolAction,
                        help="If true, ranges.*.txt are generated"
                        " w.r.t frequency distribution of remaining examples in each language,"
                        " otherwise it is generated sequentially.",
                      default=True, choices = ["false", "true"])
    parser.add_argument("--max-archives", type=int, default=1000,
                        help="max number of archives used to generate egs.*.scp")
    parser.add_argument("--seed", type=int, default=1,
                        help="Seed for random number generator")
    parser.add_argument("--minibatch-size", type=int, default=512,
                        help="It is the number of consecutive egs that is taken "
                        "from each source, and it only affects locality of disk access."
                        " This does not have to be actual minibatch size.")
    parser.add_argument("--prefix", type=str, default="egs.",
                        help="Adds a prefix to the range files. This is used to "
                        "distinguish between the train and diagnostic files."
                        "e.g. 'combine.' for combine.1.scp.")
    parser.add_argument("--lang2weight", type=str,
                        help="comma-separated list of weights, one per language.")
# now the positional arguments
    parser.add_argument("egs_scp_lists", nargs='+',
                        help="list of egs.scp files per input language."
                           "e.g. exp/lang1/egs/egs.scp exp/lang2/egs/egs.scp")
    parser.add_argument("egs_dir",
                        help="Name of egs directory e.g. exp/tdnn_multilingual_sp/egs")


    print(sys.argv, file=sys.stderr)
    args = parser.parse_args()

    return args


def select_random_lang(lang_len, tot_egs, random_selection):
    """ Returns a random language index w.r.t
        amount of examples in each language.
        It works based on sampling from a
        discrete distribution, where it returns i
        with prob(i) = (num_egs in lang(i)/ tot_egs).
        tot_egs is sum of lang_len.
    """
    assert(tot_egs > 0)
    rand_int = random.randint(0, tot_egs - 1)
    count = 0
    for l in range(len(lang_len)):
        if random_selection:
            if  rand_int <= (count + lang_len[l]):
                return l
            else:
                count += lang_len[l]
        else:
            if (lang_len[l] > 0):
                return l
    return -1


def  process_multilingual_egs(args):
    args = get_args()
    random.seed(args.seed)
    rand_select = args.random_lang

    # read egs.scp for input languages
    scp_lists = args.egs_scp_lists
    num_langs = len(scp_lists)

    scp_files = [open(scp_lists[lang], 'r') for lang in range(num_langs)]

    lang2len = [0] * num_langs
    for lang in range(num_langs):
        lang2len[lang] = sum(1 for line in open(scp_lists[lang]))
        logger.info("Number of examples for language {0} "
                    "is {1}.".format(lang, lang2len[lang]))

    # If weights are not provided, the weights are 1.0.
    if args.lang2weight is None:
        lang2weight = [ 1.0 ] * num_langs
    else:
        lang2weight = args.lang2weight.split(",")
        assert(len(lang2weight) == num_langs)

    if not os.path.exists("{0}/temp".format(args.egs_dir)):
        os.makedirs("{0}/temp".format(args.egs_dir))
    num_lang_file = open("{0}/info/{1}num_tasks".format(args.egs_dir, args.prefix), "w")
    print("{0}".format(num_langs), file=num_lang_file)

    # Each element of all_egs (one per num_archive * num_jobs) is
    # an array of 3-tuples (lang-id, local-start-egs-line, num-egs)
    all_egs = []
    lang_len = lang2len[:]
    # total num of egs in all languages
    tot_num_egs = sum(lang2len[i] for i in range(len(lang2len)))
    num_archives = max(1, min(args.max_archives, tot_num_egs / args.samples_per_iter))

    num_arch_file = open("{0}/info/{1}num_archives".format(
                            args.egs_dir,
                            args.prefix),
                         "w")
    print("{0}".format(num_archives), file=num_arch_file)
    num_arch_file.close()
    this_num_egs_per_archive = tot_num_egs / (num_archives * args.num_jobs)

    logger.info("Generate temporary scp.job.archive_index files to store "
                "egs.scp correspond to each archive.")
    for job in range(args.num_jobs):
        for archive_index in range(num_archives):
            archfile = open("{0}/temp/{1}scp.{2}.{3}"
                            "".format(args.egs_dir, args.prefix,
                                      job + 1, archive_index + 1),
                            "w")
            this_egs = [] # this will be array of 2-tuples (lang-id start-frame num-frames)

            num_egs = 0
            while num_egs <= this_num_egs_per_archive:
                num_left_egs = sum(num_left_egs_per_lang for
                                   num_left_egs_per_lang in lang_len)
                if num_left_egs > 0:
                    lang_id = select_random_lang(lang_len, num_left_egs, rand_select)
                    start_egs = lang2len[lang_id] - lang_len[lang_id]
                    this_egs.append((lang_id, start_egs, args.minibatch_size))
                    for scpline in range(args.minibatch_size):
                        scp_key = scp_files[lang_id].readline().splitlines()[0]
                        print("{0} {1}".format(scp_key, lang_id),
                              file=archfile)

                    lang_len[lang_id] = lang_len[lang_id] - args.minibatch_size
                    num_egs = num_egs + args.minibatch_size
                    # If num of remaining egs in each lang is less than minibatch_size,
                    # they are discarded.
                    if lang_len[lang_id] < args.minibatch_size:
                        lang_len[lang_id] = 0
                        logger.info("Ran out of data for language {0}".format(
                            lang_id))
                else:
                    logger.info("Ran out of data for all languages.")
                    break
            all_egs.append(this_egs)
            archfile.close()

    logger.info("combining examples across all jobs correspond for each archive.")
    for archive in range(num_archives):
        logger.info("Combine {1}job.{0}.scp across all jobs into "
                    "{1}{0}.scp.".format(archive, args.prefix))
        this_ranges = []
        f = open("{0}/temp/{1}ranges.{2}.txt".format(
                    args.egs_dir, args.prefix, archive + 1),
                 'w')
        o = open("{0}/{1}output.{2}.ark".format(
                    args.egs_dir, args.prefix, archive + 1),
                 'w')
        w = open("{0}/{1}weight.{2}.ark".format(
                    args.egs_dir, args.prefix, archive + 1),
                 'w')
        scp_per_archive_file = open("{0}/{1}{2}.scp"
                                    "".format(args.egs_dir,
                                              args.prefix, archive + 1),
                                    'w')

        # check files before writing.
        if f is None:
            raise Exception("Error opening file {0}".format(f))
        if o is None:
            raise Exception("Error opening file {0}".format(o))
        if w is None:
            raise Exception("Error opening file {0}".format(w))
        if scp_per_archive_file is None:
            raise Exception("Error opening file {0}".format(scp_per_archive_file))

        for job in range(args.num_jobs):
            scp = ("{0}/temp/{1}scp.{2}.{3}".format(args.egs_dir, args.prefix,
                                                    job + 1, archive + 1))
            with open(scp, "r") as scpfile:
                for line in scpfile:
                    scp_line = line.splitlines()[0].split()
                    print("{0} {1}".format(scp_line[0], scp_line[1]),
                          file=scp_per_archive_file)
                    print("{0} output-{1}".format(scp_line[0], scp_line[2]),
                          file=o)
                    print("{0} {1}".format(
                            scp_line[0],
                            lang2weight[int(scp_line[2])]),
                          file=w)
            os.remove(scp)

        for (lang_id, start_eg_line, num_egs) in all_egs[num_archives * job + archive]:
            this_ranges.append((lang_id, start_eg_line, num_egs))

        # write ranges.archive
        for (lang_id, start_eg_line, num_egs) in this_ranges:
            print("{0} {1} {2}".format(lang_id, start_eg_line, num_egs), file=f)

        f.close()
        o.close()
        w.close()
        scp_per_archive_file.close()
    logger.info("allocate_multilingual_examples.py finished generating "
                "{0}*.scp, {0}ranges.*.txt, {0}output.*.ark "
                "and {0}weight.*.ark files".format(args.prefix))


def main():
    try:
        args = get_args()
        process_multilingual_egs(args)
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
  main()
