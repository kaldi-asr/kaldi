#!/usr/bin/env python3

# Copyright  2018  Johns Hopkins University (author: Daniel Povey)
# Copyright  2018  Hossein Hadian

# License: Apache 2.0.

import os
import argparse
import sys
import re
import logging
import traceback
import random

sys.path.insert(0, 'steps')

logger = logging.getLogger('libs')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(pathname)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('Starting choose_egs_to_merge.py')




def get_args():
    parser = argparse.ArgumentParser(description="Chooses groups of examples to merge into groups "
                                     "of size given by the --chunks-per-group option, based on speaker "
                                     "information (preferentially, chunks from the same utterance "
                                     "and, if possible, the same speaker, get combined into "
                                     "groups).  This script also computes a held-out subset of...",
                                     epilog="E.g. " + sys.argv[0] + "*** TODO *** ",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--random-seed', type=int,
                        default = 123, help='Random seed.')
    parser.add_argument("--chunks-per-group", type=int, default=4,
                        help="Number of chunks per speaker in the final egs (actually "
                        "means the number of chunks per group of chunks, and they are "
                        "only preferentially taken from the same speaker.")
    parser.add_argument("--num-repeats", type=int, default=1,
                        help="The number of times the data is to be repeated.  Must divide "
                        "--chunks-per-group.  Suggest to try only 1 or 2.  The idea "
                        "is to divide chunks into groups in different ways, to give "
                        "more variety to the egs (since the adaptation information "
                        "will differ.")
    parser.add_argument("--heldout-data-selection-proportion", type=float,
                        default=0.2,
                        help="This parameter governs the selection of the heldout "
                        "subset and the statistically matched training subset. "
                        "It does not affect the size of that subset, but only "
                        "affects what pool the examples are drawb from.  "
                        "Smaller values of this mean that the heldout groups "
                        "will be preferentially drawn from groups that "
                        "'contaminate' the least number of other groups, "
                        "and so require the least data to be removed from the "
                        "training set.  Setting this to 1.0 would mean that "
                        "the heldout subset is drawn completely at random "
                        "(which might be more wasteful of training data, but "
                        "gives a selection that's statistically more "
                        "representative).")
    parser.add_argument("--num-heldout-groups", type=int, default=200,
                        help="Number of utterance groups "
                        "that will go in the heldout subset (and in the "
                        "statistically matched training subset)")
    parser.add_argument("--utt2uniq", type=str, default='',
                        help="File used in setups with data "
                        "augmentation, that maps from utterance-ids to the "
                        "pre-augmentation utterance-id.  The reason it's needed "
                        "is to ensure that the heldout set is properly held "
                        "out (i.e., that different versions of those utterances "
                        "weren't trained on.  If not specified, we assume the "
                        "identity map.")
    parser.add_argument("--scp-in", type=str, required=True,
                        help="The scp file in, likely containing chain egs.  The "
                        "keys are expected to be of the form: "
                        "'<utterance_id>-<first_frame>-<left_context>-<num_frames>-<right_context>-v1', "
                        "where the left_context, num_frames and right_context are required to be the "
                        "same in order for keys to be in a group (note: it's best if the "
                        "--extra-left-context-initial and --extra-right-context-final options "
                        "are not used, and if the --frames-per-chunk is a single number, in "
                        "order to prevent this constraint from splitting up the utterances from "
                        "a single speaker")
    parser.add_argument("--training-data-out", type=str, required=True,
                        help="The output file containing the chunks that are to be grouped; each "
                        "line will contain --chunks-per-group (e.g. 4) rxfilenames, obtained "
                        "from the second field of the input --scp-in file.")
    parser.add_argument("--heldout-subset-out", type=str, required=True,
                        help="This is the name of the file to which the heldout data subset "
                        "will be written; the format is the same as --training-data-out.")
    parser.add_argument("--training-subset-out", type=str, required=True,
                        help="This is the name of the file to which the statistically matched "
                        "(to --heldout-subset-out) set of training data will be written")

    print(sys.argv, file=sys.stderr)
    args = parser.parse_args()

    return args


"""
Notes on plan for how to implement this (we can keep this as documentation, but
we'll maybe move some of it around when things get implemented).
This is a rather simple plan and we might later implement something more
sophisticated that does a better job of keeping chunks from the same utterance
or the same speaker together.
Basically we rely on the fact that the input utterances come in in sorted order
(so utterances from adjacent speakers will naturally be together.
We read the entries in the input scp file as a list, keeping them in the order
they were in the input (which will naturally keep together chunks from the
same utterance and utterances from the same speaker, since the raw egs were
not randomized).  We split that list into distinct sub-lists, each with a unique value
of <left_context>-<num_frames>-<right_context>.  In the normal case
there will be just one such sub-list.
In the case where --chunks-per-group=4 and --num-repeats=1, the groups of
chunks would then just be (and we do this for each of the sub-lists):
the first 4 chunks; the second 4 chunks; and so on.  In the case where
--chunks-per-group=4 and --num-repeats=2, we'd obtain the groups as above, then
we'd discard the first 2 chunks of each sub-list and repeat the process, giving
us twice the original number of groups.  If you want you can just
assert that --num-repeats is either 1 or 2 for now; higher values don't
really make sense with the current approach for choosing groups.
Once we have the groups as above, we need to figure out the subset of
size --num-heldout-groups which will be chosen to appear in the output
file --heldout-subset-out.  We'll also be choosing another subset of
the same size to appear in the file --training-subset-out; and we'll
be excluding some groups from the output --training-data-out (any
utterances that appeared in --heldout-subset-out, or which were linked
with such utterances via the --utt2uniq map, will be excluded).
The way we choose the groups to appear in --heldout-subset-out is as follows.
Firstly: in cases where the utt2uniq file is undefined, treat it as the identity
map.  We are given list of groups.  We compute, for each group, the set of
utterances represented in it, and from that, the set of "uniq" values (a "uniq"
value is a string, representing a pre-augmentation utterance-id).  For each
"uniq" value, we will compute the set of group-ids in which it was represented.
For a given group, we take the union of all those sets for its "uniq" value, and
remove its own group-id; this gives us the set of other groups that share a
pre-augmentation utterance in common with this group.  This set might be empty
only in the case where there was no augmentation and --num-repeats=1, and some
particular utterance had been split into exactly 4 chunks which all ended up in
the same group.
From the information above we can sort the groups by the number of groups we'd
have to hold out if we were to put that group in the heldout set.  Then if, say,
--heldout-data-selection-proportion=0.2, we take the bottom 20% of groups by
this measure, meaning the groups which will cause less training data to have to
be held out.  This is the set from which we'll select the heldout data and the
matched subset of training data.  Call this the "candidate set".  We first
choose --num-heldout-groups groups from the candidate set.  This is the heldout
subset.  From the heldout subset we compute the set of "uniq" values represented,
and we remove from the training set any groups which share those "uniq" values.
Next we need to choose the matched subset of training examples.  The way we do
this is that we choose --num-heldout-groups from the "candidate set", after
excluding groups that were in the heldout subset or which were removed from the
training set because they contained "uniq" values in common with those in the
heldout set.  If this fails because there were too few groups in the candidate
set, just double --heldout-data-selection-proportion and retry.  Make sure to do
something sensible in the case where the dataset is too tiny to choose the
requested heldout set size (i.e. print an informative error message before
dying).
"""

class Chunk:
    """ This is a data structure for a chunk. A chunk is a single entry
        of the --scp-in file.
        'eg'  second field of --scp-in file
    """
    def __init__(self, scp_line):
        result = re.match("^(.*)-(\d+)-(\d+)-(\d+)-(\d+)-v1\s+(.*)$", scp_line)
        self.utt_id, first_frame, left_context, num_frames, right_context, self.eg = result.groups()
        self.chunk_id = self.utt_id + '-' + first_frame
        self.context_structure = '-'.join((left_context, num_frames, right_context))
    def __repr__(self):
        return '{}-{} {}'.format(self.chunk_id, self.context_structure, self.eg)


def read_all_chunks(scp_file):
    """ Loads all the lines of the --scp-in file as chunk objects.
    """
    chunks = []
    with open(scp_file, 'r', encoding='latin-1') as f:
        for line in f:
            try:
                chunks.append(Chunk(line.strip()))
            except:
                logger.error('Bad line: ' + line.strip())
                raise
    return chunks

def load_utt2uniq(filename):
    """ Loads the --utt2uniq file as a dict.
    """
    utt2uniq = {}
    with open(filename, 'r', encoding='latin-1') as f:
        for line in f:
            uttid, base_uttid = line.strip().split()
            utt2uniq[uttid] = base_uttid
    return utt2uniq

def write_egs(filename, group_indexes, all_groups):
    """ Writes the output egs, i.e. the second field of
        the --scp-in file for specific chunks specified by `group_indexes`.
    """
    with open(filename, 'w', encoding='latin-1') as f:
        for group_index in group_indexes:
            for chunk in all_groups[group_index]:
                f.write('{}\n'.format(chunk.eg))



def choose_egs(args):
    """ The main part of the program.
    """
    random.seed(args.random_seed)
    logger.info('Set random seed to {}.'.format(args.random_seed))
    all_chunks = read_all_chunks(args.scp_in)
    logger.info('Loaded {} chunks.'.format(len(all_chunks)))

    chunk_to_sublist = {}
    for chunk in all_chunks:
        if chunk.context_structure not in chunk_to_sublist:
            chunk_to_sublist[chunk.context_structure] = [chunk]
        else:
            chunk_to_sublist[chunk.context_structure].append(chunk)

    logger.info('Created {} sub-lists with uniqe context '
                'structure.'.format(len(chunk_to_sublist)))


    assert(args.num_repeats == 1 or args.num_repeats == 2)
    groups = []  # All groups from all sub-lists
    for context_structure in sorted(chunk_to_sublist.keys()):
        sublist = chunk_to_sublist[context_structure]
        logger.info('Processing chunks with context '
                    'structure: {}'.format(context_structure))
        num_groups = (len(sublist) +
                      args.chunks_per_group - 1) // args.chunks_per_group
        for i in range(num_groups):
            group = sublist[i * args.chunks_per_group : (i + 1) * args.chunks_per_group]
            groups.append(group)
            if args.num_repeats == 2:
                shift = args.chunks_per_group // 2
                group = sublist[i * args.chunks_per_group + shift :
                                (i + 1) * args.chunks_per_group + shift]
                if group:
                    groups.append(group)

    logger.info('Created a total of {} groups.'.format(len(groups)))

    utt2uniq = {}
    if args.utt2uniq:
        utt2uniq = load_utt2uniq(args.utt2uniq)
        logger.info('Loaded utt2uniq file with {} entries.'.format(len(utt2uniq)))
    else:
        logger.info('--utt2uniq not specified; using identity map.')


    uniq_to_groups = {}  # uniq to set of groups that include it
    for i, group in enumerate(groups):
        for chunk in group:
            uniq = utt2uniq.get(chunk.utt_id, chunk.utt_id)
            if uniq not in uniq_to_groups:
                uniq_to_groups[uniq] = set([i])
            else:
                uniq_to_groups[uniq].add(i)

    logger.info('Computed uniq-to-groups for {} uniqs. Average number of '
                'groups representing a uniq is '
                '{}'.format(len(uniq_to_groups),
                            sum([len(g) for g in uniq_to_groups.values()]) /
                            len(uniq_to_groups)))

    # This is indexed by group-index (same len as groups). other_groups[i] is
    # the set of other groups which share some utterance with group i.
    other_groups = [set() for g in groups]
    for i, group in enumerate(groups):
        for chunk in group:
            uniq = utt2uniq.get(chunk.utt_id, chunk.utt_id)
            other_groups_this_uniq = uniq_to_groups[uniq]
            other_groups[i].update(other_groups_this_uniq)

    for i, other in enumerate(other_groups):  # Remove self
        other.remove(i)

    # 'group_shared_size' is a list of pairs (i, n) where i is group-index and
    # n is the number of groups that we'd
    # have to hold out if we were to put that group in the heldout set.
    group_shared_size = [(i, len(other)) for i, other in enumerate(other_groups)]
    # Sort it on n:
    group_shared_size.sort(key=lambda tup: tup[1])

    total_num_groups = len(groups)
    training_set = set(range(total_num_groups))  # All groups
    candidate_set_size = int(args.heldout_data_selection_proportion
                             * total_num_groups)
    logger.info('Initial candidate set size: {}'.format(candidate_set_size))
    if args.num_heldout_groups > candidate_set_size:
        logger.error('args.heldout_data_selection_proportion is too small or '
                     'there are too few groups.')
        sys.exit(1)

    candidate_set = set([tup[0] for tup in group_shared_size[:candidate_set_size]])
    heldout_list = random.sample(candidate_set, args.num_heldout_groups)


    # Remove all the heldout groups (and any other groups sharing some utterance
    # with them) from both the candidate set and the training set
    for group_index in heldout_list:
        for shared_group_index in other_groups[group_index]:
            candidate_set.discard(shared_group_index)
            training_set.discard(shared_group_index)
        candidate_set.discard(group_index)
        training_set.discard(group_index)

    logger.info('Candidate set size after removing heldout '
                'groups: {}'.format(len(candidate_set)))
    if args.num_heldout_groups > len(candidate_set):
        logger.warn('Not enough groups left in the candidate set. Doubling it.')
        candidate_set = set([tup[0] for tup in
                             group_shared_size[:candidate_set_size * 2]])
        for group_index in heldout_list:
            for shared_group_index in other_groups[group_index]:
                candidate_set.discard(shared_group_index)
            candidate_set.discard(group_index)
        logger.info('Candidate set size after doubling and removing heldout '
                    'groups: {}'.format(len(candidate_set)))
        if args.num_heldout_groups > len(candidate_set):
            logger.error('args.heldout_data_selection_proportion is too small '
                         'or there are too few groups. Not enough groups left.')
            sys.exit(1)

    train_subset_list = random.sample(candidate_set, args.num_heldout_groups)


    # Write the outputs:
    write_egs(args.training_data_out, training_set, groups)
    write_egs(args.heldout_subset_out, heldout_list, groups)
    write_egs(args.training_subset_out, train_subset_list, groups)


def main():
    try:
        args = get_args()
        choose_egs(args)
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()