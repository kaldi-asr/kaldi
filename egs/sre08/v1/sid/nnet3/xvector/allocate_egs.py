#!/usr/bin/env python3

# Copyright      2017 Johns Hopkins University (Author: Daniel Povey)
#                2017 Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2017 David Snyder
# Apache 2.0

# This script, which is used in getting training examples, decides
# which examples will come from which recordings, and at what point
# during the training.

# You call it as (e.g.)
#
#  allocate_egs.py --min-frames-per-chunk=50 --max-frames-per-chunk=200 \
#   --frames-per-iter=1000000 --num-repeats=60 --num-archives=169 \
#   --num-jobs=24  exp/xvector_a/egs/temp/utt2len.train exp/xvector_a/egs
#
# The program outputs certain things to the temp directory (e.g.,
# exp/xvector_a/egs/temp) that will enable you to dump the chunks for xvector
# training.  What we'll eventually be doing is invoking the following program
# with something like the following args:
#
#  nnet3-xvector-get-egs [options] exp/xvector_a/temp/ranges.1 \
#    scp:data/train/feats.scp ark:exp/xvector_a/egs/egs_temp.1.ark \
#    ark:exp/xvector_a/egs/egs_temp.2.ark ark:exp/xvector_a/egs/egs_temp.3.ark
#
# where exp/xvector_a/temp/ranges.1 contains something like the following:
#
#   utt1  0  1  0   65 0
#   utt1  6  7  160 50 0
#   utt2  ...
#
# where each line is interpreted as follows:
#  <source-utterance> <relative-archive-index> <absolute-archive-index> \
#    <start-frame-index> <num-frames> <spkr-label>
#
# Note: <relative-archive-index> is the zero-based offset of the archive-index
# within the subset of archives that a particular ranges file corresponds to;
# and <absolute-archive-index> is the 1-based numeric index of the destination
# archive among the entire list of archives, which will form part of the
# archive's filename (e.g. egs/egs.<absolute-archive-index>.ark);
# <absolute-archive-index> is only kept for debug purposes so you can see which
# archive each line corresponds to.
#
# For each line of the ranges file, we specify an eg containing a chunk of data
# from a given utterane, the corresponding speaker label, and the output
# archive.  The list of archives corresponding to ranges.n will be written to
# output.n, so in exp/xvector_a/temp/outputs.1 we'd have:
#
#  ark:exp/xvector_a/egs/egs_temp.1.ark ark:exp/xvector_a/egs/egs_temp.2.ark \
#    ark:exp/xvector_a/egs/egs_temp.3.ark
#
# The number of these files will equal 'num-jobs'.  If you add up the
# word-counts of all the outputs.* files you'll get 'num-archives'.  The number
# of frames in each archive will be about the --frames-per-iter.
#
# This program will also output to the temp directory a file called
# archive_chunk_length which tells you the frame-length associated with
# each archive, e.g.,
# 1   60
# 2   120
# the format is:  <archive-index> <num-frames>.  The <num-frames> will always
# be in the range [min-frames-per-chunk, max-frames-per-chunk].


# We're using python 3.x style print but want it to work in python 2.x.
from __future__ import print_function
from __future__ import division
import re, os, argparse, sys, math, warnings, random

def get_args():
    parser = argparse.ArgumentParser(description="Writes ranges.*, outputs.* and archive_chunk_lengths files "
                                 "in preparation for dumping egs for xvector training.",
                                 epilog="Called by sid/nnet3/xvector/get_egs.sh")
    parser.add_argument("--prefix", type=str, default="",
                   help="Adds a prefix to the output files. This is used to distinguish between the train "
                   "and diagnostic files.")
    parser.add_argument("--num-repeats", type=int, default=10, help="Number of times each speaker repeats within an archive.")
    parser.add_argument("--min-frames-per-chunk", type=int, default=50,
                    help="Minimum number of frames-per-chunk used for any archive")
    parser.add_argument("--max-frames-per-chunk", type=int, default=300,
                    help="Maximum number of frames-per-chunk used for any archive")
    parser.add_argument("--randomize-chunk-length", type=str,
                    help="If true, randomly pick a chunk length in [min-frames-per-chunk, max-frames-per-chunk]."
                    "If false, the chunk length varies from min-frames-per-chunk to max-frames-per-chunk"
                    "according to a geometric sequence.",
                    default="true", choices = ["false", "true"])
    parser.add_argument("--frames-per-iter", type=int, default=1000000,
                    help="Target number of frames for each archive")
    parser.add_argument("--num-archives", type=int, default=-1,
                    help="Number of archives to write");
    parser.add_argument("--num-jobs", type=int, default=-1,
                    help="Number of jobs we're going to use to write the archives; the ranges.* "
                    "and outputs.* files are indexed by job.  Must be <= the --num-archives option.");
    parser.add_argument("--seed", type=int, default=123,
                    help="Seed for random number generator")
    parser.add_argument("--num-pdfs", type=int, default=-1,
                    help="Num pdfs")

    # now the positional arguments
    parser.add_argument("--utt2len-filename", type=str, required=True,
                    help="utt2len file of the features to be used as input (format is: "
                    "<utterance-id> <num-frames>)");
    parser.add_argument("--utt2int-filename", type=str, required=True,
                    help="utt2int file of the features to be used as input (format is: "
                    "<utterance-id> <id>)");
    parser.add_argument("--egs-dir", type=str, required=True,
                    help="Name of egs directory, e.g. exp/xvector_a/egs");

    print(' '.join(sys.argv), file=sys.stderr)
    print(sys.argv, file=sys.stderr)
    args = parser.parse_args()
    args = process_args(args)
    return args

def process_args(args):
    if args.num_repeats < 1:
        raise Exception("--num-repeats should have a minimum value of 1")
    if not os.path.exists(args.utt2int_filename):
        raise Exception("This script expects --utt2int-filename to exist")
    if not os.path.exists(args.utt2len_filename):
        raise Exception("This script expects --utt2len-filename to exist")
    if args.min_frames_per_chunk <= 1:
        raise Exception("--min-frames-per-chunk is invalid.")
    if args.max_frames_per_chunk < args.min_frames_per_chunk:
        raise Exception("--max-frames-per-chunk is invalid.")
    if args.frames_per_iter < 1000:
        raise Exception("--frames-per-iter is invalid.")
    if args.num_archives < 1:
        raise Exception("--num-archives is invalid")
    if args.num_jobs > args.num_archives:
        raise Exception("--num-jobs is invalid (must not exceed num-archives)")
    return args

# Create utt2len
def get_utt2len(utt2len_filename):
    utt2len = {}
    f = open(utt2len_filename, "r")
    if f is None:
        sys.exit("Error opening utt2len file " + str(utt2len_filename))
    utt_ids = []
    lengths = []
    for line in f:
        tokens = line.split()
        if len(tokens) != 2:
            sys.exit("bad line in utt2len file " + line)
        utt2len[tokens[0]] = int(tokens[1])
    f.close()
    return utt2len
    # Done utt2len

# Handle utt2int, create spk2utt, spks
def get_labels(utt2int_filename):
    f = open(utt2int_filename, "r")
    if f is None:
        sys.exit("Error opening utt2int file " + str(utt2int_filename))
    spk2utt = {}
    utt2spk = {}
    for line in f:
        tokens = line.split()
        if len(tokens) != 2:
            sys.exit("bad line in utt2int file " + line)
        spk = int(tokens[1])
        utt = tokens[0]
        utt2spk[utt] = spk
        if spk not in spk2utt:
            spk2utt[spk] = [utt]
        else:
            spk2utt[spk].append(utt)
    spks = spk2utt.keys()
    f.close()
    return spks, spk2utt, utt2spk
    # Done utt2int


# this function returns a random integer utterance index, limited to utterances
# above a minimum length in frames, with probability proportional to its length.
def get_random_utt(spkr, spk2utt):
    this_utts = spk2utt[spkr]
    this_num_utts = len(this_utts)
    i = random.randint(0, this_num_utts-1)
    utt = this_utts[i]
    return utt

def random_chunk_length(min_frames_per_chunk, max_frames_per_chunk):
    ans = random.randint(min_frames_per_chunk, max_frames_per_chunk)
    return ans

# This function returns an integer in the range
# [min-frames-per-chunk, max-frames-per-chunk] according to a geometric
# sequence. For example, suppose min-frames-per-chunk is 50,
# max-frames-per-chunk is 200, and args.num_archives is 3. Then the
# lengths for archives 0, 1, and 2 will be 50, 100, and 200.
def deterministic_chunk_length(archive_id, num_archives, min_frames_per_chunk, max_frames_per_chunk):
  if max_frames_per_chunk == min_frames_per_chunk:
    return max_frames_per_chunk
  elif num_archives == 1:
    return int(max_frames_per_chunk);
  else:
    return int(math.pow(float(max_frames_per_chunk)/
                     min_frames_per_chunk, float(archive_id) /
                     (num_archives-1)) * min_frames_per_chunk + 0.5)



# given an utterance length utt_length (in frames) and two desired chunk lengths
# (length1 and length2) whose sum is <= utt_length,
# this function randomly picks the starting points of the chunks for you.
# the chunks may appear randomly in either order.
def get_random_offset(utt_length, length):
    if length > utt_length:
        sys.exit("code error: length > utt-length")
    free_length = utt_length - length

    offset = random.randint(0, free_length)
    return offset


def main():
    args = get_args()
    if not os.path.exists(args.egs_dir + "/temp"):
        os.makedirs(args.egs_dir + "/temp")
    random.seed(args.seed)
    utt2len = get_utt2len(args.utt2len_filename)
    spks, spk2utt, utt2spk = get_labels(args.utt2int_filename)
    if args.num_pdfs == -1:
        args.num_pdfs = max(spks) + 1

    # archive_chunk_lengths is an mapping from archive id to the number of
    # frames in examples of that archive.
    archive_chunk_lengths = []
    # all_egs contains 2-tuples of the form (utt-id, offset)
    all_egs= []

    prefix = ""
    if args.prefix != "":
        prefix = args.prefix + "_"

    info_f = open(args.egs_dir + "/temp/" + prefix + "archive_chunk_lengths", "w")
    if info_f is None:
        sys.exit(str("Error opening file {0}/temp/" + prefix + "archive_chunk_lengths").format(args.egs_dir));
    for archive_index in range(args.num_archives):
        print("Processing archive {0}".format(archive_index + 1))
        if args.randomize_chunk_length == "true":
            # don't constrain the lengths to be the same
            length = random_chunk_length(args.min_frames_per_chunk, args.max_frames_per_chunk)
        else:
            length = deterministic_chunk_length(archive_index, args.num_archives, args.min_frames_per_chunk, args.max_frames_per_chunk);
        print("{0} {1}".format(archive_index + 1, length), file=info_f)
        archive_chunk_lengths.append(length)
        this_num_egs = int(float(args.frames_per_iter) / length + 1)
        this_egs = [ ] # A 2-tuple of the form (utt-id, start-frame)
        spkrs = args.num_repeats * list(spk2utt.keys())
        random.shuffle(spkrs)
        for n in range(this_num_egs):
            if len(spkrs) == 0:
                print("Ran out of speakers for archive {0}".format(archive_index + 1))
                break
            spkr = spkrs.pop()
            utt_len = 0
            while utt_len < length:
                utt = get_random_utt(spkr, spk2utt)
                utt_len = utt2len[utt]
            offset = get_random_offset(utt_len, length)
            this_egs.append( (utt, offset) )
        all_egs.append(this_egs)
    info_f.close()

    # work out how many archives we assign to each job in an equitable way.
    num_archives_per_job = [ 0 ] * args.num_jobs
    for i in range(0, args.num_archives):
        num_archives_per_job[i % args.num_jobs]  = num_archives_per_job[i % args.num_jobs] + 1

    pdf2num = {}
    cur_archive = 0
    for job in range(args.num_jobs):
        this_ranges = []
        this_archives_for_job = []
        this_num_archives = num_archives_per_job[job]

        for i in range(0, this_num_archives):
            this_archives_for_job.append(cur_archive)
            for (utterance_index, offset) in all_egs[cur_archive]:
                this_ranges.append( (utterance_index, i, offset) )
            cur_archive = cur_archive + 1

        f = open(args.egs_dir + "/temp/" + prefix + "ranges." + str(job + 1), "w")
        if f is None:
            sys.exit("Error opening file " + args.egs_dir + "/temp/" + prefix + "ranges." + str(job + 1))
        for (utterance_index, i, offset) in sorted(this_ranges):
            archive_index = this_archives_for_job[i]
            print("{0} {1} {2} {3} {4} {5}".format(utterance_index,
                                           i,
                                           archive_index + 1,
                                           offset,
                                           archive_chunk_lengths[archive_index],
                                           utt2spk[utterance_index]),
              file=f)
            if utt2spk[utterance_index] in pdf2num:
                 pdf2num[utt2spk[utterance_index]] += 1
            else:
                pdf2num[utt2spk[utterance_index]] = 1
        f.close()


        f = open(args.egs_dir + "/temp/" + prefix + "outputs." + str(job + 1), "w")
        if f is None:
            sys.exit("Error opening file " + args.egs_dir + "/temp/" + prefix + "outputs." + str(job + 1))
        print( " ".join([ str("{0}/" + prefix + "egs_temp.{1}.ark").format(args.egs_dir, n + 1) for n in this_archives_for_job ]),
           file=f)
        f.close()

    f = open(args.egs_dir + "/" + prefix + "pdf2num", "w")
    nums = []
    for k in range(0, args.num_pdfs):
        if k in pdf2num:
          nums.append(pdf2num[k])
        else:
          nums.append(0)

    print(" ".join(map(str, nums)), file=f)
    f.close()

    print("allocate_egs.py: finished generating " + prefix + "ranges.* and " + prefix + "outputs.* files")

if __name__ == "__main__":
    main()

