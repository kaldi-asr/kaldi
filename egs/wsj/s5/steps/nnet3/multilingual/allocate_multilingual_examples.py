#!/usr/bin/env python

# This script generates ranges.* used for generating egs.JOB.scp for multilingual setup.
# Also this script generates outputs.*.scp and weight.*.scp, where each line 
# corresponds to language-id and weight for same line in egs.*.scp. 
# weight.*.scp used to scale the output's posterior during training.
# ranges.*.scp is generated w.r.t frequency distribution of remaining examples 
# in each language. 
#
# You call this script as (e.g.)
# 
# allocate_multilingual_examples.py --num-jobs 10 --num-archives 100 --minibatch-size 512
# --lang2weight exp/multi/lang2weight exp/multi/lang2len exp/multi/egs
#
# This script outputs specific ranges.* files to the temp directory (exp/multi/egs/temp)
# that will enable you to creat egs.*.scp files for multilingual training.
# exp/multi/egs/temp/ranges.* contains something like the following:
# e.g.
# lang1 0 0 256
# lang2 1 256 256
#
# where each line can be interpreted as follows:
# <source-language> <local-scp-line> <num-examples>
#
# note that <local-scp-line> is the zero-based line number in egs.scp for 
# that language.
# num-examples is multiple of actual minibatch-size.
#
#
# egs.1.scp is generated using ranges.1.scp as following:
# "num_examples" consecutive examples starting from line "local-scp-line" from 
# egs.scp file for language "source-lang" is copied to egs.1.scp.
#
#

from __future__ import print_function
import re, os, argparse, sys, math, warnings, random
import numpy as np

def GetArgs():

  parser = argparse.ArgumentParser(description="Writes ranges.*, outputs.* and weights.* files "
                                   "in preparation for dumping egs for multilingual training.",
                                   epilog="Called by steps/nnet3/multilingual/get_egs.sh")

  parser.add_argument("--num-archives", type=int, default=100,
                      help="Number of archives to write");
  parser.add_argument("--num-jobs", type=int, default=10,
                      help="Number of jobs we're going to use to write the archives; the ranges.* "
                                          "and outputs.* files are indexed by job.  Must be <= the --num-archives option.");
  parser.add_argument("--seed", type=int, default=1,
                      help="Seed for random number generator")

  parser.add_argument("--minibatch-size", type=int, default=512,
                      help="The minibatch size used to generate scp files per job. "
                           "It should be multiple of actual minibatch size.");
  
  parser.add_argument("--prefix", type=str, default="",
                      help="Adds a prefix to the range files. This is used to distinguish between the train "
                      "and diagnostic files.")

  parser.add_argument("--lang2weight", type=str, 
                      help="lang2weight file contains the weight per language to scale output posterior for that language.(format is: "
                           "<lang-id> <weight>)");
# now the positional arguments

  parser.add_argument("lang2len",
                      help="lang2len file is number of the examples per language to be used as input (format is: "
                           "<lang-id> <approx-num-examples>)");

  parser.add_argument("egs_dir",
                      help="Name of egs directory e.g. exp/multilingual_a/egs");

  print(' '.join(sys.argv))

  args = parser.parse_args()
 
  return args


# Returns a random language number w.r.t 
# amount of examples in each language.
# It works based on sampling from a 
# discrete distribution, where it returns i 
# with prob(i) as (num_egs in lang(i)/ tot_egs).
# tot_egs is sum of lang_len.
def RandomLang(lang_len, tot_egs):
  assert(tot_egs > 0)
  rand_int = random.randint(0, tot_egs - 1)
  count = 0
  for l in range(len(lang_len)):
    if rand_int > count and rand_int <= (count + lang_len[l]):
      rand_lang = l
      break
    else:
      count += lang_len[l]
  assert(rand_lang >= 0 and rand_lang < len(lang_len))
  return rand_lang

# Read lang2len file and return lang2len array
# where lang2len[i] is num_egs for language i.
def ReadLang2Len(lang2len_file):
  f = open(lang2len_file, "r");
  if f is None:
    sys.exit("Error opening lang2len file " + str(lang2len_file))
  lang_ids = []
  lengths = []
  lang2len = []
  for line in f:
    a = line.split()
    if len(a) != 2:
      sys.exit("bad line in lang2len file " + line)
    lang_ids.append(a[0])
    lengths.append(int(a[1]))
    lang2len.append(int(a[1]))
  f.close()
  return lang2len

# struct to keep archives correspond to each job
class ArchiveToJob():
  def __init__(self, job_id, archives_for_job):
    self.job_id = job_id
    self.archives = archives_for_job

def Main():
  args = GetArgs()
  random.seed(args.seed)
   
  lang2len = ReadLang2Len(args.lang2len)

  # If weights are not provided, the scaling weights
  # are zero.
  if args.lang2weight is None:
    lang2weight = [ 1.0 ] * len(lang2len)
  else:
    lang2weight = ReadLang2Len(args.lang2weight)
    assert(len(lang2weight) == len(lang2len))
  
  if not os.path.exists(args.egs_dir + "/temp"):
    os.makedirs(args.egs_dir + "/temp")
 
  # Each element of all_egs (one per archive) is
  # an array of 3-tuples (lang-id, local-start-egs-line, num-egs)
  all_egs = []
  lang_len = lang2len[:]
  tot_num_egs = np.sum(lang2len) # total num of egs in all languages
  this_num_egs_per_archive = tot_num_egs / args.num_archives # num of egs per archive
 
  for archive_index in range(args.num_archives):
    print("Processing job {0}".format(archive_index + 1))
    this_egs = [] # this will be array of 2-tuples (lang-id start-frame num-frames)
    #for n in range(this_num_egs_per_archive):
    num_egs = 0
    while num_egs <= this_num_egs_per_archive:
      rem_egs = np.sum(lang_len)
      if rem_egs > 0:
        lang_id = RandomLang(lang_len, rem_egs)
        start_egs = lang2len[lang_id] - lang_len[lang_id]
        this_egs.append((lang_id, start_egs, args.minibatch_size));
        lang_len[lang_id] = lang_len[lang_id] - args.minibatch_size;
        num_egs = num_egs + args.minibatch_size;
        # If the num of remaining egs in each lang is less than minibatch_size,
        # they are discarded.
        if lang_len[lang_id] < args.minibatch_size:
          lang_len[lang_id] = 0
          print("Run out of data for language {0}".format(lang_id))
      else:
        print("Run out of data for all languages.")
        break
    all_egs.append(this_egs)
  
  # work out archives we assign to each job
  num_archives_per_job = [ 0 ] * args.num_jobs
  this_archives_for_job = [ArchiveToJob(i, []) for i in range(args.num_jobs)]

  for i in range(0, args.num_archives-1):
    job_num = i % args.num_jobs
    num_archives_per_job[job_num] = num_archives_per_job[job_num] + 1
    this_archives_for_job[job_num].job_id = job_num
    this_archives_for_job[job_num].archives.append(i)

  cur_archive = 0
  for job in range(args.num_jobs):
    this_ranges = []
    for archive in this_archives_for_job[job].archives:
      for (lang_id, start_eg_line, num_egs) in all_egs[archive]:
        this_ranges.append((lang_id, start_eg_line, num_egs))
    f = open(args.egs_dir + "/temp/" + args.prefix + "ranges." + str(job + 1), "w")
    o = open(args.egs_dir + "/" + args.prefix + "output." + str(job + 1), "w")
    w = open(args.egs_dir + "/" + args.prefix + "weight." + str(job + 1), "w") 
    if f is None:
      sys.exit("Error opening file " + args.egs_dir + "/temp/" + args.prefix + "ranges." + str(job + 1))
    if o is None:
      sys.exit("Error opening file " + args.egs_dir + "/" + args.prefix + "output." + str(job + 1))
    if w is None:
      sys.exit("Error opening file " + args.egs_dir + "/" + args.prefix + "weight." + str(job + 1))

    for (lang_id, start_eg_line, num_egs) in this_ranges:
      print("{0} {1} {2}".format(lang_id, start_eg_line, num_egs), file=f)
      for l in range(num_egs):
        print("{0}".format(lang_id), file=o)
        print("{0}".format(lang2weight[lang_id]), file=w)
    f.close()
    o.close()
    w.close()
  print("allocate_multilingual_examples.py finished generating " + args.prefix +  "ranges.* and " + args.prefix + "output.*" + args.prefix + "weight.* files")

if __name__ == "__main__":
  Main()
