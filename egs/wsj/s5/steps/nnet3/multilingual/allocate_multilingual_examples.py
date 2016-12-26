#!/usr/bin/env python

# This script generates egs.Archive.scp and ranges.* used for generating egs.Archive.scp 
# for multilingual setup.
# Also this script generates outputs.*.scp and weight.*.scp, where each line 
# corresponds to language-id and weight for the same example in egs.*.scp. 
# weight.*.scp used to scale the output's posterior during training.
# ranges.*.scp is generated w.r.t frequency distribution of remaining examples 
# in each language. 
#
# You call this script as (e.g.)
#
# allocate_multilingual_examples.py [opts] num-of-languages example-scp-lists multilingual-egs-dir
#
# allocate_multilingual_examples.py --num-jobs 10 --samples-per-iter 10000 --minibatch-size 512
# --lang2weight exp/multi/lang2weight 2 "exp/lang1/egs.scp exp/lang2/egs.scp" 
# exp/multi/egs
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
import re, os, argparse, sys, math, warnings, random, io, imp
import numpy as np

nnet3_train_lib = imp.load_source('ntl', 'steps/nnet3/nnet3_train_lib.py')

def GetArgs():

  parser = argparse.ArgumentParser(description="Writes ranges.*, outputs.* and weights.* files "
                                   "in preparation for dumping egs for multilingual training.",
                                   epilog="Called by steps/nnet3/multilingual/get_egs.sh")
  parser.add_argument("--samples-per-iter", type=int, default=40000,
                      help="The target number of egs in each archive of egs, "
                      "(prior to merging egs). ");
  parser.add_argument("--num-jobs", type=int, default=20,
                      help="This can be used for better randomness in distributing languages across archives."
                      ", where egs.job.archive.scp generated randomly and examples are combined "
                      " across all jobs as eg.archive.scp.")
  parser.add_argument("--random-lang", type=str, action=nnet3_train_lib.StrToBoolAction, 
                      help="If true, the lang-id in ranges.* selected"
                      " w.r.t frequency distribution of remaining examples in each language,"
                      " otherwise it is selected sequentially.",
                      default=True, choices = ["false", "true"])
  parser.add_argument("--max-archives", type=int, default=1000,
                      help="max number of archives used to generate egs.*.scp");
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
  parser.add_argument("num_langs", type=int,
                      help="num of languages used in multilingual training setup.");
  parser.add_argument("egs_scp_lists", type=str,
                      help="list of egs.scp files per input language."
                           "e.g. exp/lang1/egs/egs.scp exp/lang2/egs/egs.scp");

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
def RandomLang(lang_len, tot_egs, random_selection):
  assert(tot_egs > 0)
  rand_int = random.randint(0, tot_egs - 1)
  count = 0
  for l in range(len(lang_len)):
    if random_selection:
      if rand_int > count and rand_int <= (count + lang_len[l]):
        rand_lang = l
        break
      else:
        count += lang_len[l]
    else: 
      if (lang_len[l] > 0):
        rand_lang = l
        break
  assert(rand_lang >= 0 and rand_lang < len(lang_len))
  return rand_lang

# Read lang2weight file and return lang2weight array
# where lang2weight[i] is weight for language i.
def ReadLang2weight(lang2w_file):
  f = open(lang2w_file, "r");
  if f is None:
    sys.exit("Error opening lang2weight file " + str(lang2w_file))
  lang2w = []
  for line in f:
    a = line.split()
    if len(a) != 2:
      sys.exit("bad line in lang2weight file " + line)
    lang2w.append(int(a[1]))
  f.close()
  return lang2w

# struct to keep archives correspond to each job
class ArchiveToJob():
  def __init__(self, job_id, archives_for_job):
    self.job_id = job_id
    self.archives = archives_for_job

def Main():
  args = GetArgs()
  random.seed(args.seed)
  num_langs = args.num_langs 
  rand_select = args.random_lang

  # read egs.scp for input languages
  scp_lists = args.egs_scp_lists.split();
  assert(len(scp_lists) == num_langs);
 
  scp_files = [open(scp_lists[lang], 'r') for lang in range(num_langs)]
  
  # computes lang2len, where lang2len[i] shows number of 
  # examples for language i.
  lang2len = [0] * num_langs
  for lang in range(num_langs):
    lang2len[lang] = sum(1 for line in open(scp_lists[lang]))
    print("num of examples for language {0} is {1}".format(lang, lang2len[lang]))

  # If weights are not provided, the scaling weights
  # are zero.
  if args.lang2weight is None:
    lang2weight = [ 1.0 ] * num_langs
  else:
    lang2weight = ReadLang2Len(args.lang2weight)
    assert(len(lang2weight) == num_langs)

  if not os.path.exists(args.egs_dir + "/temp"):
    os.makedirs(args.egs_dir + "/temp")

  num_lang_file = open(args.egs_dir + "/info/" + args.prefix + "num_lang", "w");
  print("{0}".format(num_langs), file = num_lang_file) 


  # Each element of all_egs (one per num_archive * num_jobs) is
  # an array of 3-tuples (lang-id, local-start-egs-line, num-egs)
  all_egs = []
  lang_len = lang2len[:]
  tot_num_egs = np.sum(lang2len) # total num of egs in all languages
  num_archives = max(1, min(args.max_archives, tot_num_egs / args.samples_per_iter))
  

  num_arch_file = open(args.egs_dir + "/info/" + args.prefix + "num_archives", "w");
  print("{0}".format(num_archives), file = num_arch_file)
  num_arch_file.close()

  this_num_egs_per_archive = tot_num_egs / (num_archives * args.num_jobs) # num of egs per archive
  for job_index in range(args.num_jobs):
    for archive_index in range(num_archives):
      # Temporary scp.job_index.archive_index files to store egs.scp correspond to each archive.
      print("Processing archive {0} for job {1}".format(archive_index + 1, job_index + 1))
      archfile = open(args.egs_dir + "/temp/" + args.prefix + "scp." + str(job_index + 1) + "." + str(archive_index + 1), "w")

      this_egs = [] # this will be array of 2-tuples (lang-id start-frame num-frames)
      
      num_egs = 0
      while num_egs <= this_num_egs_per_archive:
        rem_egs = np.sum(lang_len)
        if rem_egs > 0:
          lang_id = RandomLang(lang_len, rem_egs, rand_select)
          start_egs = lang2len[lang_id] - lang_len[lang_id]
          this_egs.append((lang_id, start_egs, args.minibatch_size))
          for scpline in range(args.minibatch_size):
            print("{0} {1}".format(scp_files[lang_id].readline().splitlines()[0], lang_id), file = archfile)

          lang_len[lang_id] = lang_len[lang_id] - args.minibatch_size
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
      archfile.close()

  # combine examples across all jobs correspond to each archive.
  for archive in range(num_archives):
    print("Processing archive {0} by combining all jobs.".format(archive + 1)) 
    this_ranges = []
    f = open(args.egs_dir + "/temp/" + args.prefix + "ranges." + str(archive + 1), "w")
    o = open(args.egs_dir + "/" + args.prefix + "output." + str(archive + 1), "w")
    w = open(args.egs_dir + "/" + args.prefix + "weight." + str(archive + 1), "w") 
    scp_per_archive_file = open(args.egs_dir + "/" + args.prefix + "egs." + str(archive + 1) + ".scp", "w")

    # check files befor writing.
    if f is None:
      sys.exit("Error opening file " + args.egs_dir + "/temp/" + args.prefix + "ranges." + str(job + 1))
    if o is None:
      sys.exit("Error opening file " + args.egs_dir + "/" + args.prefix + "output." + str(job + 1))
    if w is None:
      sys.exit("Error opening file " + args.egs_dir + "/" + args.prefix + "weight." + str(job + 1))
    if scp_per_archive_file is None:
      sys.exit("Error opening file " + args.egs_dir + "/" + args.prefix + "egs." + str(archive + 1) + ".scp", "w")

    for job in range(args.num_jobs):
      # combine egs.job.archive.scp across all jobs.
      scp = args.egs_dir + "/temp/" + args.prefix + "scp." + str(job + 1) + "." + str(archive + 1)
      with open(scp,"r") as scpfile:
        for line in scpfile:
          scp_line = line.splitlines()[0].split()
          print("{0} {1}".format(scp_line[0], scp_line[1]), file=scp_per_archive_file)
          print("{0} output-{1}".format(scp_line[0], scp_line[2]), file=o)
          print("{0} {1}".format(scp_line[0], lang2weight[int(scp_line[2])]), file=w) 
      os.remove(scp)

      # combine ranges.* across all jobs for archive
      for (lang_id, start_eg_line, num_egs) in all_egs[num_archives * job + archive]:
        this_ranges.append((lang_id, start_eg_line, num_egs))

    # write ranges.archive
    for (lang_id, start_eg_line, num_egs) in this_ranges:
      print("{0} {1} {2}".format(lang_id, start_eg_line, num_egs), file=f)
    
    scp_per_archive_file.close()
    f.close()
    o.close()
    w.close()
  print("allocate_multilingual_examples.py finished generating " + args.prefix + "egs.*.scp and " + args.prefix +  "ranges.* and " + args.prefix + "output.*" + args.prefix + "weight.* files")

if __name__ == "__main__":
  Main()
