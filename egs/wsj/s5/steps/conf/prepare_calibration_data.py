#!/bin/env python

# Copyright 2015  Brno University of Technology (author: Karel Vesely)
# Apache 2.0

import sys, math

from optparse import OptionParser
desc = """
Prepare input features and training targets for logistic regression,
which calibrates the Minimum Bayes Risk posterior confidences.

The inputs are: 
- posteriors from 'ctm' transformed by logit,
- logarithm of word-length in letters,
- logarithm of average lattice-depth at position of the word,
- 10base logarithm of unigram probability of a word from language model,

The output is:
- 1 for correct word,
- 0 for incorrect word (substitution, insertion),

The iput 'ctm' is augmented by per-word tags (or 'U' is added if no tags),
'C' = correct
'S' = substitution
'I' = insertion
'U' = unknown (not part of scored segment)

The script can be used both to prepare the training data,
or to prepare input features for forwarding through trained model.
"""
usage = "%prog [opts] ctm word-filter word-length unigrams depth-per-frame-ascii.ark word-categories"
parser = OptionParser(usage=usage, description=desc)
parser.add_option("--conf-targets", help="Targets file for logistic regression (no targets generated if '') [default %default]", default='')
parser.add_option("--conf-feats", help="Feature file for logistic regression. [default %default]", default='')
(o, args) = parser.parse_args()

if len(args) != 6:
  parser.print_help()
  sys.exit(1)
ctm_file, word_filter_file, word_length_file, unigrams, depths_file, word_categories_file = args

assert(o.conf_feats != '')

# Load the ctm (optionally add eval colmn with 'U'):
ctm = [ l.split() for l in open(ctm_file) ]
if len(ctm[0]) == 6: [ l.append('U') for l in ctm ]
assert(len(ctm[0]) == 7)

# Load the word-filter,
word_filter = [ l.split() for l in open(word_filter_file) ]
word_filter = { wrd:bool(int(keep_word)) for wrd,id,keep_word in word_filter }

# Build the targets,
if o.conf_targets != '':
  with open(o.conf_targets,'w') as f:
    for (utt, chan, beg, dur, wrd, conf, score_tag) in ctm:
      # Skip the words we don't know if being correct, 
      if score_tag == 'U': continue 
      # Some words are excluded from training (partial words, hesitations, etc.),
      if not word_filter[wrd]: continue 
      # Build the key,
      key = "%s^%s^%s^%s^%s,%s,%s" % (utt, chan, beg, dur, wrd, conf, score_tag)
      # Build the target,
      tgt = 1 if score_tag == 'C' else 0 # Correct = 1, else 0,
      # Write,
      f.write('%s %d\n' % (key,tgt))

# Load the word-lengths,
word_length = [ l.split() for l in open(word_length_file) ]
word_length = { wrd:int(length) for wrd,id,length in word_length }

# Load the unigram probabilities in 10log (usually parsed from ARPA),
p_unigram_log10 = [ l.split() for l in open(unigrams) ]
p_unigram_log10 = { wrd:float(p_unigram) for wrd, p_unigram in p_unigram_log10 }

# Load the per-frame lattice-depth,
# - we assume, the 1st column in 'ctm' is the 'utterance-key' in depth file,
depths = dict()
for l in open(depths_file):
  utt,d = l.split(' ',1)
  depths[utt] = map(int,d.split())

# Load the 'word_categories' mapping for categorical input features derived from 'lang/words.txt',
wrd_to_cat = [ l.split() for l in open(word_categories_file) ]
wrd_to_cat = { wrd:int(category) for wrd,id,category in wrd_to_cat }
wrd_cat_num = max(wrd_to_cat.values()) + 1

# Build the input features,
with open(o.conf_feats,'w') as f:
  for (utt, chan, beg, dur, wrd, conf, score_tag) in ctm:
    # Build the key, same as previously,
    key = "%s^%s^%s^%s^%s,%s,%s" % (utt, chan, beg, dur, wrd, conf, score_tag)

    # Build input features,
    # - logit of MBR posterior,
    damper = 0.001 # avoid -inf,+inf from log,
    logit = math.log(float(conf)+damper) - math.log(1.0 - float(conf)+damper)
    # - log of word-length,
    log_lenwrd = math.log(word_length[wrd]) 
    # - log of frames per word-length,
    log_frame_per_letter = math.log(100.0*float(dur)/word_length[wrd])
    # - log of average-depth of lattice at the word position,
    depth_slice = depths[utt][int(round(100.0*float(beg))):int(round(100.0*(float(beg)+float(dur))))]
    log_avg_depth = math.log(float(sum(depth_slice))/len(depth_slice))
    # - categorical distribution of words with frequency higher than min-count,
    wrd_1_of_k = [0]*wrd_cat_num; 
    wrd_1_of_k[wrd_to_cat[wrd]] = 1;

    # Compose the input feature vector,
    feats = [ logit, log_lenwrd, log_frame_per_letter, p_unigram_log10[wrd], log_avg_depth ] + wrd_1_of_k
    # Store the input features, 
    f.write(key + ' [ ' + ' '.join(map(str,feats)) + ' ]\n')

