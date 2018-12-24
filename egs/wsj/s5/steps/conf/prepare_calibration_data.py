#!/usr/bin/env python

# Copyright 2015  Brno University of Technology (author: Karel Vesely)
# Apache 2.0

from __future__ import division
import sys, math

from optparse import OptionParser
desc = """
Prepare input features and training targets for logistic regression,
which calibrates the Minimum Bayes Risk posterior confidences.

The logisitc-regression input features are: 
- posteriors from 'ctm' transformed by logit,
- logarithm of word-length in letters,
- 10base logarithm of unigram probability of a word from language model,
- logarithm of average lattice-depth at position of the word (optional),

The logistic-regresion targets are:
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
parser.add_option("--lattice-depth", help="Per-frame lattice depths, ascii-ark (optional). [default %default]", default='')
(o, args) = parser.parse_args()

if len(args) != 3:
  parser.print_help()
  sys.exit(1)
ctm_file, word_feats_file, word_categories_file = args

assert(o.conf_feats != '')

# Load the ctm (optionally add eval colmn with 'U'):
ctm = [ l.split() for l in open(ctm_file) ]
if len(ctm[0]) == 6: [ l.append('U') for l in ctm ]
assert(len(ctm[0]) == 7)

# Load the word-features, the format: "wrd wrd_id filter length other_feats"
# (typically 'other_feats' are unigram log-probabilities),
word_feats = [ l.split(None,4) for l in open(word_feats_file) ]

# Prepare filtering dict,
word_filter = { wrd_id:bool(int(filter)) for (wrd,wrd_id,filter,length,other_feats) in word_feats }
# Prepare the lenght dict,
word_length = { wrd_id:float(length) for (wrd,wrd_id,filter,length,other_feats) in word_feats }
# Prepare other_feats dict,
other_feats = { wrd_id:other_feats.strip() for (wrd,wrd_id,filter,length,other_feats) in word_feats }

# Build the targets,
if o.conf_targets != '':
  with open(o.conf_targets,'w') as f:
    for (utt, chan, beg, dur, wrd_id, conf, score_tag) in ctm:
      # Skip the words we don't know if being correct, 
      if score_tag == 'U': continue 
      # Some words are excluded from training (partial words, hesitations, etc.),
      # (Value: 1 == keep word, 0 == exclude word from the targets),
      if not word_filter[wrd_id]: continue 
      # Build the key,
      key = "%s^%s^%s^%s^%s,%s,%s" % (utt, chan, beg, dur, wrd_id, conf, score_tag)
      # Build the target,
      tgt = 1 if score_tag == 'C' else 0 # Correct = 1, else 0,
      # Write,
      f.write('%s %d\n' % (key,tgt))

# Load the per-frame lattice-depth,
# - we assume, the 1st column in 'ctm' is the 'utterance-key' in depth file,
# - if the 'ctm' and 'ark' keys don't match, we leave this feature out,
if o.lattice_depth:
  depths = dict()
  for l in open(o.lattice_depth):
    utt,d = l.split(' ',1)
    depths[utt] = [int(i) for i in d.split()]

# Load the 'word_categories' mapping for categorical input features derived from 'lang/words.txt',
wrd_to_cat = [ l.split() for l in open(word_categories_file) ]
wrd_to_cat = { wrd_id:int(category) for wrd,wrd_id,category in wrd_to_cat }
wrd_cat_num = max(wrd_to_cat.values()) + 1

# Build the input features,
with open(o.conf_feats,'w') as f:
  for (utt, chan, beg, dur, wrd_id, conf, score_tag) in ctm:
    # Build the key, same as previously,
    key = "%s^%s^%s^%s^%s,%s,%s" % (utt, chan, beg, dur, wrd_id, conf, score_tag)

    # Build input features,
    # - logit of MBR posterior,
    damper = 0.001 # avoid -inf,+inf from log,
    logit = math.log(float(conf)+damper) - math.log(1.0 - float(conf)+damper)
    # - log of word-length,
    log_word_length = math.log(word_length[wrd_id]) # i.e. number of phones in a word,
    # - categorical distribution of words (with frequency higher than min-count),
    wrd_1_of_k = [0]*wrd_cat_num; 
    wrd_1_of_k[wrd_to_cat[wrd_id]] = 1;

    # Compose the input feature vector,
    feats = [ logit, log_word_length, other_feats[wrd_id] ] + wrd_1_of_k

    # Optionally add average-depth of lattice at the word position,
    if o.lattice_depth != '':
      depth_slice = depths[utt][int(round(100.0*float(beg))):int(round(100.0*(float(beg)+float(dur))))]
      log_avg_depth = math.log(float(sum(depth_slice))/len(depth_slice))
      feats += [ log_avg_depth ]

    # Store the input features, 
    f.write(key + ' [ ' + ' '.join(map(str,feats)) + ' ]\n')

