#!/bin/env python

# Copyright 2015  Brno University of Technology (author: Karel Vesely)
# Apache 2.0

import sys
import numpy as np

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
usage = "%prog [opts] ctm segments depth-per-frame-ascii.ark arpa-lm.gz words.txt logreg_inputs"
parser = OptionParser(usage=usage, description=desc)
parser.add_option("--segments", help="Mapping of sentence-relative times to absolute times, for lattice-depths. Use for ctm with absolute times. [default %default]", default='')
parser.add_option("--reco2file-and-channel", help="Mapping of recording to file and channel in 'stm' [default %default]", default='')
parser.add_option("--conf-targets", help="Targets file for logistic regression (no targets generated if '') [default %default]", default='')
parser.add_option("--conf-feats", help="Feature file for logistic regression. [default %default]", default='')
(o, args) = parser.parse_args()

if len(args) != 4:
  parser.print_help()
  sys.exit(1)
ctm_file, depths_file, arpa_gz, words_file = args

assert(o.conf_feats != '')

# Load the ctm,
try:
  ctm = np.loadtxt(ctm_file,dtype='object,object,f8,f8,object,f8,object')
except IndexError:
  # Missing 'score_tag' column, add 'U' everywhere,
  ctm = np.loadtxt(ctm_file,dtype='object,object,f8,f8,object,f8')
  ctm = [(f, chan, beg, dur, wrd, conf, 'U') for (f, chan, beg, dur, wrd, conf) in ctm]
  ctm = np.array(ctm,dtype='object,object,f8,f8,object,f8,object')

# Build the targets,
if o.conf_targets != '':
  targets = []
  for (f, chan, beg, dur, wrd, conf, score_tag) in ctm:
    # Skip words we don't know if being correct, 
    if score_tag == 'U': continue 
    # Build the key,
    key = "%s^%s^%.2f^%.2f^%s,%.2f,%s" % (f, chan, beg, dur, wrd, conf, score_tag)
    # Build the target,
    tgt = 1 if score_tag == 'C' else 0 # Correct = 1, else 0,
    # Add to list,
    targets.append((key,tgt))
  # Store the targets,
  np.savetxt(o.conf_targets, np.array(targets,dtype='object,i'), fmt='%s %d')

# Segments dicionart is indexed by utterance,
if o.segments != '':
  segments = { utt:(reco,beg,end) for (utt,reco,beg,end) in np.loadtxt(o.segments,dtype='object,object,f8,f8') }
  # Optionally apply 'reco2file_and_channel' mapping,
  if o.reco2file_and_channel != '':
    reco2file_and_channel = { reco:(file,chan) for (reco,file,chan) in np.loadtxt(o.reco2file_and_channel,dtype='object,object,object') }
  else:
    # Or trivial self-mapping using 2nd column in 'segments', (default channel is '1'),
    reco2file_and_channel = { reco:(reco,'1') for reco in np.unique(np.array(segments.values())['f0']) } 

### Load the per-frame lattice-depth,
depths = dict()
if o.segments == '':
  # The depths are stored normally,
  with open(depths_file,'r') as f:
    for l in f:
      utt,d = l.split(' ',1)
      depths[(utt,'1')] = np.array(d.split(),dtype='int')
else:
  # The depths from 'segments' get pasted into 'long' vectors covering complete unsegmented audio,
  # Getting last segment for each 'reco',
  reco_beg_end_sort = np.sort(np.array(segments.values(),dtype='object,f8,f8'), order=['f0','f2'])
  reco_beg_end_lastseg = reco_beg_end_sort[np.append(reco_beg_end_sort['f0'][1:] != reco_beg_end_sort['f0'][:-1],[True])]
  # Create buffer with zero depths,
  for (reco,beg,end) in reco_beg_end_lastseg:
    frame_total = 1 + int(np.rint(100*end))
    depths[reco2file_and_channel[reco]] = np.zeros(frame_total, dtype='int')
  # Load the depths (ASCII), fill the buffer with the depths,
  with open(depths_file,'r') as f:
    for l in f:
      utt,d = l.split(' ',1)
      d = np.array(d.split(),dtype='int')
      frame_begin = int(np.rint(100*segments[utt][1]))
      depths[reco2file_and_channel[segments[utt][0]]][frame_begin:frame_begin+len(d)] = d

# Load the unigram probabilities in 10log from ARPA,
import gzip, re
arpa_log10_unigram = dict()
with gzip.open(arpa_gz,'r') as f:
  read = False
  for l in f:
    if l.strip() == '\\1-grams:': read = True
    if l.strip() == '\\2-grams:': break
    if read and len(l.split())>=2:
      log10_p_unigram, wrd = re.split('[\t ]+',l.strip(),2)[:2]
      arpa_log10_unigram[wrd] = log10_p_unigram

# Load the 'words.txt' table for categorical distribution of words on input,
wrd_to_int = { wrd:i for wrd,i in np.loadtxt(words_file,dtype='object,i4') }
wrd_num = np.max(np.array(wrd_to_int.values(),dtype='int')) + 1

# Build the input features,
with open(o.conf_feats,'w') as inputs:
  for (f, chan, beg, dur, wrd, conf, score_tag) in ctm:
    # Build the key, same as previous,
    key = "%s^%s^%.2f^%.2f^%s,%.2f,%s" % (f, chan, beg, dur, wrd, conf, score_tag)
    # Build input features,
    # - logit of MBR posterior, 
    damper = 0.001 # avoid -inf,+inf from log,
    logit = np.log(conf+damper) - np.log(1.0 - conf+damper)
    # - log of word-length in characters,
    log_lenwrd = np.log(len(wrd)) 
    # - log of average-depth of lattice,
    log_avg_depth = np.log(np.mean(depths[(f,chan)][int(np.rint(100.0*beg)):int(np.rint(100.0*(beg+dur)))]))
    # - log of frame-per-letter ratio,
    log_frame_per_letter = np.log(100.0*dur/len(wrd))
    
    # - categorical distribution of words (DISABLED!!!, needs a lot of memory, but gave 0.015 NCE improvement),
    # wrd_1_of_k = [0]*wrd_num; 
    # wrd_1_of_k[wrd_to_int[wrd]] = 1;

    # Compose the input feature vector,
    feats = [ logit, log_lenwrd, log_avg_depth, arpa_log10_unigram[wrd], log_frame_per_letter ] # + wrd_1_of_k
    # Store the input features, 
    inputs.write(key + ' [ ' + ' '.join(map(str,feats)) + ' ]\n')

