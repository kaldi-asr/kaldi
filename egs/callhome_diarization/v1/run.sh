#!/bin/bash
# Copyright 2016   TODO for Matthew when he edits this file
#           2016   David Snyder
# Apache 2.0.
#
# TODO details on what this does.
# See README.txt for more info on the required data.

. cmd.sh
. path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc

# Prepare a collection of NIST SRE data.
# TODO: This will probably be useful for UBM, ivector extractor training, and possibly, PLDA
#
local/make_sre.sh data

# Prepare SWB for UBM and i-vector extractor training.
# TODO: This is probably reasonable training data for the Callhome system, but it might also
#       be a good idea to try Fisher.
local/make_swbd2_phase2.pl /export/corpora5/LDC/LDC99S79 \
                           data/swbd2_phase2_train
local/make_swbd2_phase3.pl /export/corpora5/LDC/LDC2002S06 \
                           data/swbd2_phase3_train
local/make_swbd_cellular1.pl /export/corpora5/LDC/LDC2001S13 \
                             data/swbd_cellular1_train
local/make_swbd_cellular2.pl /export/corpora5/LDC/LDC2004S07 \
                             data/swbd_cellular2_train

# TODO create data prep script(s) for Callhome.

utils/combine_data.sh data/train \
  data/swbd_cellular1_train data/swbd_cellular2_train \
  data/swbd2_phase2_train data/swbd2_phase3_train data/sre

for name in sre train callhome; do
  steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
      data/$name exp/make_mfcc $mfccdir
  utils/fix_data_dir.sh data/$name
done

for name in sre train callhome; do
  sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
      data/$name exp/make_vad $vaddir
  utils/fix_data_dir.sh data/$name
done

# TODO the rest of the script...
