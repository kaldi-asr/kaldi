#!/bin/bash

# Copyright 2015  Brno University of Technology (author: Karel Vesely)
# Apache 2.0

. cmd.sh
. path.sh

# Begin configuration section.
arpa_gz=data/local/lm/ami_fsh.o3g.kn.pr1-7.gz
LM=$(cat data/local/lm/final_lm).pr1-7
lmwt=12 # lmwt to select calibration targets,
lang=data/lang_ami_fsh.o3g.kn.pr1-7

dev=data/ihm/dev
dev_latdir=exp/ihm/tri4a_mmi_b0.1/decode_dev_4.mdl_${LM}
dev_ctm=$dev_latdir/ascore_${lmwt}/dev.ctm
dev_prf=$dev_latdir/ascore_${lmwt}/dev.ctm.filt.prf

eval=data/ihm/eval
eval_latdir=exp/ihm/tri4a_mmi_b0.1/decode_eval_4.mdl_${LM}
eval_ctm=$eval_latdir/ascore_${lmwt}/eval.ctm
# End configuration section.

. utils/parse_options.sh
set -euxo pipefail

# Main directory for the calibration,
caldir=${dev_latdir}/../calibration_${lmwt}
[ -d $caldir ] || mkdir -p $caldir

# Extract per-frame lattice-depth, 
# dev,
dev_latdepth=$dev_latdir/lattice_frame_depth.ark
[ -e $dev_latdepth ] || utils/lattice_depth_per_frame.sh --cmd "$train_cmd" $dev_latdir $dev_latdir
# eval,
eval_latdepth=$eval_latdir/lattice_frame_depth.ark
[ -e $eval_latdepth ] || utils/lattice_depth_per_frame.sh --cmd "$train_cmd" $eval_latdir $eval_latdir

###
### We train the calibration on dev,
###

# Add column to ctm, obtained from scoring alignment,
utils/scoring/append_prf_to_ctm.py $dev_prf $dev_ctm $caldir/dev.ctm

# Prepare training data for calibration model,
utils/scoring/prepare_calibration_data.py \
  --conf-targets $caldir/dev_train_targets.ark --conf-feats $caldir/dev_train_feats.ark \
  --segments $dev/segments --reco2file-and-channel $dev/reco2file_and_channel \
  $caldir/dev.ctm $dev_latdepth $arpa_gz $lang/words.txt

# Train the calibration model,
logistic-regression-train --binary=false ark:$caldir/dev_train_feats.ark \
  ark:$caldir/dev_train_targets.ark $caldir/calibration.mdl

# Apply calibration model to dev,
logistic-regression-eval --apply-log=false $caldir/calibration.mdl \
  ark:$caldir/dev_train_feats.ark ark,t:- | \
  awk '{ key=$1; p_corr=$4; sub(/,.*/,"",key); gsub(/\^/," ",key); print key,p_corr }' \
  >$caldir/dev.ctm.calibrated

###
### We apply the calibration to eval set,
###

# Prepare input data for calibration model,
utils/scoring/prepare_calibration_data.py --conf-feats $caldir/eval_feats.ark \
  --segments $eval/segments --reco2file-and-channel $eval/reco2file_and_channel \
  $eval_ctm $eval_latdepth $arpa_gz $lang/words.txt

# Apply calibration model to eval ctm,
logistic-regression-eval --apply-log=false $caldir/calibration.mdl \
  ark:$caldir/eval_feats.ark ark,t:- | \
  awk '{ key=$1; p_corr=$4; sub(/,.*/,"",key); gsub(/\^/," ",key); print key,p_corr }' \
  >$caldir/eval.ctm.calibrated

###
### Run scoring, 
###

hubscr=$KALDI_ROOT/tools/sctk/bin/hubscr.pl 
hubdir=`dirname $hubscr`
# dev,
$hubscr -p $hubdir -V -l english -h hub5 -g $dev/glm -r $dev/stm $caldir/dev.ctm.calibrated
# eval,
$hubscr -p $hubdir -V -l english -h hub5 -g $eval/glm -r $eval/stm $caldir/eval.ctm.calibrated

# Show the NCE improvement,
echo "# DEV #"
echo "Original NCE:"
egrep 'NCE|Sum' $dev_latdir/ascore_$lmwt/dev.ctm.filt.sys
echo "Calibrated NCE:"
egrep 'NCE|Sum' $caldir/dev.ctm.calibrated.filt.sys
echo
echo "# EVAL #"
echo "Original NCE:"
egrep 'NCE|Sum' $eval_latdir/ascore_$lmwt/eval.ctm.filt.sys
echo "Calibrated NCE:"
egrep 'NCE|Sum' $caldir/eval.ctm.calibrated.filt.sys

