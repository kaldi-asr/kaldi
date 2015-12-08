#!/bin/bash

# Copyright 2015  Brno University of Technology (author: Karel Vesely)
# Apache 2.0

. cmd.sh
. path.sh

# Begin configuration section. 
dev=data/eval2000
dev_latdir=exp/tri4/decode_eval2000_sw1_tg
lmwt=13 # lmwt to select calibration targets,
wpen=0.5
# End configuration section.

. utils/parse_options.sh
set -euxo pipefail

caldir=$dev_latdir/calibration_${lmwt}_${wpen}
[ -d $caldir ] || mkdir -p $caldir
# Extract per-frame lattice-depth, 
# dev,
dev_latdepth=$dev_latdir/lattice_frame_depth.ark
[ -e $dev_latdepth ] || utils/lattice_depth_per_frame.sh --cmd "$train_cmd" $dev_latdir $dev_latdir

# Train the calibration on dev,
# Add column to ctm, obtained from scoring alignment,
dev_ctm=$dev_latdir/score_${lmwt}_${wpen}/eval2000.ctm
dev_prf=$dev_latdir/score_${lmwt}_${wpen}/eval2000.ctm.filt.prf
utils/scoring/append_prf_to_ctm.py $dev_prf $dev_ctm $caldir/train.ctm

exit 0

# Prepare training data for calibration model,
utils/scoring/prepare_calibration_data.py \
  --conf-targets $caldir/train_targets.ark --conf-feats $caldir/train_feats.ark \
  --segments $dev/segments --wav-map $dev/wav.map \
  $caldir/train.ctm $dev_latdepth $arpa_gz $lang/words.txt

# Train the calibration model,
logistic-regression-train --binary=false ark:$caldir/train_feats.ark \
  ark:$caldir/train_targets.ark $caldir/calibration.mdl

# Apply calibration model to dev,
logistic-regression-eval --apply-log=false $caldir/calibration.mdl \
  ark:$caldir/train_feats.ark ark,t:- | \
  awk '{ key=$1; p_corr=$4; sub(/,.*/,"",key); gsub(/'\''/," ",key); print key,p_corr }' \
  >$caldir/train.ctm.calibrated

# Run scoring,
sclite="$KALDI_ROOT/tools/sctk/bin/sclite"
stm=data/conversational.dev.seg1/stm
$sclite -h $caldir/train.ctm.calibrated ctm -r $stm stm -o sum dtl prf lur -C hist bhist -D -F -e utf-8

# Apply the calibration to untranscribed set,
# Prepare input data for calibration model,
local/sst/ctm_to_logistic_regression_data.py --conf-feats $caldir/untran_feats.ark \
  ${untran_ctm} $untran_latdepth $arpa_gz $lang/words.txt

# Apply calibration model to untranscribed data,
logistic-regression-eval --apply-log=false $caldir/calibration.mdl \
  ark:$caldir/untran_feats.ark ark,t:- | \
  awk '{ key=$1; p_corr=$4; sub(/,.*/,"",key); gsub(/'\''/," ",key); print key,p_corr }' \
  >${untran_ctm}.calibrated

# Show NCE improvement,
echo "Original NCE:"
egrep 'NCE|Sum' $dev_latdir/scoring_lex_$lmwt/ctm.filt.sub.sys
echo "Calibratd NCE:"
egrep 'NCE|Sum' $caldir/train.ctm.calibrated.sys
# Show the output ctm of untranscribed set,
echo "Output: ${untran_ctm}.calibrated"
