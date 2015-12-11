#!/bin/bash
# Copyright 2015, Brno University of Technology (Author: Karel Vesely). Apache 2.0.

# Trains logistic regression, which calibrates the per-word confidences,
# which are extracted by the Minimum Bayes Risk decoding.

# begin configuration section.
cmd=
lmwt=12
decode_mbr=true
stage=0
grep_filter=
# end configuration section.

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 5 ]; then
  echo "Usage: local/score_basic.sh [--cmd (run.pl|queue.pl...)] <data-dir> <lang-dir|graph-dir> <arpa-gz> <decode-dir> <calibration-dir>"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --lmwt <int>                    # scaling for confidence extraction"
  exit 1;
fi

set -euxo pipefail

data=$1
lang=$2 # Note: may be graph directory not lang directory, but has the necessary stuff copied.
arpa_gz=$3
latdir=$4
dir=$5

model=$latdir/../final.mdl # assume model one level up from decoding dir.

for f in $data/text $lang/words.txt $latdir/lat.1.gz; do
  [ ! -f $f ] && echo "$0: Missing file $f" && exit 1
done
[ -z "$cmd" ] && echo "$0: Missing --cmd '...'" && exit 1

[ -d $dir/log ] || mkdir -p $dir/log
nj=$(cat $latdir/num_jobs)

# Store the setup,
echo $lmwt >$dir/lmwt
echo $decode_mbr >$dir/decode_mbr 

# Create the ctm with raw confidences,
# - we keep the timing relative to the utterance,
if [ $stage -le 0 ]; then
  $cmd JOB=1:$nj $dir/log/get_ctm.JOB.log \
    lattice-scale --inv-acoustic-scale=$lmwt "ark:gunzip -c $latdir/lat.JOB.gz|" ark:- \| \
    lattice-limit-depth ark:- ark:- \| \
    lattice-push --push-strings=false ark:- ark:- \| \
    lattice-align-words-lexicon --max-expand=10.0 \
     $lang/phones/align_lexicon.int $model ark:- ark:- \| \
    lattice-to-ctm-conf --decode-mbr=$decode_mbr ark:- - \| \
    utils/int2sym.pl -f 5 $lang/words.txt \
    '>' $dir/JOB.ctm
  # Merge and clean,
  set +x; for ((n=1; n<=nj; n++)); do cat $dir/${n}.ctm; done > $dir/ctm
  rm $dir/*.ctm; set -x
fi

# Get evaluation of the 'ctm' using the 'text' reference,
if [ $stage -le 1 ]; then
  utils/scoring/ctm_to_tra.py $dir/ctm - | \
  align-text --special-symbol="<eps>" ark:$data/text ark:- ark,t:- | \
  utils/scoring/wer_per_utt_details.pl --special-symbol "<eps>" \
  >$dir/align_text 
  # Append alignment to ctm,
  utils/scoring/append_eval_to_ctm.py $dir/align_text $dir/ctm $dir/ctm_aligned
fi

# Compute lattice-depth,
latdepth=$dir/lattice_frame_depth.ark
if [ $stage -le 2 ]; then
  [ -e $latdepth ] || utils/lattice_depth_per_frame.sh --cmd "$cmd" $latdir $dir
fi

# Create the training data for logistic regression,
if [ $stage -le 3 ]; then
  utils/scoring/prepare_calibration_data.py \
    --conf-targets $dir/train_targets.ark --conf-feats $dir/train_feats.ark \
    $dir/ctm_aligned $latdepth $arpa_gz $lang/words.txt
fi

# Filter the targets,
if [ $stage -le 4 ]; then
  # TODO: Filter out from ctm tokens [...], <...>, %..., ???
  grep -i -v -E "\^(\[.*\]|<.*>|%[^,]*|[^,]*-|$grep_filter)," $dir/train_targets.ark >$dir/train_targets_filt.ark
fi

# Train the logistic regression,
if [ $stage -le 5 ]; then
  logistic-regression-train --binary=false ark:$dir/train_feats.ark \
    ark:$dir/train_targets_filt.ark $dir/calibration.mdl 2>$dir/log/logistic-regression-train.log
fi

# Apply calibration model to dev,
if [ $stage -le 6 ]; then
  logistic-regression-eval --apply-log=false $dir/calibration.mdl \
    ark:$dir/train_feats.ark ark,t:- | \
    awk '{ key=$1; p_corr=$4; sub(/,.*/,"",key); gsub(/\^/," ",key); print key,p_corr }' \
    >$dir/ctm_calibrated
fi

exit 0
