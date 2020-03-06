#!/usr/bin/env bash
# Copyright 2015, Brno University of Technology (Author: Karel Vesely). Apache 2.0.

# Trains logistic regression, which calibrates the per-word confidences,
# which are extracted by the Minimum Bayes Risk decoding.

# begin configuration section.
cmd=
stage=0
# end configuration section.

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 5 ]; then
  echo "Usage: $0 [opts] <data-dir> <lang-dir|graph-dir> <decode-dir> <calibration-dir> <output-dir>"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  exit 1;
fi

set -euo pipefail

data=$1
lang=$2 # Note: may be graph directory not lang directory, but has the necessary stuff copied.
latdir=$3
caldir=$4
dir=$5

model=$latdir/../final.mdl # assume model one level up from decoding dir.
calibration=$caldir/calibration.mdl
word_feats=$caldir/word_feats
word_categories=$caldir/word_categories

for f in $lang/words.txt $word_feats $word_categories $latdir/lat.1.gz $calibration $model; do
  [ ! -f $f ] && echo "$0: Missing file $f" && exit 1
done
[ -z "$cmd" ] && echo "$0: Missing --cmd '...'" && exit 1

[ -d $dir/log ] || mkdir -p $dir/log
nj=$(cat $latdir/num_jobs)
lmwt=$(cat $caldir/lmwt)
decode_mbr=$(cat $caldir/decode_mbr)

# Store the setup,
echo $lmwt >$dir/lmwt
echo $decode_mbr >$dir/decode_mbr 
cp $calibration $dir/calibration.mdl
cp $word_feats $dir/word_feats
cp $word_categories $dir/word_categories

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
  for ((n=1; n<=nj; n++)); do cat $dir/${n}.ctm; done > $dir/ctm
  rm $dir/*.ctm
  cat $dir/ctm | utils/sym2int.pl -f 5 $lang/words.txt >$dir/ctm_int
fi

# Compute lattice-depth,
latdepth=$dir/lattice_frame_depth.ark
if [ $stage -le 1 ]; then
  [ -e $latdepth ] || steps/conf/lattice_depth_per_frame.sh --cmd "$cmd" $latdir $dir
fi

# Create the forwarding data for logistic regression,
if [ $stage -le 2 ]; then
  steps/conf/prepare_calibration_data.py --conf-feats $dir/forward_feats.ark \
    --lattice-depth $latdepth $dir/ctm_int $word_feats $word_categories
fi

# Apply calibration model to dev,
if [ $stage -le 3 ]; then
  logistic-regression-eval --apply-log=false $calibration \
    ark:$dir/forward_feats.ark ark,t:- | \
    awk '{ key=$1; p_corr=$4; sub(/,.*/,"",key); gsub(/\^/," ",key); print key,p_corr }' | \
    utils/int2sym.pl -f 5 $lang/words.txt \
    >$dir/ctm_calibrated
fi

exit 0
