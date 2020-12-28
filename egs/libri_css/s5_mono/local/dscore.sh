#!/bin/bash
# Copyright  2020   Desh Raj

# Apache 2.0.
#
# This script installs a fork of the dscore toolkit 
# (https://github.com/nryant/dscore), which also supports
# evaluating the overlapping regions only. It then scores
# the output sys_rttm based on the provided ref_rttm.

score_overlaps_only=true

echo "$0 $@"  # Print the command line for logging

set -e

. ./path.sh
. parse_options.sh || exit 1;

if [ $# != 2 ]; then
  echo "Usage: $0 <ref-rttm> <hyp-rttm>"
  echo "e.g.: $0 data/test/rttm exp/test_diarization/rttm"
  exit 1;
fi

ref_rttm=$1
hyp_rttm=$2
  
if ! [ -d dscore ]; then
  git clone https://github.com/desh2608/dscore.git -b libricss --single-branch
  cd dscore
  python3 -m pip install --user -r requirements.txt
  cd ..
fi

# Create per condition ref and hyp RTTM files for scoring per condition
mkdir -p tmp
trap "rm -r tmp" EXIT

conditions="0L 0S OV10 OV20 OV30 OV40"
cp $ref_rttm tmp/ref.all
cp $hyp_rttm tmp/hyp.all
for rttm in ref hyp; do
  for cond in $conditions; do
    cat tmp/$rttm.all | grep $cond > tmp/$rttm.$cond
  done
done

echo "Scoring all regions..."
for cond in $conditions 'all'; do
  echo -n "Condition: $cond: "
  ref_rttm_path=$(readlink -f tmp/ref.$cond)
  hyp_rttm_path=$(readlink -f tmp/hyp.$cond)
  cd dscore
  python3 score.py -r $ref_rttm_path -s $hyp_rttm_path --global_only
  cd ..
done

# We also score overlapping regions only
if [ $score_overlaps_only == "true" ]; then
  echo "Scoring overlapping regions..."
  for cond in $conditions 'all'; do
    echo -n "Condition: $cond: "
    ref_rttm_path=$(readlink -f tmp/ref.$cond)
    hyp_rttm_path=$(readlink -f tmp/hyp.$cond)
    cd dscore
    python3 score.py -r $ref_rttm_path -s $hyp_rttm_path --overlap_only --global_only
    cd ..
  done
fi
