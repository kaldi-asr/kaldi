#!/usr/bin/env bash
# Copyright 2012-2014  Johns Hopkins University (Author: Daniel Povey, Yenda Trmal)
# Copyright 2019       Johns Hopkins University (Author: Shinji Watanabe)
# Apache 2.0
#
# This script scores the multi-speaker LibriCSS recordings.
# It first calculates the best search parameter configurations by using the dev set
# and then uses these to score both sets.

cmd=run.pl
dev=exp/chain_cleaned/tdnn_1d_sp/decode_dev
eval=exp/chain_cleaned/tdnn_1d_sp/decode_eval

conditions="0L 0S OV10 OV20 OV30 OV40"

echo "$0 $@"  # Print the command line for logging
[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 0 ]; then
    echo "Usage: $0 [--cmd (run.pl|queue.pl...)]"
    echo "This script scores the LibriCSS full recordings"
    echo " Options:"
    echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
    echo "    --dev <dev-decode-dir>          # dev set decoding directory"
    echo "    --eval <eval-decode-dir>        # eval set decoding directory"
    exit 1;
fi

# get language model weight and word insertion penalty from the dev set
best_lmwt=`cat $dev/scoring_kaldi/wer_details/lmwt`
best_wip=`cat $dev/scoring_kaldi/wer_details/wip`

echo "best LM weight: $best_lmwt"
echo "insertion penalty weight: $best_wip"

echo "==== development set ===="
# development set
# we report scores by overlap type, i.e., 0L, 0S, OV10, and so on.

# get the scores per utterance
score_result=$dev/scoring_kaldi/wer_details/per_utt

for cond in $conditions; do
  # get nerror
  nerr=`grep "\#csid" $score_result | grep $cond | awk '{sum+=$4+$5+$6} END {print sum}'`
  # get nwords from references (NF-2 means to exclude utterance id and " ref ")
  nwrd=`grep "\#csid" $score_result | grep $cond | awk '{sum+=$3+$4+$6} END {print sum}'`
  # compute wer with scale=2
  wer=`echo "scale=2; 100 * $nerr / $nwrd" | bc`  
  # report the results
  echo -n "Condition $cond: "
  echo -n "#words $nwrd, "
  echo -n "#errors $nerr, "
  echo "wer $wer %"
done

echo -n "overall: "
# get nerror
nerr=`grep "\#csid" $score_result | awk '{sum+=$4+$5+$6} END {print sum}'`
# get nwords from references (NF-2 means to exclude utterance id and " ref ")
nwrd=`grep "\#csid" $score_result | awk '{sum+=$3+$4+$6} END {print sum}'`
# compute wer with scale=2
wer=`echo "scale=2; 100 * $nerr / $nwrd" | bc`
echo -n "#words $nwrd, "
echo -n "#errors $nerr, "
echo "wer $wer %"

echo "==== evaluation set ===="
# evaluation set
# get the scoring result per utterance. Copied from local/score.sh
mkdir -p $eval/scoring_kaldi/wer_details_devbest
$cmd $eval/scoring_kaldi/log/stats1.log \
     cat $eval/scoring_kaldi/penalty_$best_wip/$best_lmwt.txt \| \
     align-text --special-symbol="'***'" ark:$eval/scoring_kaldi/test_filt.txt ark:- ark,t:- \|  \
     utils/scoring/wer_per_utt_details.pl --special-symbol "'***'" \> $eval/scoring_kaldi/wer_details_devbest/per_utt

score_result=$eval/scoring_kaldi/wer_details_devbest/per_utt

for cond in $conditions; do
    # get nerror
    nerr=`grep "\#csid" $score_result | grep $cond | awk '{sum+=$4+$5+$6} END {print sum}'`
    # get nwords from references (NF-2 means to exclude utterance id and " ref ")
    nwrd=`grep "\#csid" $score_result | grep $cond | awk '{sum+=$3+$4+$6} END {print sum}'`
    # compute wer with scale=2
    wer=`echo "scale=2; 100 * $nerr / $nwrd" | bc`

    # report the results
    echo -n "Condition $cond: "
    echo -n "#words $nwrd, "
    echo -n "#errors $nerr, "
    echo "wer $wer %"
done

echo -n "overall: "
# get nerror
nerr=`grep "\#csid" $score_result | awk '{sum+=$4+$5+$6} END {print sum}'`
# get nwords from references (NF-2 means to exclude utterance id and " ref ")
nwrd=`grep "\#csid" $score_result | awk '{sum+=$3+$4+$6} END {print sum}'`
# compute wer with scale=2
wer=`echo "scale=2; 100 * $nerr / $nwrd" | bc`
echo -n "overall: "
echo -n "#words $nwrd, "
echo -n "#errors $nerr, "
echo "wer $wer %"


