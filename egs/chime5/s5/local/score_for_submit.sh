#!/usr/bin/env bash
# Copyright 2012-2014  Johns Hopkins University (Author: Daniel Povey, Yenda Trmal)
# Apache 2.0
#
# This script provides official CHiME-5 challenge submission scores per room and session.
# It first calculates the best search parameter configurations by using the dev set
# and also create the transcriptions for dev and eval sets to be submitted.
# The default setup does not calculate scores of the evaluation set since
# the evaluation transcription is not distributed (July 9 2018)

cmd=run.pl
dev=exp/chain_train_worn_u100k_cleaned/tdnn1a_sp/decode_dev_beamformit_ref
eval=exp/chain_train_worn_u100k_cleaned/tdnn1a_sp/decode_eval_beamformit_ref
do_eval=false

echo "$0 $@"  # Print the command line for logging
[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 0 ]; then
    echo "Usage: $0 [--cmd (run.pl|queue.pl...)]"
    echo "This script provides official CHiME-5 challenge submission scores"
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
# get the scoring result per utterance
score_result=$dev/scoring_kaldi/wer_details/per_utt
for session in S02 S09; do
    for room in DINING KITCHEN LIVING; do
	# get nerror
	nerr=`grep "\#csid" $score_result | grep $room | grep $session | awk '{sum+=$4+$5+$6} END {print sum}'`
	# get nwords from references (NF-2 means to exclude utterance id and " ref ")
	nwrd=`grep "\#csid" $score_result | grep $room | grep $session | awk '{sum+=$3+$4+$6} END {print sum}'`
	# compute wer with scale=2
	wer=`echo "scale=2; 100 * $nerr / $nwrd" | bc`
	
	# report the results
	echo -n "session $session "
	echo -n "room $room: "
	echo -n "#words $nwrd, "
	echo -n "#errors $nerr, "
	echo "wer $wer %"
    done
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
for session in S01 S21; do
    for room in DINING KITCHEN LIVING; do
	if $do_eval; then
	    # get nerror
	    nerr=`grep "\#csid" $score_result | grep $room | grep $session | awk '{sum+=$4+$5+$6} END {print sum}'`
	    # get nwords from references (NF-2 means to exclude utterance id and " ref ")
	    nwrd=`grep "\#csid" $score_result | grep $room | grep $session | awk '{sum+=$3+$4+$6} END {print sum}'`
	    # compute wer with scale=2
	    wer=`echo "scale=2; 100 * $nerr / $nwrd" | bc`
	
	    # report the results
	    echo -n "session $session "
	    echo -n "room $room: "
	    echo -n "#words $nwrd, "
	    echo -n "#errors $nerr, "
	    echo "wer $wer %"
	fi
    done
done
if $do_eval; then
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
else
    echo "skip evaluation scoring"
    echo ""
    echo "==== when you submit your result to the CHiME-5 challenge ===="
    echo "Please rename your recognition results of "
    echo "$dev/scoring_kaldi/penalty_$best_wip/$best_lmwt.txt"
    echo "$eval/scoring_kaldi/penalty_$best_wip/$best_lmwt.txt"
    echo "with {dev,eval}_<last name>_<affiliation>.txt, e.g., dev_watanabe_jhu.txt and eval_watanabe_jhu.txt, "
    echo "and submit both of them as your final challenge result"
    echo "=================================================================="    
fi

