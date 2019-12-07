#!/bin/bash
# Apache 2.0
#
# This script provides CHiME-6 challenge track 2 submission scores.
# It calculates the best search parameter configurations by using the dev set
# and provides wer for dev and eval

cmd=run.pl
stage=0
word_ins_penalty=0.0,0.5,1.0
min_lmwt=7
max_lmwt=17
dev_decodedir=exp/chain_train_worn_simu_u400k_cleaned_rvb/tdnn1b_sp/decode_dev_beamformit_dereverb_diarized_2stage
eval_decodedir=exp/chain_train_worn_simu_u400k_cleaned_rvb/tdnn1b_sp/decode_eval_beamformit_dereverb_diarized_2stage
dev_datadir=dev_beamformit_dereverb_diarized_hires
eval_datadir=eval_beamformit_dereverb_diarized_hires

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

if [ $# -ne 0 ]; then
    echo "Usage: $0 [--cmd (run.pl|queue.pl...)]"
    echo "This script provides CHiME-6 challenge submission scores"
    echo " Options:"
    echo "    --cmd (run.pl|queue.pl...)            # specify how to run the sub-processes."
    echo "    --dev_decodedir <dev-decode-dir>      # dev set decoding directory"
    echo "    --eval_decodedir <eval-decode-dir>    # eval set decoding directory"
    echo "    --dev_datadir <dev-data-dir>          # dev set data directory"
    echo "    --eval_datadir <eval-data-dir>        # eval set data directory"
    echo "    --min_lmwt <int>                      # minumum LM-weight for lattice rescoring "
    echo "    --max_lmwt <int>                      # maximum LM-weight for lattice rescoring "
    
    exit 1;
fi

if [ $stage -le 1 ]; then
  # obtaining multi speaker WER for all lmwt and wip
  for wip in $(echo $word_ins_penalty | sed 's/,/ /g'); do
    for LMWT in $(seq $min_lmwt $max_lmwt); do
      local/multispeaker_score.sh --cmd "$cmd" \
      --datadir $dev_datadir --get_stats false data/$dev_datadir/text \
      $dev_decodedir/scoring_kaldi/penalty_$wip/$LMWT.txt \
      $dev_decodedir/scoring_kaldi_multispeaker_$wip/$LMWT/
    done
  done
fi

if [ $stage -le 2 ]; then
  # obtaining best lmwt, wip and wer
  # adding /dev/null to the command list below forces grep to output the filename
  mkdir -p $dev_decodedir/scoring_kaldi_multispeaker
  grep WER $dev_decodedir/scoring_kaldi_multispeaker_*/*/per_speaker_wer/array_wer.txt /dev/null \
    | utils/best_wer.sh >& $dev_decodedir/scoring_kaldi_multispeaker/best_wer

  best_wer_file=$(awk '{print $NF}' $dev_decodedir/scoring_kaldi_multispeaker/best_wer)
  best_lmwt=$(echo $best_wer_file | awk -F/ '{N=NF-2; print $N}')
  best_wip=$(echo $best_wer_file | awk -F_ '{N=NF-3; print $N}' | awk -F/ '{N=NF-2; print $N}')
fi

echo "best LM weight: $best_lmwt"
echo "best insertion penalty weight: $best_wip"

if [ $stage -le 3 ]; then
  # obtaining per utterance stats for dev
  local/multispeaker_score.sh --cmd "$cmd" \
    --datadir $dev_datadir data/$dev_datadir/text \
    $dev_decodedir/scoring_kaldi/penalty_$best_wip/$best_lmwt.txt \
    $dev_decodedir/scoring_kaldi_multispeaker_$best_wip/$best_lmwt/
fi

if [ $stage -le 4 ]; then
  # obtaining per utterance stats for eval
  local/multispeaker_score.sh --cmd "$cmd" \
    --datadir $eval_datadir data/$eval_datadir/text \
    $eval_decodedir/scoring_kaldi/penalty_$best_wip/$best_lmwt.txt \
    $eval_decodedir/scoring_kaldi_multispeaker_$best_wip/$best_lmwt/
fi

if [ $stage -le 5 ]; then
  # storing best lmwt and wip and printing best wer for dev and eval
  echo $best_lmwt > $dev_decodedir/scoring_kaldi_multispeaker/lmwt
  echo $best_wip >  $dev_decodedir/scoring_kaldi_multispeaker/wip

  echo "$(<$dev_decodedir/scoring_kaldi_multispeaker_$best_wip/$best_lmwt/per_speaker_wer/array_wer.txt)"
  echo "$(<$eval_decodedir/scoring_kaldi_multispeaker_$best_wip/$best_lmwt/per_speaker_wer/array_wer.txt)"
fi

