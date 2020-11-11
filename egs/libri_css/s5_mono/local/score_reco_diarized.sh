#!/usr/bin/env bash
# Apache 2.0
#
# This script performs CHiME-6 track 2 style scoring for the diarized data.
# This means that all permutations of reference and hypothesis speakers are
# scored and the best one is selected to compute a kind of "speaker-attributed" WER.
# It calculates the best search parameter configurations by using the dev set
# and provides wer for dev and eval

cmd=run.pl
stage=0
word_ins_penalty=0.0,0.5,1.0
min_lmwt=7
max_lmwt=17
dev_decodedir=
eval_decodedir=
dev_datadir=
eval_datadir=
multistream=false # Set to true if input audio was separated (e.g. CSS)

echo "$0 $@"  # Print the command line for logging

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

conditions="0L 0S OV10 OV20 OV30 OV40"

if [ $# -ne 0 ]; then
    echo "Usage: $0 [--cmd (run.pl|queue.pl...)]"
    echo "This script provides CHiME-6 style SA-WER scoring for LibriCSS"
    echo " Options:"
    echo "    --cmd (run.pl|queue.pl...)            # specify how to run the sub-processes."
    echo "    --dev_decodedir <dev-decode-dir>      # dev set decoding directory"
    echo "    --eval_decodedir <eval-decode-dir>    # eval set decoding directory"
    echo "    --dev_datadir <dev-data-dir>          # dev set data directory"
    echo "    --eval_datadir <eval-data-dir>        # eval set data directory"
    echo "    --min_lmwt <int>                      # minumum LM-weight for lattice rescoring "
    echo "    --max_lmwt <int>                      # maximum LM-weight for lattice rescoring "
    echo "    --multistream <true|false>            # set to true if scoring multistream audio"
    
    exit 1;
fi

mkdir -p $dev_decodedir/scoring_kaldi_multispeaker

if [ $stage -le 1 ]; then
  # obtaining multi speaker WER for all lmwt and wip
  for wip in $(echo $word_ins_penalty | sed 's/,/ /g'); do
    $cmd LMWT=$min_lmwt:$max_lmwt \
      $dev_decodedir/scoring_kaldi_multispeaker/multispeaker_score.LMWT.log \
      local/multispeaker_score.sh --multistream $multistream \
      --datadir $dev_datadir --get_stats false data/$dev_datadir/text \
      $dev_decodedir/scoring_kaldi/penalty_$wip/$LMWT.txt \
      $dev_decodedir/scoring_kaldi_multispeaker/penalty_$wip/$LMWT
    done
  done
fi

if [ $stage -le 2 ]; then
  # obtaining best lmwt, wip and wer
  echo "Selecting best LM weight and WIP for condition $cond"
  grep WER $dev_decodedir/scoring_kaldi_multispeaker/penalty_*/*/per_speaker_wer/best_wer_average | \
      utils/best_wer.sh >& $dev_decodedir/scoring_kaldi_multispeaker/best_wer_average

  best_wer_file=$(awk '{print $NF}' $dev_decodedir/scoring_kaldi_multispeaker/best_wer_average)
  best_lmwt=$(echo $best_wer_file | cut -d'/' -f7)
  best_wip=$(echo $best_wer_file | cut -d'/' -f6 | cut -d'_' -f2)

  # printing and storing best lmwt, best_array and wip
  echo "best LM weight: $best_lmwt"
  echo "best insertion penalty weight: $best_wip"

  echo $best_lmwt > $dev_decodedir/scoring_kaldi_multispeaker/lmwt
  echo $best_wip >  $dev_decodedir/scoring_kaldi_multispeaker/wip
fi

if [ $stage -le 3 ]; then
  # Get WER for all conditions for the selected LMWT and WIP and remove other files
  best_lmwt="$(cat $dev_decodedir/scoring_kaldi_multispeaker/lmwt)"
  best_wip="$(cat $dev_decodedir/scoring_kaldi_multispeaker/wip)"
  cat $dev_decodedir/scoring_kaldi_multispeaker/penalty_$best_wip/$best_lmwt/per_speaker_wer/best_wer_all \
    > $dev_decodedir/scoring_kaldi_multispeaker/best_wer
  echo "Cleaning up WER files..."
  find $dev_decodedir/scoring_kaldi_multispeaker/penalty_*/*/per_speaker_wer -maxdepth 1 -name "wer_*" -delete

  # Compute overall WER average
  cat $dev_decodedir/scoring_kaldi_multispeaker/best_wer | awk '
    {
      ERR+=$5; WC+=$7; INS+=$8; DEL+=$10; SUB+=$12;
    }END{
      WER=ERR*100/WC;
      printf("%%WER %.2f [ %d / %d, %d ins, %d del, %d sub ]",WER,ERR,WC,INS,DEL,SUB);
    }
    ' > $dev_decodedir/scoring_kaldi_multispeaker/best_wer_average
fi

# Now scoring the eval set using best LMWT and WIP

if [ $stage -le 4 ]; then
  # obtaining per recording stats for eval
  best_lmwt="$(cat $dev_decodedir/scoring_kaldi_multispeaker/lmwt)"
  best_wip="$(cat $dev_decodedir/scoring_kaldi_multispeaker/wip)"
  local/multispeaker_score.sh \
    --multistream $multistream \
    --datadir $eval_datadir data/$eval_datadir/text \
    $eval_decodedir/scoring_kaldi/penalty_$best_wip/$best_lmwt.txt \
    $eval_decodedir/scoring_kaldi_multispeaker/penalty_$best_wip/$best_lmwt/
fi

if [ $stage -le 5 ]; then
  # obtaining eval wer corresponding to best lmwt, best_array and wip of dev
  best_lmwt="$(cat $dev_decodedir/scoring_kaldi_multispeaker/lmwt)"
  best_wip="$(cat $dev_decodedir/scoring_kaldi_multispeaker/wip)"

  find $eval_decodedir/scoring_kaldi_multispeaker/penalty_$best_wip/$best_lmwt/per_speaker_wer -maxdepth 1 -name "wer_*" -delete

  # Compute the average WER stats for all conditions individually.
  wer_dir=$eval_decodedir/scoring_kaldi_multispeaker/penalty_$best_wip/$best_lmwt/per_speaker_wer
  for cond in $conditions; do
    grep $cond $wer_dir/best_wer_all | awk -v COND="$cond" '
      {
        ERR+=$5; WC+=$7; INS+=$8; DEL+=$10; SUB+=$12;
      }END{
        WER=ERR*100/WC;
        printf("%s %%WER %.2f [ %d / %d, %d ins, %d del, %d sub ]\n",COND,WER,ERR,WC,INS,DEL,SUB);
      }
      '
  done > $eval_decodedir/scoring_kaldi_multispeaker/best_wer

  # Compute overall WER average
  cat $wer_dir/best_wer_all | awk '
    {
      ERR+=$5; WC+=$7; INS+=$8; DEL+=$10; SUB+=$12;
    }END{
      WER=ERR*100/WC;
      printf("%%WER %.2f [ %d / %d, %d ins, %d del, %d sub ]",WER,ERR,WC,INS,DEL,SUB);
    }
    ' > $eval_decodedir/scoring_kaldi_multispeaker/best_wer_average
fi

# printing dev and eval wer
echo "Dev WERs:" 
cat $dev_decodedir/scoring_kaldi_multispeaker/best_wer
echo "Eval WERs:" 
cat $eval_decodedir/scoring_kaldi_multispeaker/best_wer

