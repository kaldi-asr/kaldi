#!/usr/bin/env bash
# Apache 2.0
#
# This script computes concatenated minimum-permutation WERs for
# the pipeline where ASR is performed before diarization. Since
# each utterance is assigned a single speaker and there is no
# lmwt tuning required, this script is simpler and faster than
# the scoring scripts for the "ASR after diarization" case. As
# input, it takes the "hyp_text" file which was generated from
# the CTM, "labels" file which is the output of the
# diarizer, and the "text" file (reference).

cmd=run.pl
stage=0

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

conditions="0L 0S OV10 OV20 OV30 OV40"

if [ $# -ne 4 ]; then
  echo "Usage: $0 [--cmd (run.pl|queue.pl...)]"
  echo "This script provides CHiME-6 style SA-WER scoring for CSS 2.0"
  echo "Options: "
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --stage (0 | int) # scoring stage."
  exit 1;
fi

data_dir=$1
hyp_file=$2
labels_file=$3
out_dir=$4

ref_file=$data_dir/text.bak
recording_ids=( $(awk '{$1=$1;sub(/_[0-9]*$/, "", $1); print $1}' $data_dir/wav.scp | sort -u) )
wer_dir=$out_dir/per_speaker_wer
name=`basename $data_dir`

mkdir -p $out_dir/per_speaker_output
mkdir -p  $wer_dir

if [ $stage -le 1 ]; then
  # First get per-speaker texts for reference and hypothesis and store
  # in output text files
  local/get_perspeaker_output_segmented.py $ref_file $hyp_file \
    $labels_file $out_dir/per_speaker_output
fi

if [ $stage -le 2 ]; then
  # Now for each recording, we score all pairs of ref/hyp speaker outputs
  for reco_id in "${recording_ids[@]}"; do
    # Get list of ref files
    reco_ref_files=( $( ls $out_dir/per_speaker_output/ref* | grep $reco_id ) )
    # Get list of hyp files
    reco_hyp_files=( $( ls $out_dir/per_speaker_output/hyp* | grep $reco_id ) )
    for reco_ref in "${reco_ref_files[@]}"; do
      for reco_hyp in "${reco_hyp_files[@]}"; do
        ref_spkid=$( basename "$reco_ref" | cut -d'_' -f2 )
        hyp_spkid=$( basename "$reco_hyp" | cut -d'_' -f2 )
        # compute WER with combined texts
        compute-wer --text --mode=present ark:$reco_ref ark:$reco_hyp \
          > $wer_dir/wer_${reco_id}_r${ref_spkid}h${hyp_spkid} 2>/dev/null
      done
    done
  done
fi

if [ $stage -le 3 ]; then
  for reco_id in "${recording_ids[@]}"; do
    # For each recording, we create a summary file of all permutations
    >$wer_dir/summary_$reco_id
    reco_wer_files=( $( ls $wer_dir/wer_* | grep $reco_id ) )
    for reco_wer in "${reco_wer_files[@]}"; do
      ref_hyp_spkid=$( basename "$reco_wer" | cut -d'_' -f5 )
      cur_wer=$( head -1 $reco_wer )
      printf "$ref_hyp_spkid %s\n" "${cur_wer}" >> $wer_dir/summary_$reco_id
    done

    # Now we get best wer for each recording id
    cat $wer_dir/summary_$reco_id \
      | local/best_wer_matching.py \
      > $wer_dir/best_wer_$reco_id
 
  done
  rm $wer_dir/best_wer_all 2> /dev/null
  awk '
  function basename(file, a, n) {
    n = split(file, a, "/")
    return a[n]
  }
  {printf "%s %s\n", basename(FILENAME), $0}' $wer_dir/best_wer_* > $wer_dir/best_wer_all
fi

if [ $stage -le 4 ]; then
  # Compute the average WER stats for all conditions individually.
  >$wer_dir/best_wer_cond
  for cond in $conditions; do
    grep $cond $wer_dir/best_wer_all | awk -v COND="$cond" '
      {
        ERR+=$5; WC+=$7; INS+=$8; DEL+=$10; SUB+=$12;
      }END{
        WER=ERR*100/WC;
        printf("%s %%WER %.2f [ %d / %d, %d ins, %d del, %d sub ]\n",COND,WER,ERR,WC,INS,DEL,SUB);
      }
      '
  done > $wer_dir/best_wer_cond

  # Compute overall WER average
  cat $wer_dir/best_wer_all | awk '
    {
      ERR+=$5; WC+=$7; INS+=$8; DEL+=$10; SUB+=$12;
    }END{
      WER=ERR*100/WC;
      printf("%%WER %.2f [ %d / %d, %d ins, %d del, %d sub ]",WER,ERR,WC,INS,DEL,SUB);
    }
    ' > $wer_dir/best_wer_average
fi

if [ $stage -le 5 ]; then
  echo "Cleaning up WER files..."
  find $wer_dir/ -maxdepth 1 -name "wer_*" -delete
fi

# printing final wer
echo "Condition-wise cpWERs for $name:" 
cat $wer_dir/best_wer_cond
echo "Average cpWER:" 
cat $wer_dir/best_wer_average

