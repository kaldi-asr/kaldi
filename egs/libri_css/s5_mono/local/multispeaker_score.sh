#!/usr/bin/env bash
# Copyright   2019   Ashish Arora, Yusuke Fujita
#             2020   Desh Raj
# Apache 2.0.
# This script takes a reference and hypothesis text file, and performs 
# multispeaker scoring.

stage=0
datadir=
get_stats=false # TODO: Implement 'true' (i.e. per utterance alignment of output)
multistream=false # Set to true if input audio was separated (e.g. CSS)

multistream_opt=
if [ $multistream == "true" ]; then
  multistream_opt="--multi-stream"
fi

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: $0 <ref-file> <hyp-file> <out-dir>"
  echo "e.g.: $0 data/diarized/text data/dev \
    exp/chain_cleaned/tdnn_1d_sp/decode_dev_diarized/scoring_kaldi/penalty_1.0/10.txt \
    exp/chain_cleaned/tdnn_1d_sp/decode_dev_diarized/scoring_kaldi_multispeaker"
  exit 1;
fi

ref_file=$1
hyp_file=$2
out_dir=$3

output_dir=$out_dir/per_speaker_output
wer_dir=$out_dir/per_speaker_wer

if [ $multistream ]; then
  recording_ids=( $(awk '{$1=$1;sub(/_[0-9]*$/, "", $1); print $1}' data/$datadir/wav.scp | sort -u) )
else
  recording_ids=( $(awk '{print $1}' data/$datadir/wav.scp) )
fi

for f in $ref_file $hyp_file; do
  [ ! -f $f ] && echo "$0: No such file $f" && exit 1;
done

if [ $stage -le 0 ]; then
  # generate per speaker per recording files for reference and hypothesis
  mkdir -p $output_dir $wer_dir
  local/wer_output_filter < $ref_file > $output_dir/ref_filt.txt
  local/wer_output_filter < $hyp_file > $output_dir/hyp_filt.txt
  local/get_perspeaker_output.py --affix "ref" $output_dir/ref_filt.txt data/$datadir/utt2spk.bak $output_dir
  local/get_perspeaker_output.py --affix "hyp" $multistream_opt $output_dir/hyp_filt.txt data/$datadir/utt2spk $output_dir
fi

if [ $stage -le 1 ]; then
  # Now for each recording, we score all pairs of ref/hyp speaker outputs
  for reco_id in "${recording_ids[@]}"; do
    # Get list of ref files
    reco_ref_files=( $( ls $output_dir/ref* | grep $reco_id ) )
    # Get list of hyp files
    reco_hyp_files=( $( ls $output_dir/hyp* | grep $reco_id ) )
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

if [ $stage -le 2 ]; then
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

# Also compute the average WER stats over all conditions. This will be used
# for LMWT and WIP selection.
if [ $stage -le 3 ]; then
  cat $wer_dir/best_wer_all | sed 's/,//g' | awk '
    {
      ERR+=$5; WC+=$7; INS+=$8; DEL+=$10; SUB+=$12;
    }END{
      WER=ERR*100/WC;
      printf("%%WER %.2f [ %d / %d, %d ins, %d del, %d sub ]",WER,ERR,WC,INS,DEL,SUB);
    }
    ' > $wer_dir/best_wer_average
fi
