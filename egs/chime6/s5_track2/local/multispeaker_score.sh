#!/bin/bash
# Copyright   2019   Ashish Arora, Yusuke Fujita
# Apache 2.0.
# This script takes a reference and hypothesis text file, and performs 
# multispeaker scoring.

stage=0
cmd=queue.pl
num_spkrs=4
num_hyp_spk=4
datadir=dev_beamformit_dereverb
declare -a recording_id_array=("S02_U06" "S09_U06")
echo "$0 $@"  # Print the command line for logging
if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: $0 <ref-file> <hyp-file> <out-dir>"
  echo "e.g.: $0 data/diarized/text data/dev \
    exp/chain_train_worn_simu_u400k_cleaned_rvb/tdnn1b_sp/decode_dev_xvector_sad/scoring_kaldi/penalty_1.0/10.txt \
    exp/chain_train_worn_simu_u400k_cleaned_rvb/tdnn1b_sp/decode_dev_xvector_sad/scoring_kaldi_multispeaker"
  echo "Options: "
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  exit 1;
fi

ref_file=$1
hyp_file=$2
out_dir=$3

output_dir=$out_dir/per_speaker_output
wer_dir=$out_dir/per_speaker_wer

# For dev and evaluation set, we take corresopnding arrays
if [[ ${datadir} == *dev* ]]; then
  recording_id_array=("S02_U06" "S09_U06")
fi

if [[ ${datadir} == *eval* ]]; then
  recording_id_array=("S01_U06" "S21_U06")
fi

for f in $ref_file $hyp_file; do
  [ ! -f $f ] && echo "$0: No such file $f" && exit 1;
done

if [ $stage -le 0 ]; then
  echo "$0 generate per speaker per session file at paragraph level for the reference"
  echo "and per speaker per array file at paraghaph level for the hypothesis"
  mkdir -p $output_dir $wer_dir
  local/wer_output_filter < $ref_file > $output_dir/ref_filt.txt
  local/wer_output_filter < $hyp_file > $output_dir/hyp_filt.txt
  local/get_ref_perspeaker_persession_file.py $output_dir/ref_filt.txt $output_dir
  local/get_hyp_perspeaker_perarray_file.py $output_dir/hyp_filt.txt $output_dir
fi

if [ $stage -le 1 ]; then
  if [ $num_hyp_spk -le 3 ]; then
    echo "$0 create dummy per speaker per array hypothesis files for if the"
    echo " perdicted number of speakers by diarization is less than 4 "
    for recording_id in "${recording_id_array[@]}"; do
      for (( i=$num_hyp_spk+1; i<$num_spkrs+1; i++ )); do
        echo 'utt ' > ${dir}/hyp_${recording_id}_${i}_comb
      done
    done
  fi
fi

if [ $stage -le 2 ]; then
  echo "$0 calculate wer for each ref and hypothesis speaker"
  for recording_id in "${recording_id_array[@]}"; do
    for (( i=0; i<$((num_spkrs * num_spkrs)); i++ )); do
      ind_r=$((i / num_spkrs + 1))
      ind_h=$((i % num_spkrs + 1))
      sessionid="$(echo $recording_id | cut -d'_' -f1)"

      # compute WER with combined texts
      compute-wer --text --mode=present ark:${output_dir}/ref_${sessionid}_${ind_r}_comb \
        ark:${output_dir}/hyp_${recording_id}_${ind_h}_comb \
        > $wer_dir/wer_${recording_id}_r${ind_r}h${ind_h} 2>/dev/null
    done

    local/get_best_error.py $wer_dir $recording_id
  done
fi

if [ $stage -le 3 ]; then
  echo "$0 print best word error rate"
  echo "$0 it will print best wer for each recording and each array"
  cat $wer_dir/best_wer* > $wer_dir/all.txt
  cat $wer_dir/all.txt | local/print_dset_error.py $output_dir/recordinid_spkorder
fi

mkdir -p $wer_dir/wer_details $wer_dir/wer_details/log/
if [ $stage -le 4 ]; then
  echo "$0 generate per utterance wer details at utterance level"
  while read -r line;
  do
    recording_id=$(echo "$line" | cut -f1 -d ":")
    spkorder_str=$(echo "$line" | cut -f2 -d ":")
    sessionid=$(echo "$line" | cut -f1 -d "_")
    IFS='_' read -r -a spkorder_list <<< "$spkorder_str"
    IFS=" "
    ind_r=1
    for ind_h in "${spkorder_list[@]}"; do

      $cmd $wer_dir/wer_details/log/${recording_id}_r${ind_r}h${ind_h}_comb.log \
        align-text ark:${output_dir}/ref_${sessionid}_${ind_r}_comb ark:${output_dir}/hyp_${recording_id}_${ind_h}_comb ark:$output_dir/alignment_${sessionid}_r${ind_r}h${ind_h}.txt

      # split hypothesis texts along with reference utterances using word alignment of combined texts
      local/gen_aligned_hyp.py $output_dir/alignment_${sessionid}_r${ind_r}h${ind_h}.txt ${output_dir}/ref_wc_${sessionid}_${ind_r} > ${output_dir}/hyp_${recording_id}_r${ind_r}h${ind_h}_ref_segmentation

      ## compute per utterance alignments
      $cmd $wer_dir/wer_details/log/${recording_id}_r${ind_r}h${ind_h}_per_utt.log \
        cat ${output_dir}/hyp_${recording_id}_r${ind_r}h${ind_h}_ref_segmentation \| \
        align-text --special-symbol="'***'" ark:${output_dir}/ref_${sessionid}_${ind_r} ark:- ark,t:- \|  \
        utils/scoring/wer_per_utt_details.pl --special-symbol "'***'" \| tee $wer_dir/wer_details/per_utt_${recording_id}_r${ind_r}h${ind_h} || exit 1

      $cmd $wer_dir/wer_details/log/${recording_id}_r${ind_r}h${ind_h}_ops.log \
        cat $wer_dir/wer_details/per_utt_${recording_id}_r${ind_r}h${ind_h} \| \
        utils/scoring/wer_ops_details.pl --special-symbol "'***'" \| \
        sort -b -i -k 1,1 -k 4,4rn -k 2,2 -k 3,3 \> $wer_dir/wer_details/ops_${recording_id}_r${ind_r}h${ind_h} || exit 1;

      ind_r=$(( ind_r + 1 ))
    done
  done < $output_dir/recordinid_spkorder
  echo "$0 done generating per utterance wer details"
fi

echo "$0 done scoring"
