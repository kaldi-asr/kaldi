#!/usr/bin/env bash
# Copyright   2019   Ashish Arora, Vimal Manohar
# Apache 2.0.
# This script takes an rttm file, and performs decoding on on a test directory.
# The output directory contains a text file which can be used for scoring.


stage=0
nj=8
cmd=queue.pl
echo "$0 $@"  # Print the command line for logging
if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;
if [ $# != 6 ]; then
  echo "Usage: $0 <rttm-dir> <in-data-dir> <lang-dir> <model-dir> <ivector-dir> <out-dir>"
  echo "e.g.: $0 data/rttm data/dev data/lang_chain exp/chain_train_worn_simu_u400k_cleaned_rvb \
                 exp/nnet3_train_worn_simu_u400k_cleaned_rvb data/dev_diarized"
  echo "Options: "
  echo "  --nj <nj>                                        # number of parallel jobs."
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  exit 1;
fi

rttm_dir=$1
data_in=$2
lang_dir=$3
asr_model_dir=$4
ivector_extractor=$5
out_dir=$6

for f in $rttm_dir/rttm $data_in/wav.scp $data_in/text.bak \
         $lang_dir/L.fst $asr_model_dir/tree_sp/graph/HCLG.fst \
         $asr_model_dir/tdnn1b_sp/final.mdl; do
  [ ! -f $f ] && echo "$0: No such file $f" && exit 1;
done

if [ $stage -le 0 ]; then
  echo "$0 copying data files in output directory"
  cp $rttm_dir/rttm $rttm_dir/rttm_1
  sed -i 's/'.ENH'/''/g' $rttm_dir/rttm_1
  # removing participant introduction from the hypothesis rttm
  # UEM file contains the scoring durations for each recording
  local/truncate_rttm.py $rttm_dir/rttm_1 local/uem_file $rttm_dir/rttm_introduction_removed
  mkdir -p ${out_dir}_hires
  cp ${data_in}/{wav.scp,utt2spk} ${out_dir}_hires
  utils/data/get_reco2dur.sh ${out_dir}_hires
fi

if [ $stage -le 1 ]; then
  echo "$0 creating segments file from rttm and utt2spk, reco2file_and_channel "
  local/convert_rttm_to_utt2spk_and_segments.py --append-reco-id-to-spkr=true $rttm_dir/rttm_introduction_removed \
    <(awk '{print $2".ENH "$2" "$3}' $rttm_dir/rttm_introduction_removed |sort -u) \
    ${out_dir}_hires/utt2spk ${out_dir}_hires/segments

  utils/utt2spk_to_spk2utt.pl ${out_dir}_hires/utt2spk > ${out_dir}_hires/spk2utt

  awk '{print $1" "$1" 1"}' ${out_dir}_hires/wav.scp > ${out_dir}_hires/reco2file_and_channel
  utils/fix_data_dir.sh ${out_dir}_hires || exit 1;
fi

if [ $stage -le 2 ]; then
  echo "$0 extracting mfcc freatures using segments file"
  steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --nj $nj --cmd queue.pl ${out_dir}_hires
  steps/compute_cmvn_stats.sh ${out_dir}_hires
  cp $data_in/text.bak ${out_dir}_hires/text
fi

if [ $stage -le 3 ]; then
  echo "$0 performing decoding on the extracted features"
  local/nnet3/decode.sh --affix 2stage --acwt 1.0 --post-decode-acwt 10.0 \
    --frames-per-chunk 150 --nj $nj --ivector-dir $ivector_extractor \
    $out_dir $lang_dir $asr_model_dir/tree_sp/graph $asr_model_dir/tdnn1b_sp/
fi

