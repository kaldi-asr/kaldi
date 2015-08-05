#!/bin/bash

set -e

. path.sh

cmd=run.pl

. utils/parse_options.sh

data_id=dev
ivector_affix=
ivector_dir=exp/nnet2_multicondition

diarization_dir=$ivector_dir/diarization_${data_id}${ivector_affix}
ctm_file=exp/nnet2_multicondition/nnet_ms_a/decode_dev_aspire_whole_uniformsegmented_win10_over5_v23_voiced_256_128_128_iterfinal_pp_fg/score_13/penalty_0.5/ctm.filt
reco2file_and_channel=data/dev_aspire/reco2file_and_channel
dir=$ivector_dir/diarization_${data_id}${ivector_affix}/ctm_filter

if [ $# -ne 4 ]; then
  echo "Usage: diarization/filter_ctm.sh <diarization-dir> <ctm-file> <reco2file-and-channel> <dir>"
  echo " e.g.: diarization/filter_ctm.sh $diarization_dir $ctm_file $reco2file_and_channel $dir"
  exit 1
fi

diarization_dir=$1
ctm_file=$2
reco2file_and_channel=$3
dir=$4

nj=$(cat $diarization_dir/num_jobs)

for n in `seq $nj`; do 
  cat $diarization_dir/diarization/diarization_segmentation.$n.scp
done > $dir/diarization_segmentation.scp

$cmd $dir/log/compute_ctm_conf.log \
  segmentation-compute-class-ctm-conf "ark,s:segmentation-post-process --merge-adjacent-segments scp:$dir/diarization_segmentation.scp ark:- |" $ctm_file  $reco2file_and_channel ark,t:- \| diarization/convert_speaker_conf_to_labels_string.pl --min-spk-conf 0.9 '>' $dir/remove_labels.csl.txt

$cmd $dir/log/get_nocrosstalk_segmentation.log \
  segmentation-remove-segments --remove-labels-rspecifier=ark,t:$dir/remove_labels.csl.txt scp:$dir/diarization_segmentation.scp ark:- \| segmentation-post-process --merge-labels=0:1:2:3 --merge-dst-label=0 --merge-adjacent-segments ark:- ark:$dir/non_crosstalk_segmentation.ark

$cmd $dir/log/filter_ctm.log \
  segmentation-filter-ctm ark:$dir/non_crosstalk_segmentation.ark $ctm_file $reco2file_and_channel ${ctm_file}.nocrosstalk
