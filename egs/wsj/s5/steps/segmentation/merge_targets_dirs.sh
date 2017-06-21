#! /bin/bash

# Copyright 2017  Vimal Manohar
# Apache 2.0

cmd=run.pl 
nj=4
weights=        # A comma-separated list of weights corresponding to each
                # target source being combined.
remove_mismatch_frames=true     # If true, the mismatch frames are removed by 
                                # setting targets to 0 in the following cases:
                                # a) If none of the sources have a column with value > 0.5
                                # b) If two sources have columns with value > 0.5, but
                                # they occur at different indexes e.g. silence prob is > 0.5 for the
                                # targets from alignment, and speech prob > 0.5 for the targets from
                                # decoding

[ -f ./path.sh ] && . ./path.sh 
. utils/parse_options.sh

if [ $# -lt 3 ]; then
  cat <<EOF
  This script merges targets dirs created from multiple sources (systems) into
  single targets matrices.
  Usage: steps/segmentation/merge_targets_dirs.py <data> <targets-1> <targets-2> ... <merged-targets>
  e.g.: steps/segmentation/merge_targets_dirs.py --weights 1.0,0.5 \
      data/train_whole \
      exp/segmentation1a/tri3b_train_whole_sup_targets_sub3 \
      exp/segmentation1a/tri3b_train_whole_targets_sub3 \
      exp/segmentation1a/tri3b_train_whole_combined_targets_sub3
EOF
  exit 1
fi

data=$1
dir=${@: -1}  # last argument to the script
shift;

targets_dirs=( $@ )  # read the remaining arguments into an array
unset targets_dir[${#targets_dirs[@]}-1]  # 'pop' the last argument which is odir
num_sources=${#targets_dirs[@]}  # number of targets to combine

utils/data/split_data.sh --per-utt $data $nj

frame_subsampling_factor=1
if [ ! -f ${targets_dir[0]}/frame_subsampling_factor ]; then
  frame_subsampling_factor=$(cat ${targets_dir[0]}/frame_subsampling_factor) || exit 1
fi

for t in ${targets_dirs[@]}; do
  this_frame_subsampling_factor=1
  if [ ! -f $t/frame_subsampling_factor ]; then
    this_frame_subsampling_factor=$(cat $t/frame_subsampling_factor) || exit 1
  fi
  if [ $this_frame_subsampling_factor -ne $frame_subsampling_factor ]; then
    echo "$0: Mismatch in frame_subsampling_factor in $t and ${targets_dir[0]}; $this_frame_subsampling_factor vs $frame_subsampling_factor"
    exit 1
  fi
  targets_rspecifiers+=("scp:utils/filter_scp.pl $sdata/JOB/utt2spk $t/targets.scp |")
done

# convert $dir to an absolute pathname.
fdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $dir ${PWD}`

$cmd JOB=1:$nj $dir/log/merge_targets.JOB.log \
  paste-feats "${targets_rspecifier[@]}" ark,t:- \| \
  steps/segmentation/internal/merge_targets.py --weights="$weights" \
    --remove-mismatch-frames=$remove_mismatch_frames - - \| \
  copy-feats ark,t:- ark,scp:$fdir/targets.JOB.ark,$fdir/targets.JOB.scp || exit 1

for n in `seq $nj`; do
  cat $dir/targets.$n.scp
done > $dir/targets.scp

if [ $frame_subsampling_factor -ne 1 ]; then
  echo $frame_subsampling_factor > $dir/frame_subsampling_factor
fi

steps/segmentation/validate_targets_dir.sh $dir $data || exit 1

echo "$0: Merged target directories to $dir"

exit 0
