#!/bin/bash

# Copyright 2014  Guoguo Chen
#           2016  Vimal Manohar
# Apache 2.0

if [ $# -ne 4 ]; then
    cat <<EOF
Usage: $0 [options] <model-dir> <lang> <data-in> <segmented-data-out> <work-dir>
 e.g.: $0 exp/tri3 data/train_unsegmented data/train_segmented exp/tri3_segmentation
This script performs segmentation of the data in <data-in>, writing the segmented
data (with a segments file) to <segmented-data-out>.  The purpose of this script is
to divide up the input data (which may consist of long recordings such as television
shows or audiobooks) into segments which are of manageable length for further
processing, along with the portion of the transcript that seems to match each segment.
The output data is not necessarily particularly clean; you are advised to 
run steps/cleanup/clean_and_segment_data.sh on the output in order to further
clean it and eliminate data where the transcript doesn't seem to match.
EOF
    exit 1
fi

srcdir=$1
lang=$2
data=$3
out_data=$4
dir=$5

for f in $data/feats.scp; do
  if [ ! -f $f ]; then
    echo "$0: Could not find file $f"
    exit 1
  fi
done

data_id=`basename $data`
mkdir -p $dir

utils/data/get_utt2dur.sh $data

data_uniform_seg=$dir/${data_id}_uniform_seg

if [ $stage -le 1 ]; then
  echo "$0: Splitting data directory $data into uniform segments..."

  # TODO: Write this script
  utils/data/get_uniform_subsegments.py \
    --max-segment-duration=$max_segment_duration \
    --overlap-duration=$overlap_duration \
    $data/segments > $dir/uniform_sub_segments

  utils/subsegment_data_dir.sh $data $dir/uniform_sub_segments $data_uniform_seg
  awk '{print $1" "$2}' $dir/uniform_sub_segments | \
    utils/apply_map.pl $data/text > $data_uniform_seg/text
fi

if [ $stage -le 2 ]; then
  echo "$0: Building biased-language-model decoding graphs..."
  steps/cleanup/make_biased_lm_graphs.sh $graph_opts \
    --nj $nj --cmd "$cmd" \
     $data $lang $dir/graphs
  cat $dir/uniform_sub_segments | awk '{print $1" "$2}' | \
    utils/apply_map.pl -f 2 $dir/graphs/HCLG.fsts.scp > $dir/HCLG.fsts.scp
fi

beam=15.0

if [ $stage -le 3 ]; then
  echo "$0: Decoding with biased language models..."
  transform_opt=
  if [ -f $srcdir/trans.1 ]; then
    # If srcdir contained trans.* then we assume they are fMLLR transforms for
    # this data, and we use them.
    transform_opt="--transform-dir $srcdir"
  fi
  steps/cleanup/decode_segmentation.sh \
    --beam $beam --nj $nj --cmd "$cmd --mem 4G" $transform_opt \
    --skip-scoring true --allow-partial false \
    $dir $data_uniform_seg $dir/lats
fi

if [ $stage -le 4 ]; then
  steps/get_ctm.sh --use-segments false \
    --cmd "$cmd --mem 4G" $data_uniform_seg $lang \
    $dir/lats
fi

if [ $stage -le 5 ]; then
  steps/cleanup/internal/align_ctm_ref.py \
    $dir/lats/score_$lmwt/${data_id}_uniform_seg.ctm \
    $data_uniform_seg $dir/lats/score_$lmwt/${data_id}_uniform_seg.ctm_align
fi

