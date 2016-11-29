#!/bin/bash

# Copyright 2014  Guoguo Chen
#           2016  Vimal Manohar
# Apache 2.0

set -e 
set -o pipefail

. path.sh

set -u

# Uniform segmentation options
max_segment_duration=30
overlap_duration=5

# TF-IDF similarity search options
num_neighbors_to_search=2
neighbor_tfidf_threshold=0

stage=-1

cmd=run.pl

. utils/parse_options.sh

if [ $# -ne 5 ]; then
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

# First we split the data into segments of around 30s long, on which 
# it would be possible to do a decoding.
if [ $stage -le 1 ]; then
  echo "$0: Splitting data directory $data into uniform segments..."

  # TODO: Write this script
  # utils/data/get_uniform_subsegments.py \
  #   --max-segment-duration=$max_segment_duration \
  #   --overlap-duration=$overlap_duration \
  #   $data/segments > $dir/uniform_sub_segments

  utils/data/get_segments_for_data.sh $data > $data/segments

  cat $data/utt2spk | awk 'BEGIN{i=1} {print $1" "i; i++}' > $data/utt2label

  segmentation-init-from-segments --shift-to-zero=true \
    --utt2label-rspecifier=ark,t:$data/utt2label $data/segments ark:- | \
    segmentation-post-process \
      --max-segment-length=`perl -e "print $max_segment_duration * 100"` \
      --overlap-length=`perl -e "print $overlap_duration * 100"` \
      ark:- ark:- | \
    segmentation-to-segments --single-speaker \
      ark:- ark:/dev/null $dir/uniform_sub_segments
  #- | \
  #  utils/int2sym.pl -f 2 $data/utt2label > $dir/uniform_sub_segments

  awk '{print $1" "$2}' $dir/uniform_sub_segments > $dir/new2old_utts

  utils/data/subsegment_data_dir.sh $data $dir/uniform_sub_segments $data_uniform_seg
  # Map the original text file to the uniform segments,
  # even though the segments don't actually correspond to the whole text. 
  # This is ok because this text file is actually used to train the biased
  # LM and not for any other task.
  cat $dir/new2old_utts | \
    utils/apply_map.pl -f 2 $data/text > $data_uniform_seg/text
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
  steps/cleanup/internal/get_ctm.sh \
    --lmwt $lmwt --cmd "$cmd --mem 4G" \
    $data_uniform_seg $lang $dir/lats
fi

if [ $stage -le 5 ]; then
  # Split the original text into documents, over which we can do 
  # searching reasonably efficiently.
  # Since the Smith-Waterman alignment is linear in the length of the 
  # text, we want to keep it reasonably small (a few thousand words). The
  # relevant documents can be found using TF-IDF similarity and nearby 
  # documents can also be picked for the Smith-Waterman alignment stage.
  steps/cleanup/internal/split_text_into_docs.pl --max-words $max_words \
    $data/text $dir/docs/docs.txt $dir/docs/doc2text
  utis/utt2spk_to_spk2utt.pl $dir/docs/doc2text > $dir/docs/text2doc

  steps/cleanup/internal/compute_tf_idf.py \
    --tf-weighting-scheme="raw" \
    --idf-weighting-scheme="log" \
    --output-idf-stats=$dir/docs/idf_stats.txt \
    $dir/docs/docs.txt $dir/docs/src_tf_idf.txt
fi

exit 0

if [ $stage -le 5 ]; then
  for n in `seq $nj`; do
    awk 'print ($1" "$1" A")' $data_uniform_seg/split$nj/$n/text > \
    $dir/lats/fake_reco2file_and_channel.$n
  done

  echo $nj > $dir/docs/num_jobs

  $cmd JOB=1:$nj $dir/docs/log/get_tfidf_for_texts.JOB.log \
    utils/filter_scp.pl $data_uniform_seg/split$nj/JOB/utt2spk \
      $dir/new2old_utts '>' $dir/docs/split$nj/JOB/new2old_utts '&&' \
    cut -d ' ' -f 2 $dir/docs/split$nj/JOB/new2old_utts \| \
    utils/filter_scp.pl /dev/stdin $dir/docs/text2doc \| \
      utils/spk2utt_to_utt2spk.pl '>' \
      $dir/docs/split$nj/JOB/doc2text '&&' \
    utils/filter_scp.pl $dir/docs/split$nj/JOB/doc2text \
      $dir/docs/docs.txt '>' $dir/docs/split$nj/JOB/docs.txt '&&' \
    steps/cleanup/internal/compute_tf_idf.py \
      --tf-weighting-scheme="raw" \
      --idf-weighting-scheme="log" \
      --input-idf-stats=$dir/docs/idf_stats.txt \
      - $dir/docs/split$nj/JOB/src_tf_idf.txt 

  $cmd JOB=1:$nj $dir/lats/log/get_ctm_edits.JOB.log \
    steps/cleanup/internal/ctm_to_text.pl \
      $dir/lats/score_$lmwt/${data_id}_uniform_seg.ctm.JOB \| \
    steps/cleanup/internal/compute_tf_idf.py \
      --tf-weighting-scheme="normalized" \
      --idf-weighting-scheme="log" \
      --input-idf-stats=$dir/docs/idf_stats.txt \
      --accumulate-over-docs=false \
      - - \| \
    steps/cleanup/internal/retrieve_similar_docs.py \
      --query-tfidf=- \
      --source-tfidf=$dir/docs/split$nj/JOB/src_tf_idf.txt \
      --source-text2doc=$dir/docs/text2doc \
      --query-doc2source=$dir/new2old_utts \
      --num-neighbors-to-search=$num_neighors_to_search \
      --neighbor-tfidf-threshold=$neighbor_tfidf_threshold \
      --output-docs=- \| \
    steps/cleanup/internal/stitch_documents.py \
      --query2docs=- \
      --input-documents=$dir/docs/split$nj/JOB/docs.txt \
      --output-documents=- \| \
    steps/cleanup/internal/align_ctm_ref.py --eps-symbol="***" \
      --hyp-format=CTM \
      --reco2file-and-channel=$dir/lats/fake_reco2file_and_channel.JOB \
      $dir/lats/score_$lmwt/${data_id}_uniform_seg.ctm.JOB \
      - \
      $dir/lats/score_$lmwt/${data_id}_uniform_seg.ctm_edits.JOB

  cat $dir/lats/score_$lmwt/${data_id}_uniform_seg.ctm_edits.* > $dir/lats/score_$lmwt/${data_id}_uniform_seg.ctm_edits
fi

if [ $stage -le 6 ]; then
  utils/copy_data_dir.sh $data_uniform_seg $out_data

  cat $dir/lats/score_$lmwt/${data_id}_uniform_seg.ctm_edits | \
    python -c "
import sys
text = []
prev_utt = ''
for line in sys.stdin.readlines():
  parts = line.strip().split()
  utt = parts[0]

  if prev_utt != '' and utt != prev_utt:
    print ('{0} {1}'.format(prev_utt, ' '.join(text)))
    text = []
  word = parts[6] if (parts[6] != '$eps_symbol') else ''
  text.append(word)
  prev_utt = utt
if prev_utt != '' and len(text) > 0:
    print ('{0} {1}'.format(prev_utt, ' '.join(text)))" \
      > $out_data/text
fi
