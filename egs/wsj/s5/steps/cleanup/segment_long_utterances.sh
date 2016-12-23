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

# Decode options
graph_opts=
nj=4
lmwt=10

# TF-IDF similarity search options
max_words=1000
num_neighbors_to_search=1
neighbor_tfidf_threshold=0.5

# First-pass segmentation opts
min_segment_length=0.5
min_new_segment_length=1.0
max_tainted_length=0.05
max_edge_silence_length=0.5
max_edge_non_scored_length=0.5
max_internal_silence_length=2.0
max_internal_non_scored_length=2.0
unk_padding=0.05
max_junk_proportion=0.1
min_split_point_duration=0.1
max_deleted_words_kept_when_merging=1
silence_factor=1
incorrect_words_factor=1
tainted_words_factor=1
max_wer=50
max_segment_length_for_merging=60
max_bad_proportion=0.75
max_intersegment_incorrect_words_length=1
max_segment_length_for_splitting=10
hard_max_segment_length=15
min_silence_length_to_split_at=0.3
min_non_scored_length_to_split_at=0.3

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

for f in $data/feats.scp $data/cmvn.scp $srcdir/tree \
  $srcdir/final.mdl $srcdir/cmvn_opts; do
  if [ ! -f $f ]; then
    echo "$0: Could not find file $f"
    exit 1
  fi
done

data_id=`basename $data`
mkdir -p $dir

data_uniform_seg=$dir/${data_id}_uniform_seg

# First we split the data into segments of around 30s long, on which 
# it would be possible to do a decoding.
if [ $stage -le 1 ]; then
  echo "$0: Splitting data directory $data into uniform segments..."

  utils/data/get_utt2dur.sh $data

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
fi

if [ $stage -le 2 ]; then
  utils/data/subsegment_data_dir.sh $data $dir/uniform_sub_segments $data_uniform_seg

  # make frame-overlap slightly larger so that the number of frames decreases 
  # by perhaps at most 1 to fix some round-off errors.
  rm $data/reco2num_frames $data/reco2dur || true
  utils/data/get_reco2num_frames.sh --frame-shift 0.01 --frame-overlap 0.0151 \
    $data

  awk '{print $1" "$2}' $data_uniform_seg/segments | \
    utils/apply_map.pl -f 2 $data/reco2num_frames > \
    $data_uniform_seg/utt2max_frames

  utils/data/get_subsegment_feats.sh $data/feats.scp 0.01 0.015 \
    $dir/uniform_sub_segments | \
    utils/data/fix_subsegmented_feats.pl $data_uniform_seg/utt2max_frames > \
    $data_uniform_seg/feats.scp 

  # Map the original text file to the uniform segments,
  # even though the segments don't actually correspond to the whole text. 
  # This is ok because this text file is actually used to train the biased
  # LM and not for any other task.
  cat $dir/new2old_utts | \
    utils/apply_map.pl -f 2 $data/text > $data_uniform_seg/text

  cp $data/cmvn.scp $data_uniform_seg/
fi

if [ $stage -le 3 ]; then
  echo "$0: Building biased-language-model decoding graphs..."

  graph_dir=$dir/graphs
  mkdir -p $graph_dir

  cp $srcdir/final.mdl $graph_dir
  cp $srcdir/tree $graph_dir
  cp $srcdir/cmvn_opts $graph_dir
  cp $srcdir/{splice_opts,delta_opts,final.mat,final.alimdl} $graph_dir 2>/dev/null || true

  utils/lang/check_phones_compatible.sh $lang/phones.txt $srcdir/phones.txt
  cp $lang/phones.txt $graph_dir

  steps/cleanup/make_biased_lm_graphs.sh $graph_opts \
    --nj $nj --cmd "$cmd" \
     $data $lang $graph_dir

  cat $dir/uniform_sub_segments | awk '{print $1" "$2}' | \
    utils/apply_map.pl -f 2 $graph_dir/HCLG.fsts.scp > $dir/HCLG.fsts.scp
  cp $graph_dir/words.txt $dir
  [ -f $graph_dir/num_pdfs ] && cp $graph_dir/num_pdfs $dir
fi

beam=15.0

decode_dir=$dir/lats
mkdir -p $decode_dir

if [ $stage -le 4 ]; then
  echo "$0: Decoding with biased language models..."
  transform_opt=
  if [ -f $srcdir/trans.1 ]; then
    # If srcdir contained trans.* then we assume they are fMLLR transforms for
    # this data, and we use them.
    transform_opt="--transform-dir $srcdir"
  fi
  
  cp $srcdir/final.mdl $decode_dir
  cp $srcdir/tree $decode_dir
  cp $srcdir/cmvn_opts $decode_dir
  cp $srcdir/{splice_opts,delta_opts,final.mat,final.alimdl} $decode_dir 2>/dev/null || true

  utils/lang/check_phones_compatible.sh $lang/phones.txt $srcdir/phones.txt
  cp $lang/phones.txt $decode_dir

  steps/cleanup/decode_segmentation.sh \
    --beam $beam --nj $nj --cmd "$cmd --mem 4G" $transform_opt \
    --skip-scoring true --allow-partial false \
    $dir $data_uniform_seg $decode_dir
fi

if [ $stage -le 5 ]; then
  steps/cleanup/internal/get_ctm.sh \
    --lmwt $lmwt --cmd "$cmd --mem 4G" \
    --print-silence true \
    $data_uniform_seg $lang $dir/lats
fi

if [ $stage -le 6 ]; then
  mkdir -p $dir/docs
  # Split the original text into documents, over which we can do 
  # searching reasonably efficiently.
  # Since the Smith-Waterman alignment is linear in the length of the 
  # text, we want to keep it reasonably small (a few thousand words). The
  # relevant documents can be found using TF-IDF similarity and nearby 
  # documents can also be picked for the Smith-Waterman alignment stage.
  steps/cleanup/internal/split_text_into_docs.pl --max-words $max_words \
    $data/text $dir/docs/doc2text $dir/docs/docs.txt
  utils/utt2spk_to_spk2utt.pl $dir/docs/doc2text > $dir/docs/text2doc

  steps/cleanup/internal/compute_tf_idf.py \
    --tf-weighting-scheme="raw" \
    --idf-weighting-scheme="log" \
    --output-idf-stats=$dir/docs/idf_stats.txt \
    $dir/docs/docs.txt $dir/docs/src_tf_idf.txt
fi

if [ $stage -le 7 ]; then
  echo $nj > $dir/docs/num_jobs
  
  mkdir -p $dir/docs/split$nj/
  $cmd JOB=1:$nj $dir/docs/log/get_tfidf_for_texts.JOB.log \
    mkdir -p $dir/docs/split$nj/JOB '&&' \
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
      $dir/docs/split$nj/JOB/docs.txt $dir/docs/split$nj/JOB/src_tf_idf.txt 
fi

if [ $stage -le 8 ]; then
  for n in `seq $nj`; do
    awk '{print ($1" "$1" 1")}' $data_uniform_seg/split$nj/$n/utt2spk > \
      $dir/lats/fake_reco2file_and_channel.$n
  done

  $cmd JOB=1:$nj $dir/lats/log/get_ctm_edits.JOB.log \
    steps/cleanup/internal/ctm_to_text.pl \
      $dir/lats/score_$lmwt/${data_id}_uniform_seg.ctm.JOB \| \
    steps/cleanup/internal/compute_tf_idf.py \
      --tf-weighting-scheme="normalized" \
      --idf-weighting-scheme="log" \
      --input-idf-stats=$dir/docs/idf_stats.txt \
      --accumulate-over-docs=false \
      - $dir/docs/split$nj/JOB/query_tf_idf.txt '&&' \
    steps/cleanup/internal/retrieve_similar_docs.py \
      --query-tfidf=$dir/docs/split$nj/JOB/query_tf_idf.txt \
      --source-tfidf=$dir/docs/split$nj/JOB/src_tf_idf.txt \
      --source-text-id2doc-ids=$dir/docs/text2doc \
      --query-id2source-text-id=$dir/new2old_utts \
      --num-neighbors-to-search=$num_neighbors_to_search \
      --neighbor-tfidf-threshold=$neighbor_tfidf_threshold \
      --relevant-docs=$dir/docs/split$nj/JOB/relevant_docs.txt '&&' \
    steps/cleanup/internal/stitch_documents.py \
      --query2docs=$dir/docs/split$nj/JOB/relevant_docs.txt \
      --input-documents=$dir/docs/split$nj/JOB/docs.txt \
      --output-documents=- \| \
    steps/cleanup/internal/align_ctm_ref.py --eps-symbol='"<eps>"' \
      --oov-word="'`cat $lang/oov.txt`'" --symbol-table=$lang/words.txt \
      --hyp-format=CTM \
      --reco2file-and-channel=$dir/lats/fake_reco2file_and_channel.JOB \
      --hyp=$dir/lats/score_$lmwt/${data_id}_uniform_seg.ctm.JOB --ref=- \
      --output=$dir/lats/score_$lmwt/${data_id}_uniform_seg.ctm_edits.JOB 
  
  for n in `seq $nj`; do
    cat $dir/lats/score_$lmwt/${data_id}_uniform_seg.ctm_edits.$n 
  done > $dir/lats/score_$lmwt/ctm_edits
  
fi

if [ $stage -le 9 ]; then
  steps/cleanup/internal/resolve_ctm_edits_overlaps.py \
    ${data_uniform_seg}/segments $dir/lats/score_$lmwt/ctm_edits $dir/ctm_edits
fi

if [ $stage -le 10 ]; then
  echo "$0: using default values of non-scored words..."

  # At the level of this script we just hard-code it that non-scored words are
  # those that map to silence phones (which is what get_non_scored_words.py
  # gives us), although this could easily be made user-configurable.  This list
  # of non-scored words affects the behavior of several of the data-cleanup
  # scripts; essentially, we view the non-scored words as negotiable when it
  # comes to the reference transcript, so we'll consider changing the reference
  # to match the hyp when it comes to these words.
  steps/cleanup/internal/get_non_scored_words.py $lang > $dir/non_scored_words.txt
fi

if [ $stage -le 11 ]; then
  echo "$0: modifying ctm-edits file to allow repetitions [for dysfluencies] and "
  echo "   ... to fix reference mismatches involving non-scored words. "

  $cmd $dir/log/modify_ctm_edits.log \
    steps/cleanup/internal/modify_ctm_edits.py --verbose=3 $dir/non_scored_words.txt \
    $dir/ctm_edits $dir/ctm_edits.modified

  echo "   ... See $dir/log/modify_ctm_edits.log for details and stats, including"
  echo " a list of commonly-repeated words."
fi

if [ $stage -le 12 ]; then
  echo "$0: applying 'taint' markers to ctm-edits file to mark silences and"
  echo "  ... non-scored words that are next to errors."
  $cmd $dir/log/taint_ctm_edits.log \
       steps/cleanup/internal/taint_ctm_edits.py --remove-deletions=false \
       $dir/ctm_edits.modified $dir/ctm_edits.tainted
  echo "... Stats, including global cor/ins/del/sub stats, are in $dir/log/taint_ctm_edits.log."
fi

if [ $stage -le 13 ]; then
  echo "$0: creating segmentation from ctm-edits file."

  segmentation_opts=(
  --min-segment-length=$min_segment_length
  --min-new-segment-length=$min_new_segment_length
  --max-tainted-length=$max_tainted_length
  --max-edge-silence-length=$max_edge_silence_length
  --max-edge-non-scored-length=$max_edge_non_scored_length
  --max-internal-silence-length=$max_internal_silence_length
  --max-internal-non-scored-length=$max_internal_non_scored_length
  --unk-padding=$unk_padding
  --max-junk-proportion=$max_junk_proportion
  --min-split-point-duration=$min_split_point_duration
  --max-deleted-words-kept-when-merging=$max_deleted_words_kept_when_merging
  --merging-score.silence-factor=$silence_factor
  --merging-score.incorrect-words-factor=$incorrect_words_factor
  --merging-score.tainted-words-factor=$tainted_words_factor
  --merging.max-wer=$max_wer
  --merging.max-segment-length=$max_segment_length_for_merging
  --merging.max-bad-proportion=$max_bad_proportion
  --merging.max-intersegment-incorrect-words-length=$max_intersegment_incorrect_words_length
  --splitting.max-segment-length=$max_segment_length_for_splitting
  --splitting.hard-max-segment-length=$hard_max_segment_length
  --splitting.min-silence-length=$min_silence_length_to_split_at
  --splitting.min-non-scored-length=$min_non_scored_length_to_split_at
  )
  
  $cmd $dir/log/segment_ctm_edits.log \
    steps/cleanup/internal/segment_ctm_edits_mild.py \
      ${segmentation_opts[@]} \
      --oov-symbol-file=$lang/oov.txt \
      --ctm-edits-out=$dir/ctm_edits.segmented \
      --word-stats-out=$dir/word_stats.txt \
      $dir/non_scored_words.txt \
      $dir/ctm_edits.tainted $dir/text $dir/segments

  echo "$0: contents of $dir/log/segment_ctm_edits.log are:"
  cat $dir/log/segment_ctm_edits.log
  echo "For word-level statistics on p(not-being-in-a-segment), with 'worst' words at the top,"
  echo "see $dir/word_stats.txt"
  echo "For detailed utterance-level debugging information, see $dir/ctm_edits.segmented"
fi

mkdir -p $out_data
if [ $stage -le 14 ]; then
  utils/data/subsegment_data_dir.sh $data_uniform_seg $dir/segments $dir/text $out_data
  
  utils/data/get_reco2num_frames.sh --frame-shift 0.01 --frame-overlap 0.0151 \
    $data

  awk '{print $1" "$2}' $out_data/segments | \
    utils/apply_map.pl -f 2 $data/reco2num_frames > \
    $out_data/utt2max_frames

  utils/data/get_subsegment_feats.sh $data_uniform_seg/feats.scp 0.01 0.015 \
    $dir/segments | \
    utils/data/fix_subsegmented_feats.pl $out_data/utt2max_frames > \
    $out_data/feats.scp 
fi
