#!/bin/bash

# Copyright 2014  Guoguo Chen
#           2016  Vimal Manohar
# Apache 2.0

. path.sh

set -e 
set -o pipefail
set -u

# Uniform segmentation options
max_segment_duration=30
overlap_duration=5
seconds_per_spk_max=30

# Decode options
graph_opts=
beam=15.0
lattice_beam=1.0
nj=4
lmwt=10

# TF-IDF similarity search options
max_words=1000
num_neighbors_to_search=1
neighbor_tfidf_threshold=0.5

align_full_hyp=false

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

if [ $# -ne 6 ]; then
    cat <<EOF
Usage: $0 [options] <model-dir> <lang> <data-in> <text-in> <segmented-data-out> <work-dir>
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
text=$4
out_data=$5
dir=$6

for f in $data/feats.scp $text $srcdir/tree \
  $srcdir/final.mdl $srcdir/cmvn_opts; do
  if [ ! -f $f ]; then
    echo "$0: Could not find file $f"
    exit 1
  fi
done

data_id=`basename $data`
mkdir -p $dir

data_uniform_seg=$dir/${data_id}_uniform_seg
  
frame_shift=`utils/data/get_frame_shift.sh $data`

# First we split the data into segments of around 30s long, on which 
# it would be possible to do a decoding. 
if [ $stage -le 1 ]; then
  echo "$0: Stage 1 (Splitting data directory $data into uniform segments)"

  utils/data/get_utt2dur.sh $data

  # TODO: Write this script
  # utils/data/get_uniform_subsegments.py \
  #   --max-segment-duration=$max_segment_duration \
  #   --overlap-duration=$overlap_duration \
  #   $data/segments > $dir/uniform_sub_segments

  if [ ! -f $data/segments ]; then
    utils/data/get_segments_for_data.sh $data > $data/segments
  fi

  cat $data/utt2spk | awk 'BEGIN{i=1} {print $1" "i; i++}' > $data/utt2label

  segmentation-init-from-segments --shift-to-zero=true \
    --utt2label-rspecifier=ark,t:$data/utt2label $data/segments ark:- | \
    segmentation-post-process \
      --max-segment-length=`perl -e "print $max_segment_duration * 100"` \
      --overlap-length=`perl -e "print $overlap_duration * 100"` \
      ark:- ark:- | \
    segmentation-to-segments --single-speaker --frame-shift=$frame_shift \
      --frame-overlap=0 ark:- ark:/dev/null $dir/uniform_sub_segments

  # Get a mapping from the new to old utterance-ids.
  # Old-utterance is the whole recording.
  awk '{print $1" "$2}' $dir/uniform_sub_segments > $dir/new2old_utts
fi

if [ $stage -le 2 ]; then
  echo "$0: Stage 2 (Prepare uniform sub-segmented data directory)"
  rm -r $data_uniform_seg || true

  if [ ! -z "$seconds_per_spk_max" ]; then
    utils/data/subsegment_data_dir.sh \
      $data $dir/uniform_sub_segments $dir/${data_id}_uniform_seg.temp

    utils/data/modify_speaker_info.sh --seconds-per-spk-max $seconds_per_spk_max \
      $dir/${data_id}_uniform_seg.temp $data_uniform_seg 
  else
    utils/data/subsegment_data_dir.sh \
      $data $dir/uniform_sub_segments $data_uniform_seg
  fi

  # make frame-overlap slightly larger so that the number of frames decreases 
  # by perhaps at most 1 to fix some round-off errors.
  utils/data/get_utt2num_frames.sh --cmd "$cmd" --nj $nj $data
  awk '{print $1" "$2}' $dir/uniform_sub_segments | \
    utils/apply_map.pl -f 2 $data/utt2num_frames > \
    $data_uniform_seg/utt2max_frames

  #rm $data/reco2num_frames $data/reco2dur || true
  #utils/data/get_reco2num_frames.sh --cmd "$cmd" --nj $nj \
  #  --frame-shift $frame_shift \
  #  --frame-overlap `perl -e "print $frame_shift * 1.51"` \
  #  $data

  #awk '{print $1" "$2}' $data_uniform_seg/segments | \
  #  utils/apply_map.pl -f 2 $data/reco2num_frames > \
  #  $data_uniform_seg/utt2max_frames

  utils/data/get_subsegmented_feats.sh $data/feats.scp $frame_shift 0.0 \
    $dir/uniform_sub_segments | \
    utils/data/fix_subsegmented_feats.pl $data_uniform_seg/utt2max_frames > \
    $data_uniform_seg/feats.scp 

  # Map the original text file to the uniform segments,
  # even though the segments don't actually correspond to the whole text. 
  # This is ok because this text file is actually used to train the biased
  # LM and not for any other task.
  cut -d ' ' -f 1,2 $data/segments > $data/utt2reco
  cat $dir/new2old_utts | utils/apply_map.pl -f 2 $data/utt2reco | \
    utils/filter_scp.pl -f 2 $text | \
    utils/apply_map.pl -f 2 $text > $data_uniform_seg/text

  wc_orig=$(cat $dir/uniform_sub_segments | wc -l)
  wc_text=$(cat $data_uniform_seg/text | wc -l)
  
  if [ $[$wc_orig*9] -gt $[$wc_text*10] ]; then
    echo "$0: Lost too many segments; orig ($wc_orig) vs ($wc_text)"
    exit 1
  fi

  utils/fix_data_dir.sh $data_uniform_seg

  # Compute new cmvn stats for the segmented data directory
  steps/compute_cmvn_stats.sh $data_uniform_seg/
fi

graph_dir=$dir/graphs_uniform_seg

if [ $stage -le 3 ]; then
  echo "$0: Stage 3 (Building biased-language-model decoding graphs)"

  cp $srcdir/final.mdl $dir
  cp $srcdir/tree $dir
  cp $srcdir/cmvn_opts $dir
  cp $srcdir/{splice_opts,delta_opts,final.mat,final.alimdl} $dir 2>/dev/null || true
  cp $srcdir/phones.txt $dir

  # Make graphs w.r.t. to the original text (recording-level) and then copy it to the segments. 
  steps/cleanup/make_biased_lm_graphs.sh $graph_opts \
    --nj $nj --cmd "$cmd" \
     $text $lang $dir $dir/graphs

  mkdir -p $graph_dir
  cat $data_uniform_seg/segments | awk '{print $1" "$2}' | \
    utils/apply_map.pl -f 2 $dir/graphs/HCLG.fsts.scp > \
    $graph_dir/HCLG.fsts.scp
  
  cp $lang/words.txt $graph_dir
  cp -r $lang/phones $graph_dir
  [ -f $dir/graphs/num_pdfs ] && cp $dir/graphs/num_pdfs $graph_dir/
fi

decode_dir=$dir/lats
mkdir -p $decode_dir

if [ $stage -le 4 ]; then
  echo "$0: Decoding with biased language models..."
  
  if [ -f $srcdir/trans.1 ]; then
    steps/cleanup/decode_fmllr_segmentation.sh \
      --beam $beam --lattice-beam $lattice_beam --nj $nj --cmd "$cmd --mem 4G" \
      --skip-scoring true --allow-partial false \
      $graph_dir $data_uniform_seg $decode_dir
  else 
    steps/cleanup/decode_segmentation.sh \
      --beam $beam --lattice-beam $lattice_beam --nj $nj --cmd "$cmd --mem 4G" \
      --skip-scoring true --allow-partial false \
      $graph_dir $data_uniform_seg $decode_dir
  fi
fi

if [ $stage -le 5 ]; then
  steps/cleanup/internal/get_ctm.sh \
    --lmwt $lmwt --cmd "$cmd --mem 4G" \
    --print-silence true \
    $data_uniform_seg $lang $decode_dir
fi

if [ $stage -le 6 ]; then
  mkdir -p $dir/docs
  # Split the original text into documents, over which we can do 
  # searching reasonably efficiently. Also get a mapping from the original
  # text to the created documents (i.e. text2doc)
  # Since the Smith-Waterman alignment is linear in the length of the 
  # text, we want to keep it reasonably small (a few thousand words). 
  steps/cleanup/internal/split_text_into_docs.pl --max-words $max_words \
    $text $dir/docs/doc2text $dir/docs/docs.txt
  utils/utt2spk_to_spk2utt.pl $dir/docs/doc2text > $dir/docs/text2doc
fi

if [ $stage -le 7 ]; then
  echo $nj > $dir/docs/num_jobs
  
  utils/split_data.sh $data_uniform_seg $nj

  mkdir -p $dir/docs/split$nj/
  
  # First compute IDF stats
  $cmd $dir/log/compute_source_idf_stats.log \
    steps/cleanup/internal/compute_tf_idf.py \
    --tf-weighting-scheme="raw" \
    --idf-weighting-scheme="log" \
    --output-idf-stats=$dir/docs/idf_stats.txt \
    $dir/docs/docs.txt $dir/docs/src_tf_idf.txt
  
  # Split documents so that they can be accessed easily by parallel jobs.
  $cmd JOB=1:$nj $dir/docs/log/split_docs.JOB.log \
    mkdir -p $dir/docs/split$nj/JOB '&&' \
    utils/data/get_reco2utt.sh $data_uniform_seg/split$nj/JOB '&&' \
    utils/filter_scp.pl $data_uniform_seg/split$nj/JOB/reco2utt $dir/docs/text2doc \| \
    utils/spk2utt_to_utt2spk.pl \| \
    utils/filter_scp.pl /dev/stdin $dir/docs/docs.txt '>' \
    $dir/docs/split$nj/JOB/docs.txt

  # Compute TF-IDF for the source documents. 
  $cmd JOB=1:$nj $dir/docs/log/get_tfidf_for_source_texts.JOB.log \
    steps/cleanup/internal/compute_tf_idf.py \
      --tf-weighting-scheme="raw" \
      --idf-weighting-scheme="log" \
      --input-idf-stats=$dir/docs/idf_stats.txt \
      $dir/docs/split$nj/JOB/docs.txt $dir/docs/split$nj/JOB/src_tf_idf.txt 
fi

if [ $stage -le 8 ]; then
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

if [ $stage -le 9 ]; then
  for n in `seq $nj`; do
    awk '{print ($1" "$1" 1")}' $data_uniform_seg/split$nj/$n/utt2spk > \
      $dir/lats/fake_reco2file_and_channel.$n
  done

  $cmd JOB=1:$nj $dir/lats/log/compute_query_tf_idf.JOB.log \
    steps/cleanup/internal/ctm_to_text.pl --non-scored-words $dir/non_scored_words.txt \
      $dir/lats/score_$lmwt/${data_id}_uniform_seg.ctm.JOB \| \
    steps/cleanup/internal/compute_tf_idf.py \
      --tf-weighting-scheme="normalized" \
      --idf-weighting-scheme="log" \
      --input-idf-stats=$dir/docs/idf_stats.txt \
      --accumulate-over-docs=false \
      - $dir/docs/split$nj/JOB/query_tf_idf.txt 
  
  # The relevant documents can be found using TF-IDF similarity and nearby
  # documents can also be picked for the Smith-Waterman alignment stage.

  cut -d ' ' -f 1,2 $data_uniform_seg/segments > $data_uniform_seg/utt2reco

  $cmd JOB=1:$nj $dir/lats/log/retrieve_similar_docs.JOB.log \
    steps/cleanup/internal/retrieve_similar_docs.py \
      --query-tfidf=$dir/docs/split$nj/JOB/query_tf_idf.txt \
      --source-tfidf=$dir/docs/split$nj/JOB/src_tf_idf.txt \
      --source-text-id2doc-ids=$dir/docs/text2doc \
      --query-id2source-text-id=$data_uniform_seg/utt2reco \
      --num-neighbors-to-search=$num_neighbors_to_search \
      --neighbor-tfidf-threshold=$neighbor_tfidf_threshold \
      --relevant-docs=$dir/docs/split$nj/JOB/relevant_docs.txt
  
  $cmd JOB=1:$nj $dir/lats/log/get_ctm_edits.JOB.log \
    steps/cleanup/internal/stitch_documents.py \
      --query2docs=$dir/docs/split$nj/JOB/relevant_docs.txt \
      --input-documents=$dir/docs/split$nj/JOB/docs.txt \
      --output-documents=- \| \
    steps/cleanup/internal/align_ctm_ref.py --eps-symbol='"<eps>"' \
      --oov-word="'`cat $lang/oov.txt`'" --symbol-table=$lang/words.txt \
      --hyp-format=CTM --align-full-hyp=$align_full_hyp \
      --reco2file-and-channel=$dir/lats/fake_reco2file_and_channel.JOB \
      --hyp=$dir/lats/score_$lmwt/${data_id}_uniform_seg.ctm.JOB --ref=- \
      --output=$dir/lats/score_$lmwt/${data_id}_uniform_seg.ctm_edits.JOB 
  
  for n in `seq $nj`; do
    cat $dir/lats/score_$lmwt/${data_id}_uniform_seg.ctm_edits.$n 
  done > $dir/lats/score_$lmwt/ctm_edits
  
fi

if [ $stage -le 10 ]; then
  steps/cleanup/internal/resolve_ctm_edits_overlaps.py \
    ${data_uniform_seg}/segments $dir/lats/score_$lmwt/ctm_edits $dir/ctm_edits
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

  utils/data/get_subsegmented_feats.sh $data_uniform_seg/feats.scp 0.01 0.015 \
    $dir/segments | \
    utils/data/fix_subsegmented_feats.pl $out_data/utt2max_frames > \
    $out_data/feats.scp 
fi
