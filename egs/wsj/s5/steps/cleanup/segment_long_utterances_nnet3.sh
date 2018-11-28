#!/bin/bash

# Copyright 2014  Guoguo Chen
#           2016  Vimal Manohar
# Apache 2.0

# This script is similar to steps/cleanup/segment_long_utterances.sh, but 
# uses nnet3 acoustic model instead of GMM acoustic model for decoding.
# This script performs segmentation of the input data based on the transcription
# and outputs segmented data along with the corresponding aligned transcription.
# The purpose of this script is to divide up the input data (which may consist
# of long recordings such as television shows or audiobooks) into segments which
# are of manageable length for further processing, along with the portion of the
# transcript that seems to match (aligns with) each segment.
# This the light-supervised training scenario where the input transcription is
# not expected to be completely clean and may have significant errors. 
# See "JHU Kaldi System for Arabic MGB-3 ASR Challenge using Diarization,
# Audio-transcript Alignment and Transfer Learning": Vimal Manohar, Daniel
# Povey, Sanjeev Khudanpur, ASRU 2017
# (http://www.danielpovey.com/files/2017_asru_mgb3.pdf) for details.
# The output data is not necessarily particularly clean; you can run
# steps/cleanup/clean_and_segment_data_nnet3.sh on the output in order to
# further clean it and eliminate data where the transcript doesn't seem to
# match.


set -e
set -o pipefail
set -u

stage=-1
cmd=run.pl
nj=4

# Uniform segmentation options
max_segment_duration=30
overlap_duration=5
seconds_per_spk_max=30

# Decode options
graph_opts=
beam=15.0
lattice_beam=1.0
lmwt=10

acwt=0.1  # Just a default value, used for adaptation and beam-pruning..
post_decode_acwt=1.0  # can be used in 'chain' systems to scale acoustics by 10 so the
                      # regular scoring script works.

# Contexts must ideally match training
extra_left_context=0  # Set to some large value, typically 40 for LSTM (must match training)
extra_right_context=0  
extra_left_context_initial=-1
extra_right_context_final=-1
frames_per_chunk=150

# i-vector options
extractor=    # i-Vector extractor. If provided, will extract i-vectors. 
              # Required if the network was trained with i-vector extractor. 
use_vad=false # Use energy-based VAD for i-vector extraction

# TF-IDF similarity search options
max_words=1000
num_neighbors_to_search=1   # Number of neighboring documents to search around the one retrieved based on maximum tf-idf similarity.
neighbor_tfidf_threshold=0.5

align_full_hyp=false  # Align full hypothesis i.e. trackback from the end to get the alignment.

# First-pass segmentation opts
# These options are passed to the script
# steps/cleanup/internal/segment_ctm_edits_mild.py
segmentation_extra_opts=
min_split_point_duration=0.1
max_deleted_words_kept_when_merging=1
max_wer=50
max_segment_length_for_merging=60
max_bad_proportion=0.75
max_intersegment_incorrect_words_length=1
max_segment_length_for_splitting=10
hard_max_segment_length=15
min_silence_length_to_split_at=0.3
min_non_scored_length_to_split_at=0.3


. ./path.sh
. utils/parse_options.sh

if [ $# -ne 5 ] && [ $# -ne 7 ]; then
  cat <<EOF
Usage: $0 [--extractor <ivector-extractor>] [options] <model-dir> <lang> <data-in> [<text-in> <utt2text>] <segmented-data-out> <work-dir>
 e.g.: $0 exp/wsj_tri2b data/lang_nosp data/train_long data/train_long/text data/train_reseg exp/segment_wsj_long_utts_train
This script performs segmentation of the data in <data-in> and writes out the
segmented data (with a segments file) to
<segmented-data-out> along with the corresponding aligned transcription.
Note: If <utt2text> is not provided, the "text" file in <data-in> is used as the
raw transcripts to train biased LM for the utterances.
If <utt2text> is provided, then it should be a mapping from the utterance-ids in
<data-in> to the transcript-keys in the file <text-in>, which will be
used to train biased LMs for the utterances.
The purpose of this script is to divide up the input data (which may consist of
long recordings such as television shows or audiobooks) into segments which are
of manageable length for further processing, along with the portion of the
transcript that seems to match each segment.
The output data is not necessarily particularly clean; you are advised to run
steps/cleanup/clean_and_segment_data.sh on the output in order to further clean
it and eliminate data where the transcript doesn't seem to match.
  main options (for others, see top of script file):
    --stage <n>             # stage to run from, to enable resuming from partially
                            # completed run (default: 0)
    --cmd '$cmd'            # command to submit jobs with (e.g. run.pl, queue.pl)
    --nj <n>                # number of parallel jobs to use in graph creation and
                            # decoding
    --graph-opts 'opts'         # Additional options to make_biased_lm_graphs.sh.
                                # Please run steps/cleanup/make_biased_lm_graphs.sh
                                # without arguments to see allowed options.
    --segmentation-extra-opts 'opts'  # Additional options to segment_ctm_edits_mild.py.
                                # Please run steps/cleanup/internal/segment_ctm_edits_mild.py
                                # without arguments to see allowed options.
    --align-full-hyp <true|false>  # If true, align full hypothesis 
                                   i.e. trackback from the end to get the alignment. 
                                   This is different from the normal 
                                   Smith-Waterman alignment, where the
                                   traceback will be from the maximum score.
    --extractor <extractor>     # i-vector extractor directory if i-vector is 
                                # to be used during decoding. Must match
                                # the extractor used for training neural-network.
    --use-vad <true|false>      # If true, uses energy-based VAD to apply frame weights
                                # for i-vector stats extraction
EOF
  exit 1
fi

srcdir=$1
lang=$2
data=$3

extra_files=
utt2text=
text=$data/text
if [ $# -eq 7 ]; then
  text=$4
  utt2text=$5
  out_data=$6
  dir=$7
  extra_files="$utt2text"
else
  out_data=$4
  dir=$5
fi

if [ ! -z "$extractor" ]; then
  extra_files="$extra_files $extractor/final.ie"
fi

for f in $data/feats.scp $text $extra_files $srcdir/tree \
  $srcdir/final.mdl $srcdir/cmvn_opts; do
  if [ ! -f $f ]; then
    echo "$0: Could not find file $f"
    exit 1
  fi
done

data_id=`basename $data`
mkdir -p $dir
cp $srcdir/final.mdl $dir
cp $srcdir/tree $dir
cp $srcdir/cmvn_opts $dir
cp $srcdir/{splice_opts,delta_opts,final.mat,final.alimdl} $dir 2>/dev/null || true
cp $srcdir/frame_subsampling_factor $dir 2>/dev/null || true

utils/lang/check_phones_compatible.sh $lang/phones.txt $srcdir/phones.txt
cp $lang/phones.txt $dir

data_uniform_seg=$dir/${data_id}_uniform_seg

# First we split the data into segments of around 30s long, on which
# it would be possible to do a decoding.
# A diarization step will be added in the future.
if [ $stage -le 1 ]; then
  echo "$0: Stage 1 (Splitting data directory $data into uniform segments)"

  utils/data/get_utt2dur.sh $data
  if [ ! -f $data/segments ]; then
    utils/data/get_segments_for_data.sh $data > $data/segments
  fi

  utils/data/get_uniform_subsegments.py \
    --max-segment-duration=$max_segment_duration \
    --overlap-duration=$overlap_duration \
    --max-remaining-duration=$(perl -e "print $max_segment_duration / 2.0") \
    $data/segments > $dir/uniform_sub_segments
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

  utils/fix_data_dir.sh $data_uniform_seg

  # Compute new cmvn stats for the segmented data directory
  steps/compute_cmvn_stats.sh $data_uniform_seg/
fi

graph_dir=$dir/graphs_uniform_seg

if [ $stage -le 3 ]; then
  echo "$0: Stage 3 (Building biased-language-model decoding graphs)"

  mkdir -p $graph_dir

  # Make graphs w.r.t. to the original text (usually recording-level)
  steps/cleanup/make_biased_lm_graphs.sh $graph_opts \
    --nj $nj --cmd "$cmd" $text \
    $lang $dir $dir/graphs
  if [ -z "$utt2text" ]; then
    # and then copy it to the sub-segments.
    cat $dir/uniform_sub_segments | awk '{print $1" "$2}' | \
      utils/apply_map.pl -f 2 $dir/graphs/HCLG.fsts.scp | \
      sort -k1,1 > \
      $graph_dir/HCLG.fsts.scp
  else
    # and then copy it to the sub-segments.
    cat $dir/uniform_sub_segments | awk '{print $1" "$2}' | \
      utils/apply_map.pl -f 2 $utt2text | \
      utils/apply_map.pl -f 2 $dir/graphs/HCLG.fsts.scp | \
      sort -k1,1 > \
      $graph_dir/HCLG.fsts.scp
  fi

  cp $lang/words.txt $graph_dir
  cp -r $lang/phones $graph_dir
  [ -f $dir/graphs/num_pdfs ] && cp $dir/graphs/num_pdfs $graph_dir/
fi

decode_dir=$dir/lats
mkdir -p $decode_dir

online_ivector_dir=
if [ ! -z "$extractor" ]; then
  online_ivector_dir=$dir/ivectors_$(basename $data_uniform_seg)

  if [ $stage -le 4 ]; then
    # Compute energy-based VAD
    if $use_vad; then
      steps/compute_vad_decision.sh $data_uniform_seg \
        $data_uniform_seg/log $data_uniform_seg/data
    fi

    steps/online/nnet2/extract_ivectors_online.sh \
      --nj $nj --cmd "$cmd --mem 4G" --use-vad $use_vad \
      $data_uniform_seg $extractor $online_ivector_dir
  fi
fi

if [ $stage -le 5 ]; then
  echo "$0: Decoding with biased language models..."

  steps/cleanup/decode_segmentation_nnet3.sh \
    --acwt $acwt --post-decode-acwt $post_decode_acwt \
    --beam $beam --lattice-beam $lattice_beam --nj $nj --cmd "$cmd --mem 4G" \
    --skip-scoring true --allow-partial false \
    --extra-left-context $extra_left_context \
    --extra-right-context $extra_right_context \
    --extra-left-context-initial $extra_left_context_initial \
    --extra-right-context-final $extra_right_context_final \
    --frames-per-chunk $frames_per_chunk \
    ${online_ivector_dir:+--online-ivector-dir $online_ivector_dir} \
    $graph_dir $data_uniform_seg $decode_dir
fi

frame_shift_opt=
if [ -f $srcdir/frame_subsampling_factor ]; then
  frame_shift_opt="--frame-shift 0.0$(cat $srcdir/frame_subsampling_factor)"
fi

if [ $stage -le 6 ]; then
  steps/get_ctm_fast.sh --lmwt $lmwt --cmd "$cmd --mem 4G" \
    --print-silence true $frame_shift_opt \
    $data_uniform_seg $lang $decode_dir $decode_dir/ctm_$lmwt
fi

# Split the original text into documents, over which we can do
# searching reasonably efficiently. Also get a mapping from the original
# text to the created documents (i.e. text2doc)
# Since the Smith-Waterman alignment is linear in the length of the
# text, we want to keep it reasonably small (a few thousand words).

if [ $stage -le 7 ]; then
  # Split the reference text into documents.
  mkdir -p $dir/docs

  # text2doc is a mapping from the original transcript to the documents
  # it is split into.
  # The format is
  # <original-transcript> <doc1> <doc2> ...
  steps/cleanup/internal/split_text_into_docs.pl --max-words $max_words \
    $text $dir/docs/doc2text $dir/docs/docs.txt
  utils/utt2spk_to_spk2utt.pl $dir/docs/doc2text > $dir/docs/text2doc
fi

if [ $stage -le 8 ]; then
  # Get TF-IDF for the reference documents.
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
  mkdir -p $dir/docs/split$nj/
  sdir=$dir/docs/split$nj
  for n in `seq $nj`; do

    # old2new_utts is a mapping from the original segments to the
    # new segments created by uniformly segmenting.
    # The format is <old-utterance> <new-utt1> <new-utt2> ...
    utils/filter_scp.pl $data_uniform_seg/split$nj/$n/utt2spk $dir/uniform_sub_segments | \
      cut -d ' ' -f 1,2 | utils/utt2spk_to_spk2utt.pl > $sdir/old2new_utts.$n.txt

    if [ ! -z "$utt2text" ]; then
      # utt2text, if provided, is a mapping from the <old-utterance> to
      # <original-transript>.
      # Since text2doc is mapping from <original-transcript> to documents, we
      # first have to find the original-transcripts that are in the current
      # split.
      utils/filter_scp.pl $sdir/old2new_utts.$n.txt $utt2text | \
        cut -d ' ' -f 2 | sort -u | \
        utils/filter_scp.pl /dev/stdin $dir/docs/text2doc > $sdir/text2doc.$n
    else
      utils/filter_scp.pl $sdir/old2new_utts.$n.txt \
        $dir/docs/text2doc > $sdir/text2doc.$n
    fi

    utils/spk2utt_to_utt2spk.pl $sdir/text2doc.$n | \
      utils/filter_scp.pl /dev/stdin $dir/docs/docs.txt > \
      $sdir/docs.$n.txt
  done

  # Compute TF-IDF for the source documents.
  $cmd JOB=1:$nj $dir/docs/log/get_tfidf_for_source_texts.JOB.log \
    steps/cleanup/internal/compute_tf_idf.py \
      --tf-weighting-scheme="raw" \
      --idf-weighting-scheme="log" \
      --input-idf-stats=$dir/docs/idf_stats.txt \
      $sdir/docs.JOB.txt $sdir/src_tf_idf.JOB.txt

  sdir=$dir/docs/split$nj
  # Make $sdir an absolute pathname.
  sdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $sdir ${PWD}`

  for n in `seq $nj`; do
    awk -v f="$sdir/src_tf_idf.$n.txt" '{print $1" "f}' \
      $sdir/text2doc.$n
  done | perl -ane 'BEGIN { %tfidfs = (); }
  {
    if (!defined $tfidfs{$F[0]}) {
      $tfidfs{$F[0]} = $F[1];
    }
  }
  END {
  while(my ($k, $v) = each %tfidfs) {
    print "$k $v\n";
  } }' > $dir/docs/source2tf_idf.scp
fi

if [ $stage -le 9 ]; then
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

if [ $stage -le 10 ]; then
  sdir=$dir/query_docs/split$nj
  mkdir -p $sdir

  # Compute TF-IDF for the query documents (decode hypotheses).
  # The output is an archive of TF-IDF indexed by the query.
  $cmd JOB=1:$nj $decode_dir/ctm_$lmwt/log/compute_query_tf_idf.JOB.log \
    steps/cleanup/internal/ctm_to_text.pl --non-scored-words $dir/non_scored_words.txt \
      $decode_dir/ctm_$lmwt/ctm.JOB \| \
    steps/cleanup/internal/compute_tf_idf.py \
      --tf-weighting-scheme="normalized" \
      --idf-weighting-scheme="log" \
      --input-idf-stats=$dir/docs/idf_stats.txt \
      --accumulate-over-docs=false \
      - $sdir/query_tf_idf.JOB.ark.txt

  # The relevant documents can be found using TF-IDF similarity and nearby
  # documents can also be picked for the Smith-Waterman alignment stage.

  # Get a mapping from the new utterance-ids to original transcripts
  if [ -z "$utt2text" ]; then
    awk '{print $1" "$2}' $dir/uniform_sub_segments > \
      $dir/new2orig_utt
  else
    awk '{print $1" "$2}' $dir/uniform_sub_segments | \
      utils/apply_map.pl -f 2 $utt2text > \
      $dir/new2orig_utt
  fi

  # The query TF-IDFs are all indexed by the utterance-id of the sub-segments.
  # The source TF-IDFs use the document-ids created by splitting the reference
  # text into documents.
  # For each query, we need to retrieve the documents that were created from
  # the same original utterance that the sub-segment was from. For this,
  # we have to load the source TF-IDF that has those documents. This
  # information is provided using the option --source-text-id2tf-idf-file.
  # The output of this script is a file where the first column is the
  # query-id (i.e. sub-segment-id) and the remaining columns, which is at least
  # one in number and a maxmium of (1 + 2 * num-neighbors-to-search) columns
  # is the document-ids for the retrieved documents.
  $cmd JOB=1:$nj $dir/log/retrieve_similar_docs.JOB.log \
    steps/cleanup/internal/retrieve_similar_docs.py \
      --query-tfidf=$dir/query_docs/split$nj/query_tf_idf.JOB.ark.txt \
      --source-text-id2tfidf=$dir/docs/source2tf_idf.scp \
      --source-text-id2doc-ids=$dir/docs/text2doc \
      --query-id2source-text-id=$dir/new2orig_utt \
      --num-neighbors-to-search=$num_neighbors_to_search \
      --neighbor-tfidf-threshold=$neighbor_tfidf_threshold \
      --relevant-docs=$dir/query_docs/split$nj/relevant_docs.JOB.txt

  $cmd JOB=1:$nj $decode_dir/ctm_$lmwt/log/get_ctm_edits.JOB.log \
    steps/cleanup/internal/stitch_documents.py \
      --query2docs=$dir/query_docs/split$nj/relevant_docs.JOB.txt \
      --input-documents=$dir/docs/split$nj/docs.JOB.txt \
      --output-documents=- \| \
    steps/cleanup/internal/align_ctm_ref.py --eps-symbol='"<eps>"' \
      --oov-word="'`cat $lang/oov.txt`'" --symbol-table=$lang/words.txt \
      --hyp-format=CTM --align-full-hyp=$align_full_hyp \
      --hyp=$decode_dir/ctm_$lmwt/ctm.JOB --ref=- \
      --output=$decode_dir/ctm_$lmwt/ctm_edits.JOB

  for n in `seq $nj`; do
    cat $decode_dir/ctm_$lmwt/ctm_edits.$n
  done > $decode_dir/ctm_$lmwt/ctm_edits

fi

if [ $stage -le 11 ]; then
  $cmd $dir/log/resolve_ctm_edits.log \
    steps/cleanup/internal/resolve_ctm_edits_overlaps.py \
    ${data_uniform_seg}/segments $decode_dir/ctm_$lmwt/ctm_edits $dir/ctm_edits
fi

if [ $stage -le 12 ]; then
  echo "$0: modifying ctm-edits file to allow repetitions [for dysfluencies] and "
  echo "   ... to fix reference mismatches involving non-scored words. "

  $cmd $dir/log/modify_ctm_edits.log \
    steps/cleanup/internal/modify_ctm_edits.py --verbose=3 $dir/non_scored_words.txt \
    $dir/ctm_edits $dir/ctm_edits.modified

  echo "   ... See $dir/log/modify_ctm_edits.log for details and stats, including"
  echo " a list of commonly-repeated words."
fi

if [ $stage -le 13 ]; then
  echo "$0: applying 'taint' markers to ctm-edits file to mark silences and"
  echo "  ... non-scored words that are next to errors."
  $cmd $dir/log/taint_ctm_edits.log \
       steps/cleanup/internal/taint_ctm_edits.py --remove-deletions=false \
       $dir/ctm_edits.modified $dir/ctm_edits.tainted
  echo "... Stats, including global cor/ins/del/sub stats, are in $dir/log/taint_ctm_edits.log."
fi

if [ $stage -le 14 ]; then
  echo "$0: creating segmentation from ctm-edits file."

  segmentation_opts=(
  --min-split-point-duration=$min_split_point_duration
  --max-deleted-words-kept-when-merging=$max_deleted_words_kept_when_merging
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
      ${segmentation_opts[@]} $segmentation_extra_opts \
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
if [ $stage -le 15 ]; then
  utils/data/subsegment_data_dir.sh $data_uniform_seg \
    $dir/segments $dir/text $out_data
fi
