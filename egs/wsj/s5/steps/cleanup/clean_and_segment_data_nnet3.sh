#!/bin/bash

# Copyright 2016  Vimal Manohar
#           2016  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

# This script demonstrates how to re-segment training data selecting only the
# "good" audio that matches the transcripts.
# This script is like clean_and_segment_data.sh, but uses nnet3 model instead of
# a GMM for decoding.
# The basic idea is to decode with an existing in-domain nnet3 acoustic model,
# and a biased language model built from the reference transcript, and then work
# out the segmentation from a ctm like file.

set -e
set -o pipefail
set -u

stage=0

cmd=run.pl
cleanup=true  # remove temporary directories and files
nj=4
# Decode options
graph_opts=
beam=15.0
lattice_beam=1.0

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

segmentation_opts=

. ./path.sh
. utils/parse_options.sh


if [ $# -ne 5 ]; then
  cat <<EOF
  Usage: $0 [--extractor <ivector-extractor>] [options] <data> <lang> <srcdir> <dir> <cleaned-data>
   This script does data cleanup to remove bad portions of transcripts and
   may do other minor modifications of transcripts such as allowing repetitions
   for disfluencies, and adding or removing non-scored words (by default:
   words that map to 'silence phones')
   Note: <srcdir> is expected to contain a nnet3-based model.
   <ivector-extractor> and decoding options like --extra-left-context must match
   the appropriate options used for training.

  e.g. $0 data/train data/lang exp/tri3 exp/tri3_cleanup data/train_cleaned
  main options (for others, see top of script file):
    --stage <n>             # stage to run from, to enable resuming from partially
                            # completed run (default: 0)
    --cmd '$cmd'            # command to submit jobs with (e.g. run.pl, queue.pl)
    --nj <n>                # number of parallel jobs to use in graph creation and
                            # decoding
    --graph-opts 'opts'         # Additional options to make_biased_lm_graphs.sh.
                                # Please run steps/cleanup/make_biased_lm_graphs.sh
                                # without arguments to see allowed options.
    --segmentation-opts 'opts'  # Additional options to segment_ctm_edits.py.
                                # Please run steps/cleanup/internal/segment_ctm_edits.py
                                # without arguments to see allowed options.
    --cleanup        <true|false>  # Clean up intermediate files afterward.  Default true.
    --extractor <extractor>     # i-vector extractor directory if i-vector is
                                # to be used during decoding. Must match
                                # the extractor used for training neural-network.
    --use-vad <true|false>      # If true, uses energy-based VAD to apply frame weights
                                # for i-vector stats extraction
EOF
  exit 1
fi

data=$1
lang=$2
srcdir=$3
dir=$4
data_out=$5


extra_files=
if [ ! -z "$extractor" ]; then
  extra_files="$extractor/final.ie"
fi

for f in $srcdir/{final.mdl,tree,cmvn_opts} $data/utt2spk $data/feats.scp \
  $lang/words.txt $lang/oov.txt $extra_files; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist."
    exit 1
  fi
done

mkdir -p $dir
cp $srcdir/final.mdl $dir
cp $srcdir/tree $dir
cp $srcdir/cmvn_opts $dir
cp $srcdir/{splice_opts,delta_opts,final.mat,final.alimdl} $dir 2>/dev/null || true
cp $srcdir/frame_subsampling_factor $dir 2>/dev/null || true

utils/lang/check_phones_compatible.sh $lang/phones.txt $srcdir/phones.txt
cp $lang/phones.txt $dir

if [ $stage -le 1 ]; then
  echo "$0: Building biased-language-model decoding graphs..."


  steps/cleanup/make_biased_lm_graphs.sh $graph_opts \
    --nj $nj --cmd "$cmd" \
     $data $lang $dir $dir/graphs
fi

online_ivector_dir=
if [ ! -z "$extractor" ]; then
  online_ivector_dir=$dir/ivectors_$(basename $data)

  if [ $stage -le 2 ]; then
    # Compute energy-based VAD
    if $use_vad; then
      steps/compute_vad_decision.sh $data \
        $data/log $data/data
    fi

    steps/online/nnet2/extract_ivectors_online.sh \
      --nj $nj --cmd "$cmd --mem 4G" --use-vad $use_vad \
      $data $extractor $online_ivector_dir
  fi
fi

if [ $stage -le 3 ]; then
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
    $dir/graphs $data $dir/lats

  # the following is for diagnostics, e.g. it will give us the lattice depth.
  steps/diagnostic/analyze_lats.sh --cmd "$cmd" $lang $dir/lats
fi

frame_shift_opt=
if [ -f $srcdir/frame_subsampling_factor ]; then
  frame_shift_opt="--frame-shift 0.0$(cat $srcdir/frame_subsampling_factor)"
fi

if [ $stage -le 4 ]; then
  echo "$0: Doing oracle alignment of lattices..."
  steps/cleanup/lattice_oracle_align.sh --cmd "$cmd --mem 4G" $frame_shift_opt \
    $data $lang $dir/lats $dir/lattice_oracle
fi


if [ $stage -le 4 ]; then
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

if [ $stage -le 5 ]; then
  echo "$0: modifying ctm-edits file to allow repetitions [for dysfluencies] and "
  echo "   ... to fix reference mismatches involving non-scored words. "

  $cmd $dir/log/modify_ctm_edits.log \
    steps/cleanup/internal/modify_ctm_edits.py --verbose=3 $dir/non_scored_words.txt \
    $dir/lattice_oracle/ctm_edits $dir/ctm_edits.modified

  echo "   ... See $dir/log/modify_ctm_edits.log for details and stats, including"
  echo " a list of commonly-repeated words."
fi

if [ $stage -le 6 ]; then
  echo "$0: applying 'taint' markers to ctm-edits file to mark silences and"
  echo "  ... non-scored words that are next to errors."
  $cmd $dir/log/taint_ctm_edits.log \
       steps/cleanup/internal/taint_ctm_edits.py $dir/ctm_edits.modified $dir/ctm_edits.tainted
  echo "... Stats, including global cor/ins/del/sub stats, are in $dir/log/taint_ctm_edits.log."
fi


if [ $stage -le 7 ]; then
  echo "$0: creating segmentation from ctm-edits file."

  $cmd $dir/log/segment_ctm_edits.log \
    steps/cleanup/internal/segment_ctm_edits.py \
      $segmentation_opts \
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

if [ $stage -le 8 ]; then
  echo "$0: working out required segment padding to account for feature-generation edge effects."
  # make sure $data/utt2dur exists.
  utils/data/get_utt2dur.sh $data
  # utt2dur.from_ctm contains lines of the form 'utt dur',  e.g.
  # AMI_EN2001a_H00_MEE068_0000557_0000594 0.35
  # where the times are ultimately derived from the num-frames in the features.
  cat $dir/lattice_oracle/ctm_edits | \
     awk '{utt=$1; t=$3+$4; if (t > dur[$1]) dur[$1] = t; } END{for (k in dur) print k, dur[k];}' | \
     sort > $dir/utt2dur.from_ctm
  # the apply_map command below gives us lines of the form 'utt dur-from-$data/utt2dur dur-from-utt2dur.from_ctm',
  # e.g. AMI_EN2001a_H00_MEE068_0000557_0000594 0.37 0.35
  utils/apply_map.pl -f 1 <(awk '{print $1,$1,$2}' <$data/utt2dur) <$dir/utt2dur.from_ctm  | \
    awk '{printf("%.3f\n", $2 - $3); }' | sort | uniq -c | sort -nr > $dir/padding_frequencies
  # there are values other than the most-frequent one (0.02) in there because
  # of wav files that were shorter than the segment info.
  padding=$(head -n 1 $dir/padding_frequencies | awk '{print $2}')
  echo "$0: we'll pad segments with $padding seconds at segment ends to correct for feature-generation end effects"
  echo $padding >$dir/segment_end_padding
fi


if [ $stage -le 8 ]; then
  echo "$0: based on the segments and text file in $dir/segments and $dir/text, creating new data-dir in $data_out"
  padding=$(cat $dir/segment_end_padding)  # e.g. 0.02
  utils/data/subsegment_data_dir.sh --segment-end-padding $padding ${data} $dir/segments $dir/text $data_out
  # utils/data/subsegment_data_dir.sh can output directories that have e.g. to many entries left in wav.scp
  # Clean this up with the fix_dat_dir.sh script
  utils/fix_data_dir.sh $data_out
fi

if [ $stage -le 9 ]; then
  echo "$0: recomputing CMVN stats for the new data"
  # Caution: this script puts the CMVN stats in $data_out/data,
  # e.g. data/train_cleaned/data.  This is not the general pattern we use.
  steps/compute_cmvn_stats.sh $data_out $data_out/log $data_out/data
fi

if $cleanup; then
  echo "$0: cleaning up intermediate files"
  rm -r $dir/graphs/fsts $dir/graphs/HCLG.fsts.scp || true
  rm -r $dir/lats/lat.*.gz $dir/lats/split_fsts || true
  rm $dir/lattice_oracle/lat.*.gz || true
fi

echo "$0: done."
