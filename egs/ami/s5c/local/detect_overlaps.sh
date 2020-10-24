#!/usr/bin/env bash

# Copyright 2020  Desh Raj (Johns Hopkins University)
# Apache 2.0.

# This script does nnet3-based overlap detection given an input 
# kaldi data directory and outputs an overlap RTTM file.

set -e 
set -o pipefail
set -u

if [ -f ./path.sh ]; then . ./path.sh; fi

nj=32
cmd=run.pl
stage=0
region_type=overlap
convert_data_dir_to_whole=false

output_name=output   # The output node in the network
output_scale= # provide scaling factors for "silence single overlap" (tune on dev set)

# Network config
iter=final  # Model iteration to use

# Contexts must ideally match training for LSTM models, but
# may not necessarily for stats components
extra_left_context=0  # Set to some large value, typically 40 for LSTM (must match training)
extra_right_context=0  
extra_left_context_initial=-1
extra_right_context_final=-1
frames_per_chunk=300

# Decoding options
graph_opts="--min-silence-duration=0 --min-speech-duration=0.03 --max-speech-duration=10.0 --min-overlap-duration 0.1 --max-overlap-duration 5.0"
acwt=0.1

# Postprocessing options
segment_padding=0.05   # Duration (in seconds) of padding added to overlap segments
min_segment_dur=0   # Minimum duration (in seconds) required for a segment to be included
                    # This is before any padding. Segments shorter than this duration will be removed.
                    # This is an alternative to --min-overlap-duration above.
merge_consecutive_max_dur=inf   # Merge consecutive segments as long as the merged segment is no longer than this many
                              # seconds. The segments are only merged if their boundaries are touching.
                              # This is after padding by --segment-padding seconds.
                              # 0 means do not merge. Use 'inf' to not limit the duration.

echo $* 

. utils/parse_options.sh

if [ $# -ne 3 ]; then
  echo "This script does nnet3-based overlap detection given an input kaldi "
  echo "data directory and outputs an RTTM file."
  echo "See script for details of the options to be supplied."
  echo "Usage: $0 <src-data-dir> <nnet-dir> <out-dir>"
  echo " e.g.: $0 data/dev exp/overlap_1a/tdnn_stats_1a exp/overlap_1a/dev"
  echo ""
  echo "Options: "
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --nj <num-job>                                 # number of parallel jobs to run."
  echo "  --stage <stage>                                # stage to do partial re-run from."
  echo "  --output-name <name>    # The output node in the network"
  echo "  --extra-left-context  <context|0>   # Set to some large value, typically 40 for LSTM (must match training)"
  echo "  --extra-right-context  <context|0>   # For BLSTM or statistics pooling"
  exit 1
fi

data_dir=$1   # The input data directory.
nnet_dir=$2   # The overlap detection neural network
out_dir=$3    # The output data directory

data_id=`basename $data_dir`
overlap_dir=${out_dir}/overlap # working directory

test_data_dir=${data_dir}
if [ $convert_data_dir_to_whole == "true" ]; then
  test_data_dir=${data_dir}_whole
  if ! [ -d $test_data_dir ]; then
    utils/data/convert_data_dir_to_whole.sh $data_dir $test_data_dir
    utils/fix_data_dir.sh $test_data_dir
    num_wavs=$(wc -l < "$data_dir"/wav.scp)
    if [ $nj -gt $num_wavs ]; then
      nj=$num_wavs
    fi
    steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --nj $nj --cmd "$cmd" \
      --write-utt2num-frames true ${test_data_dir}
    steps/compute_cmvn_stats.sh ${test_data_dir}
    utils/fix_data_dir.sh ${test_data_dir}
  fi
fi

num_wavs=$(wc -l < "$data_dir"/wav.scp)
if [ $nj -gt $num_wavs ]; then
  nj=$num_wavs
fi

###############################################################################
## Forward pass through the network network and dump the log-likelihoods.
###############################################################################

frame_subsampling_factor=1
if [ -f $nnet_dir/frame_subsampling_factor ]; then
  frame_subsampling_factor=$(cat $nnet_dir/frame_subsampling_factor)
fi

mkdir -p $overlap_dir
if [ $stage -le 1 ]; then
  if [ "$(utils/make_absolute.sh $nnet_dir)" != "$(utils/make_absolute.sh $overlap_dir)" ]; then
    cp $nnet_dir/cmvn_opts $overlap_dir || exit 1
  fi

  ########################################################################
  ## Initialize neural network for decoding using the output $output_name
  ########################################################################

  if [ ! -z "$output_name" ] && [ "$output_name" != output ]; then
    $cmd $out_dir/log/get_nnet_${output_name}.log \
      nnet3-copy --edits="rename-node old-name=$output_name new-name=output" \
      $nnet_dir/$iter.raw $overlap_dir/${iter}_${output_name}.raw || exit 1
    iter=${iter}_${output_name}
  else 
    if ! diff $nnet_dir/$iter.raw $out_dir/$iter.raw; then
      cp $nnet_dir/$iter.raw $overlap_dir/
    fi
  fi

  steps/nnet3/compute_output.sh --nj $nj --cmd "$cmd" \
    --iter ${iter} \
    --extra-left-context $extra_left_context \
    --extra-right-context $extra_right_context \
    --extra-left-context-initial $extra_left_context_initial \
    --extra-right-context-final $extra_right_context_final \
    --frames-per-chunk $frames_per_chunk --apply-exp true \
    --frame-subsampling-factor $frame_subsampling_factor \
    ${test_data_dir} $overlap_dir $out_dir || exit 1
fi

###############################################################################
## Prepare FST we search to make overlap decisions.
###############################################################################

utils/data/get_utt2dur.sh --nj $nj --cmd "$cmd" $test_data_dir || exit 1
frame_shift=$(utils/data/get_frame_shift.sh $test_data_dir) || exit 1

graph_dir=${overlap_dir}/graph_${output_name}
if [ $stage -le 2 ]; then
  mkdir -p $graph_dir

  # 0 for silence, 1 for single speaker, and 2 for overlap
  cat <<EOF > $graph_dir/words.txt
<eps> 0
silence 1
single 2
overlap 3
EOF

  $cmd $graph_dir/log/make_graph.log \
    steps/overlap/prepare_overlap_graph.py $graph_opts \
      --frame-shift=$(perl -e "print $frame_shift * $frame_subsampling_factor") - \| \
    fstcompile --isymbols=$graph_dir/words.txt --osymbols=$graph_dir/words.txt '>' \
      $graph_dir/HCLG.fst
fi

###############################################################################
## Do Viterbi decoding to create per-frame alignments.
###############################################################################

transform_opt=
if ! [ -z "$output_scale" ]; then
  # Transformation matrix for output scaling computed from provided
  # `output_scale` values
  echo $output_scale | python -c "import sys
sys.path.insert(0, 'steps')
import libs.common as common_lib

line = sys.stdin.read()
sil_prior, single_prior, ovl_prior = line.strip().split()
transform_mat = [[float(sil_prior),0,0], [0,float(single_prior),0], [0,0,float(ovl_prior)]]
common_lib.write_matrix_ascii(sys.stdout, transform_mat)" > $overlap_dir/transform_probs.mat
  transform_opt="--transform $overlap_dir/transform_probs.mat"
fi

if [ $stage -le 3 ]; then
  echo "$0: Decoding output"
  steps/segmentation/decode_sad.sh --acwt $acwt --cmd "$cmd" --nj $nj \
    $transform_opt $graph_dir $out_dir $overlap_dir
fi

###############################################################################
## Post-process output to create RTTM file containing overlaps.
###############################################################################

if [ $stage -le 4 ]; then
  steps/overlap/post_process_output.sh \
    --segment-padding $segment_padding --min-segment-dur $min_segment_dur \
    --merge-consecutive-max-dur $merge_consecutive_max_dur \
    --cmd "$cmd" --frame-shift $(perl -e "print $frame_subsampling_factor * $frame_shift") \
    --region-type $region_type \
    ${test_data_dir} ${overlap_dir} ${out_dir}
fi

echo "$0: Created output overlap RTTM at ${out_dir}/rttm_${region_type}"
exit 0
