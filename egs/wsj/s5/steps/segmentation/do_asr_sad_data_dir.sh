#! /bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0.

# This script does nnet3-based speech activity detection given an input kaldi
# directory and outputs an output kaldi directory.
# This script can also do music detection and other similar segmentation
# using appropriate options such as --output-name output-music.

set -e 
set -o pipefail
set -u

. path.sh

affix=  # Affix for the segmentation
nj=32
cmd=queue.pl
stage=-1

# Feature options (Must match training)
mfcc_config=conf/mfcc_hires_bp.conf
feat_affix=bp   # Affix for the type of feature used

convert_data_dir_to_whole=true    # If true, the input data directory is 
                                  # first converted to whole data directory (i.e. whole recordings)
                                  # and segmentation is done on that.
                                  # If false, then the original segments are 
                                  # retained and they are split into sub-segments.

# Set to true if the test data needs to be downsampled. 
# The appropriate sample-frequency is read from the mfcc_config.
do_resampling=false

output_name=output   # The output node in the network
sad_name=sad    # Base name for the directory storing the computed loglikes
                # Can be music for music detection
segmentation_name=segmentation  # Base name for the directory doing segmentation
                                # Can be segmentation_music for music detection

# SAD network config
iter=final  # Model iteration to use

# Contexts must ideally match training for LSTM models, but
# may not necessarily for stats components
extra_left_context=0  # Set to some large value, typically 40 for LSTM (must match training)
extra_right_context=0  
frames_per_chunk=150

subsampling_factor=1  # Subsampling at the output relative to nnet-output

# Decoding options
acwt=0.3
transition_scale=1.0
loopscale=0.3

# These <from>_in_<to>_weight represent the fraction of <from> probability 
# to transfer to <to> class.
speech_in_sil_weight=0.0  
garbage_in_sil_weight=0.0  
sil_in_speech_weight=0.0
garbage_in_speech_weight=0.0

post_processing_opts=   # Options passed to steps/segmentation/post_process_segments_to_datadir.sh

echo $* 

. utils/parse_options.sh

if [ $# -ne 6 ]; then
  echo "This script does nnet3-based speech activity detection given an input kaldi "
  echo "directory and outputs an output kaldi directory."
  echo "See script for details of the options to be supplied."
  echo "Usage: $0 <src-data-dir> <sad-nnet-dir> <classes-info> <mfcc-dir> <dir> <out-data-dir>"
  echo " e.g.: $0 ~/workspace/egs/ami/s5b/data/sdm1/dev exp/nnet3_sad_snr/nnet_tdnn_j_n4 \\"
  echo "    conf/sad.classes_info mfcc_hires_bp exp/segmentation_sad_snr/nnet_tdnn_j_n4 data/ami_sdm1_dev"
  echo ""
  echo "Options: "
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --nj <num-job>                                 # number of parallel jobs to run."
  echo "  --stage <stage>                                # stage to do partial re-run from."
  echo "  --convert-data-dir-to-whole <true|false>    # If true, the input data directory is "
  echo "                                              # first converted to whole data directory (i.e. whole recordings) "
  echo "                                              # and segmentation is done on that."
  echo "                                              # If false, then the original segments are "
  echo "                                              # retained and they are split into sub-segments."
  echo "  --do-resampling <true|false>     # Set to true if test data needs to be re-sampled"
  echo "  --output-name <name>    # The output node in the network"
  echo "  --extra-left-context  <context|0>   # Set to some large value, typically 40 for LSTM (must match training)"
  echo "  --extra-right-context  <context|0>   # For BLSTM or statistics pooling"
  echo "  --subsampling-factor  <int|3>   # Subsampling relative to the output of the nnet. Must match the classes-info."
  echo "  --transition-scale <float|3.0>    # LMWT for decoding"
  echo "  --loopscale <float|0.1>   # Scale on self-loop log-probabilities"
  exit 1
fi

src_data_dir=$1   # The input data directory that needs to be segmented.
                  # If convert_data_dir_to_whole is true, any segments in that will be ignored.
sad_nnet_dir=$2   # The SAD neural network
classes_info=$3   # Information about the classes using which a graph is created

# classes_info must have lines of the format:
# <class-id (1-based)> <initial-probabilitiy> <self-loop-probability> <min-number-of-states> <transition-1> <transition-2> ... <transition-N>
# where <transition-N> is <destination-class>:<transition-probability> 
# and a destination class of -1 is used to represent the final state.
# e.g.:
# 1 0.8 0.99 10 2:0.009 -1:0.001
# 2 0.2 0.99 10 1:0.009 -1:0.001

mfcc_dir=$4       # The directory to store the features
dir=$5            # Work directory
data_dir=$6       # The output data directory will be ${data_dir}_seg

affix=${affix:+_$affix}
feat_affix=${feat_affix:+_$feat_affix}

data_id=`basename $data_dir`
sad_dir=${dir}/${sad_name}${affix}_${data_id}_whole${feat_affix}
seg_dir=${dir}/${segmentation_name}${affix}_${data_id}_whole${feat_affix}

test_data_dir=data/${data_id}${feat_affix}_hires

if $convert_data_dir_to_whole; then
  if [ $stage -le 0 ]; then
    whole_data_dir=${sad_dir}/${data_id}_whole
    utils/data/convert_data_dir_to_whole.sh $src_data_dir ${whole_data_dir}
    
    if $do_resampling; then
      freq=`cat $mfcc_config | perl -pe 's/\s*#.*//g' | grep "sample-frequency=" | awk -F'=' '{if (NF == 0) print 16000; else print $2}'`
      utils/data/resample_data_dir.sh $freq $whole_data_dir
    fi

    rm -r ${test_data_dir} || true
    utils/copy_data_dir.sh ${whole_data_dir} $test_data_dir
  fi
else
  if [ $stage -le 0 ]; then
    rm -r ${test_data_dir} || true
    utils/copy_data_dir.sh $src_data_dir $test_data_dir

    if $do_resampling; then
      freq=`cat $mfcc_config | perl -pe 's/\s*#.*//g' | grep "sample-frequency=" | awk -F'=' '{if (NF == 0) print 16000; else print $2}'`
      utils/data/resample_data_dir.sh $freq $test_data_dir
    fi
  fi
fi

###############################################################################
## Extract input features 
###############################################################################

if [ $stage -le 1 ]; then
  utils/fix_data_dir.sh $test_data_dir
  steps/make_mfcc.sh --mfcc-config $mfcc_config --nj $nj --cmd "$cmd" \
    ${test_data_dir} exp/make_hires/${data_id}${feat_affix} $mfcc_dir
  steps/compute_cmvn_stats.sh ${test_data_dir} exp/make_hires/${data_id}${feat_affix} $mfcc_dir
  utils/fix_data_dir.sh ${test_data_dir}
fi

###############################################################################
## Initialize acoustic model for decoding using the output $output_name
###############################################################################

post_vec=$sad_nnet_dir/post_${output_name}.vec
if [ ! -f $sad_nnet_dir/post_${output_name}.vec ]; then
  if [ ! -f $sad_nnet_dir/post_${output_name}.txt ]; then
    echo "$0: Could not find $sad_nnet_dir/post_${output_name}.vec. "
    echo "Re-run the corresponding stage in the training script possibly "
    echo "with --compute-average-posteriors=true or compute the priors "
    echo "from the training labels"
    exit 1
  else
    post_vec=$sad_nnet_dir/post_${output_name}.txt
  fi
fi

###############################################################################
## Forward pass through the network network and dump the log-likelihoods.
###############################################################################

frame_subsampling_factor=1
if [ -f $sad_nnet_dir/frame_subsampling_factor ]; then
  frame_subsampling_factor=$(cat $sad_nnet_dir/frame_subsampling_factor)
fi

subsampling_factor=$[subsampling_factor * frame_subsampling_factor]

if [ $stage -le 4 ]; then
  if [ "$(readlink -f $sad_nnet_dir)" != "$(readlink -f $dir)" ]; then
    cp $sad_nnet_dir/cmvn_opts $dir
    cp $sad_nnet_dir/{final.mat,splice_opts} $dir || true
  fi

  if [ ! -z "$output_name" ] && [ "$output_name" != output ]; then
    $cmd $dir/log/get_nnet_${output_name}.log \
      nnet3-copy --edits="rename-node old-name=$output_name new-name=output" \
      $sad_nnet_dir/$iter.raw $dir/${iter}_${output_name}.raw || exit 1
    iter=${iter}_${output_name}
  fi

  steps/nnet3/compute_output.sh --nj $nj --cmd "$cmd" \
    --iter ${iter} --use-raw-nnet true \
    --extra-left-context $extra_left_context \
    --extra-right-context $extra_right_context \
    --frames-per-chunk $frames_per_chunk --apply-exp true \
    --frame-subsampling-factor $subsampling_factor \
    ${test_data_dir} $dir $sad_dir || exit 1
fi

###############################################################################
## Prepare HCLG graph for decoding
###############################################################################

if [ ! -f $classes_info ]; then
  echo "$0: Could not find $classes_info"
  exit 1
fi

graph_dir=${dir}/graph_${output_name}
if [ $stage -le 5 ]; then
  steps/segmentation/internal/prepare_sad_lang.py \
    --transition-scale=$transition_scale --loopscale=$loopscale \
    $classes_info $graph_dir || exit 1
  fstcompile --isymbols=$graph_dir/words.txt --osymbols=$graph_dir/words.txt \
    $graph_dir/HCLG.txt > $graph_dir/HCLG.fst
fi

###############################################################################
## Do Viterbi decoding to create per-frame alignments.
###############################################################################

if [ $stage -le 6 ]; then
  mkdir -p $seg_dir
  python -c "
print ('''[ {0} {1} {2}
{3} {4} {5} ]'''.format(
  1.0-$sil_in_speech_weight, $speech_in_sil_weight, $garbage_in_sil_weight,
  $sil_in_speech_weight, 1.0-$speech_in_sil_weight, $garbage_in_speech_weight))""" > $seg_dir/garbage.mat

  # Here --apply-log is true since we read from nnet posteriors 'nnet_output_exp'
  steps/segmentation/decode_sad.sh --acwt $acwt --cmd "$cmd" --apply-log true \
    --transform "$seg_dir/garbage.mat" --likes-prefix nnet_output_exp \
    --priors "$post_vec" \
    $graph_dir $sad_dir $seg_dir
fi

###############################################################################
## Post-process segmentation to create kaldi data directory.
###############################################################################

frame_shift=$(utils/data/get_frame_shift.sh $test_data_dir)
if [ $stage -le 7 ]; then
  steps/segmentation/post_process_sad_to_datadir.sh \
    $post_processing_opts \
    --cmd "$cmd" --frame-shift $(perl -e "print $subsampling_factor * $frame_shift") \
    ${test_data_dir} ${seg_dir} ${seg_dir} ${data_dir}_seg

  cp $src_data_dir/wav.scp ${data_dir}_seg
fi

