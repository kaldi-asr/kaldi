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

# Feature options (Must match training -- usually passed to do_corruption_data_dir_snr.sh script)
mfcc_config=conf/mfcc_hires_bp.conf
feat_affix=bp   # Affix for the type of feature used

convert_data_dir_to_whole=true    # If true, the input data directory is 
                                  # first converted to whole data directory (i.e. whole recordings)
                                  # and segmentation is done on that.
                                  # If false, then the original segments are 
                                  # retained and they are split into sub-segments.

# Set to true if the test data needs to be downsampled. 
# The appropriate sample-frequency is read from the mfcc_config.
# TODO: Use resample_data_dir instead.
do_downsampling=false

output_name=output-speech   # The output node in the network
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

frame_subsampling_factor=3  # Subsampling at the output

# Decoding options
transition_scale=3.0
loopscale=0.1

# Segmentation config for post-processing
segmentation_config=conf/segmentation_speech.conf

echo $* 

. utils/parse_options.sh

if [ $# -ne 6 ]; then
  echo "This script does nnet3-based speech activity detection given an input kaldi
  echo "directory and outputs an output kaldi directory."
  echo "See script for details of the options to be supplied."
  echo "Usage: $0 <src-data-dir> <sad-nnet-dir> <lang> <mfcc-dir> <dir> <out-data-dir>"
  echo " e.g.: $0 ~/workspace/egs/ami/s5b/data/sdm1/dev exp/nnet3_sad_snr/nnet_tdnn_j_n4 \\"
  echo "    data/lang_sad mfcc_hires_bp exp/segmentation_sad_snr/nnet_tdnn_j_n4 data/ami_sdm1_dev"
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
  echo "  --downsampling <true|false>     # Set to true if test data needs to be downsampled."
  echo "  --output-name <name>    # The output node in the network"
  echo "  --extra-left-context  <context|0>   # Set to some large value, typically 40 for LSTM (must match training)"
  echo "  --extra-right-context  <context|0>   # For BLSTM or statistics pooling"
  echo "  --frame-subsampling-factor  <int|3>   # Subsampling at the output. Must match the lang directory."
  echo "  --transition-scale <float|3.0>    # LMWT for decoding"
  echo "  --loopscale <float|0.1>   # Scale on self-loop log-probabilities"
  echo "  --segmentation-config  <config|conf/segmentation_speech.conf>  # Config for post-processing"
  exit 1
fi

src_data_dir=$1   # The input data directory that needs to be segmented.
                  # If convert_data_dir_to_whole is true, any segments in that will be ignored.
sad_nnet_dir=$2   # The SAD neural network
lang=$3           # Directory with information about the classes, using which
                  # a simple HMM topology is created.
mfcc_dir=$4       # The directory to store the features
dir=$5            # Work directory
data_dir=$5       # The output data directory will be ${data_dir}_seg

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
    
    if $do_downsampling; then
      freq=`cat $mfcc_config | perl -pe 's/\s*#.*//g' | grep "sample-frequency=" | awk -F'=' '{if (NF == 0) print 16000; else print $2}'`
      utils/data/downsample_data_dir.sh $freq $whole_data_dir
    fi

    rm -r ${test_data_dir} || true
    utils/copy_data_dir.sh ${whole_data_dir} $test_data_dir
  fi
else
  if [ $stage -le 0 ]; then
    rm -r ${test_data_dir} || true
    utils/copy_data_dir.sh $src_data_dir $test_data_dir

    if $do_downsampling; then
      freq=`cat $mfcc_config | perl -pe 's/\s*#.*//g' | grep "sample-frequency=" | awk -F'=' '{if (NF == 0) print 16000; else print $2}'`
      utils/data/downsample_data_dir.sh $freq $test_data_dir
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
## Prepare simple HMM to model transitions
###############################################################################

if [ ! -f $lang/classes_info.txt ]; then
  echo "$0: Could not find $lang/topo or $lang/classes_info.txt"
  exit 1
fi

if [ $stage -le 2 ]; then
  steps/segmentation/internal/prepare_simple_hmm_lang.py \
    $lang/classes_info.txt $lang
fi


###############################################################################
## Initialize acoustic model for decoding using the output $output_name
###############################################################################

post_vec=$sad_nnet_dir/post_${output_name}.vec
if [ ! -f $sad_nnet_dir/post_${output_name}.vec ]; then
  echo "$0: Could not find $sad_nnet_dir/post_${output_name}.vec. "
  echo "Re-run the corresponding stage in the training script."
  echo "See local/segmentation/train_lstm_sad_music.sh for example."
  exit 1
fi

if [ $stage -le 3 ]; then
  $cmd $dir/log/prepare_simple_hmm.log \
    simple-hmm-init $lang/topo $lang/init.mdl

  $cmd $dir/log/get_final_${output_name}_model.log \
    nnet3-am-init $lang/init.mdl \
    "nnet3-copy --edits='rename-node old-name=$output_name new-name=output' $sad_nnet_dir/$iter.raw - |" - \| \
    nnet3-am-adjust-priors - $sad_nnet_dir/post_${output_name}.vec \
    $dir/${iter}_${output_name}.mdl
fi
iter=${iter}_${output_name}

###############################################################################
## Forward pass through the network network and dump the log-likelihoods.
###############################################################################

if [ $stage -le 4 ]; then
  steps/nnet3/compute_output.sh --nj $nj --cmd "$cmd" \
    --iter $iter --use-raw-nnet false \
    --extra-left-context $extra_left_context \
    --extra-right-context $extra_right_context \
    --frames-per-chunk 150 \
    --frame-subsampling-factor $frame_subsampling_factor \
    ${test_data_dir} $sad_nnet_dir $sad_dir
fi

###############################################################################
## Prepare HCLG graph for decoding
###############################################################################

graph_dir=${dir}/graph_${output_name}
if [ $stage -le 5 ]; then
  cp -r $lang $graph_dir

  if [ ! -f $lang/final.mdl ]; then
    echo "$0: Could not find $lang/final.mdl!"
    echo "$0: Using $lang/init.mdl instead"
    cp $lang/init.mdl $graph_dir/final.mdl
  else
    cp $lang/final.mdl $graph_dir
  fi

  $cmd $lang/log/make_graph.log \
    make-simple-hmm-graph --transition-scale=$transition_scale \
    --self-loop-scale=$loopscale \
    $graph_dir/final.mdl \| \
    fstdeterminizestar --use-log=true \| \
    fstrmepslocal \| \
    fstminimizeencoded '>' $graph_dir/HCLG.fst
fi

###############################################################################
## Do Viterbi decoding to create per-frame alignments.
###############################################################################

if [ $stage -le 6 ]; then
  steps/segmentation/decode_sad.sh --acwt 1.0 --cmd "$decode_cmd" \
    --iter ${iter} \
    --get-pdfs true $graph_dir $sad_dir $seg_dir
fi

###############################################################################
## Post-process segmentation to create kaldi data directory.
###############################################################################

if [ $stage -le 7 ]; then
  steps/segmentation/post_process_sad_to_subsegments.sh \
    --cmd "$train_cmd" --segmentation-config $segmentation_config \
    --frame-subsampling-factor $frame_subsampling_factor \
    ${test_data_dir} $lang/phone2sad_map ${seg_dir} \
    ${seg_dir} ${data_dir}_seg

  cp $src_data_dir/wav.scp ${data_dir}_seg
fi
