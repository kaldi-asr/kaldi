#!/usr/bin/env bash

# Copyright 2019 Srikanth Madikeri (Idiap Research Institute)
#
# This script is a modification of local/chain/run_tdnn.sh adapted to the chain2 recipes.

# This is a basic TDNN experiment.
# run_tdnn_1a.sh in local/chain2 but uses new kaldi recipe.

# steps/info/chain_dir_info.pl exp/chain2/tdnn1a_sp
# exp/chain2/tdnn1a_sp: num-iters=6 nj=2..5 combine=-0.038->-0.033 (over 3)

# local/chain2/compare_wer.sh exp/chain2/tdnn1a_sp
# System                tdnn1a_sp
#WER dev_clean_2 (tgsmall)      17.50
#WER dev_clean_2 (tglarge)      12.67
# Final train prob        -0.0626
# Final valid prob        -0.0539
# Final train prob (xent)   -1.5220
# Final valid prob (xent)   -1.3991
# Num-params                10005600

# Set -e here so that we catch if any executable fails immediately
set -euo pipefail

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=0
decode_nj=10
train_set=train_clean_5
test_sets=dev_clean_2
gmm=tri3b
nnet3_affix=

# The rest are configs specific to this script.  Most of the parameters
# are just hardcoded at this level, in the commands below.
affix=1a   # affix for the TDNN directory name
tree_affix=
train_stage=-10
get_egs_stage=-10
decode_iter=

# training options
# training chunk-options
chunk_width=140,100,160
xent_regularize=0.1
bottom_subsampling_factor=1  # I'll set this to 3 later, 1 is for compatibility with a broken ru.
frame_subsampling_factor=3
langs="default"  # list of language names

# The amount of extra left/right context we put in the egs.  Note: this could
# easily be zero, since we're not using a recurrent topology, but we put in a
# little extra context so that we have more room to play with the configuration
# without re-dumping egs.
egs_extra_left_context=5
egs_extra_right_context=5

# The number of chunks (of length: see $chunk_width above) that we group
# together for each "speaker" (actually: pseudo-speaker, since we may have
# to group multiple speaker together in some cases).
chunks_per_group=4

# training options
srand=0
remove_egs=true
reporting_email=

#decode options
test_online_decoding=true  # if true, it will run the last decoding stage.


# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 11" if you have already
# run those things.
local/nnet3/run_ivector_common.sh --stage $stage \
                                  --train-set $train_set \
                                  --gmm $gmm \
                                  --nnet3-affix "$nnet3_affix" || exit 1;

# Problem: We have removed the "train_" prefix of our training set in
# the alignment directory names! Bad!
gmm_dir=exp/$gmm
ali_dir=exp/${gmm}_ali_${train_set}_sp
tree_dir=exp/chain2${nnet3_affix}/tree_sp${tree_affix:+_$tree_affix}
lang=data/lang_chain
lat_dir=exp/chain2${nnet3_affix}/${gmm}_${train_set}_sp_lats
dir=exp/chain2${nnet3_affix}/tdnn${affix}_sp
train_data_dir=data/${train_set}_sp_hires
lores_train_data_dir=data/${train_set}_sp
train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires

for f in $gmm_dir/final.mdl $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp \
    $lores_train_data_dir/feats.scp $ali_dir/ali.1.gz; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done

if [ $stage -le 10 ]; then
  echo "$0: creating lang directory $lang with chain-type topology"
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  if [ -d $lang ]; then
    if [ $lang/L.fst -nt data/lang/L.fst ]; then
      echo "$0: $lang already exists, not overwriting it; continuing"
    else
      echo "$0: $lang already exists and seems to be older than data/lang..."
      echo " ... not sure what to do.  Exiting."
      exit 1;
    fi
  else
    cp -r data/lang $lang
    silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
    nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
    # Use our special topology... note that later on may have to tune this
    # topology.
    steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
  fi
fi

if [ $stage -le 11 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --nj 75 --cmd "$train_cmd" ${lores_train_data_dir} \
    data/lang $gmm_dir $lat_dir
  rm $lat_dir/fsts.*.gz # save space
fi

if [ $stage -le 12 ]; then
  # Build a tree using our new topology.  We know we have alignments for the
  # speed-perturbed data (local/nnet3/run_ivector_common.sh made them), so use
  # those.  The num-leaves is always somewhat less than the num-leaves from
  # the GMM baseline.
  # This will be a two-level tree (with the smaller number of leaves specified
  # by the '--num-clusters' option); this is needed by the adaptation framework
  # search below for 'tree.map'
  steps/nnet3/chain/build_tree.sh \
     --frame-subsampling-factor ${frame_subsampling_factor} \
     --context-opts "--context-width=2 --central-position=1" \
     --cmd "$train_cmd" 3500 ${lores_train_data_dir} \
     $lang $ali_dir $tree_dir
fi


# $dir/configs will contain xconfig and config files for the initial
# models.  It's a scratch space used by this script but not by
# scripts called from here.
mkdir -p $dir/configs/
# $dir/init will contain the initial models
mkdir -p $dir/init/

learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)

if [ $stage -le 14 ]; then

  # Note: we'll use --bottom-subsampling-factor=3, so all time-strides for the
  # top network should be interpreted at the 30ms frame subsampling rate.
  num_leaves=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')

  echo "$0: creating top model"
  cat <<EOF > $dir/configs/default.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-2,-1,0,1,2,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-renorm-layer name=tdnn1 dim=512
  relu-renorm-layer name=tdnn2 dim=512 input=Append(-1,0,1)
  relu-renorm-layer name=tdnn3 dim=512 input=Append(-1,0,1)
  relu-renorm-layer name=tdnn4 dim=512 input=Append(-3,0,3)
  relu-renorm-layer name=tdnn5 dim=512 input=Append(-3,0,3)
  relu-renorm-layer name=tdnn6 dim=512 input=Append(-6,-3,0)
  relu-renorm-layer name=prefinal-chain dim=512 target-rms=0.5
  output-layer name=output include-log-softmax=false dim=$num_leaves max-change=1.5
  output-layer name=output-default input=prefinal-chain include-log-softmax=false dim=$num_leaves max-change=1.5
  relu-renorm-layer name=prefinal-xent input=tdnn6 dim=512 target-rms=0.5
  output-layer name=output-xent dim=$num_leaves learning-rate-factor=$learning_rate_factor max-change=1.5
  output-layer name=output-default-xent input=prefinal-xent dim=$num_leaves learning-rate-factor=$learning_rate_factor max-change=1.5
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/default.xconfig --config-dir $dir/configs/
fi
wait;

init_info=$dir/init/info.txt
if [ $stage -le 15 ]; then

  if [ ! -f $dir/init/default_trans.mdl ]; then # checking this because it may have been copied in a previous run of the same script
      copy-transition-model $tree_dir/final.mdl $dir/init/default_trans.mdl  || exit 1 &
  else
      echo "Keeping the old $dir/init/default_trans.mdl as it already exists."
  fi

  if [ ! -f $dir/configs/ref.raw ]; then
      echo "Expected $dir/configs/ref.raw to exist"
      exit
  fi

  nnet3-info $dir/configs/ref.raw  > $dir/configs/temp.info 
  model_left_context=`fgrep 'left-context' $dir/configs/temp.info | awk '{print $2}'`
  model_right_context=`fgrep 'right-context' $dir/configs/temp.info | awk '{print $2}'`
  cat >$init_info <<EOF
frame_subsampling_factor $frame_subsampling_factor
langs $langs
model_left_context $model_left_context
model_right_context $model_right_context
EOF
  rm $dir/configs/temp.info
fi

# Make phone LM and denominator and normalization FST
if [ $stage -le 16 ]; then
  echo "$0: Making Phone LM and denominator and normalization FST"
  mkdir -p $dir/den_fsts/log

  # We may later reorganize this.
  cp $tree_dir/tree $dir/default.tree

  echo "$0: creating phone language-model"
  $train_cmd $dir/den_fsts/log/make_phone_lm_default.log \
    chain-est-phone-lm --num-extra-lm-states=2000 \
       "ark:gunzip -c $gmm_dir/ali.*.gz | ali-to-phones $gmm_dir/final.mdl ark:- ark:- |" \
       $dir/den_fsts/default.phone_lm.fst

  echo "$0: creating denominator FST"
  $train_cmd $dir/den_fsts/log/make_den_fst.log \
     chain-make-den-fst $dir/default.tree $dir/init/default_trans.mdl $dir/den_fsts/default.phone_lm.fst \
     $dir/den_fsts/default.den.fst $dir/den_fsts/default.normalization.fst || exit 1;
fi

model_left_context=$(awk '/^model_left_context/ {print $2;}' $dir/init/info.txt)
model_right_context=$(awk '/^model_right_context/ {print $2;}' $dir/init/info.txt)
if [ -z $model_left_context ]; then
    echo "ERROR: Cannot find entry for model_left_context in $dir/init/info.txt"
fi
if [ -z $model_right_context ]; then
    echo "ERROR: Cannot find entry for model_right_context in $dir/init/info.txt"
fi
# Note: we add frame_subsampling_factor/2 so that we can support the frame
# shifting that's done during training, so if frame-subsampling-factor=3, we
# train on the same egs with the input shifted by -1,0,1 frames.  This is done
# via the --frame-shift option to nnet3-chain-copy-egs in the script.
egs_left_context=$[model_left_context+(frame_subsampling_factor/2)+egs_extra_left_context]
egs_right_context=$[model_right_context+(frame_subsampling_factor/2)+egs_extra_right_context]

for d in $dir/raw_egs $dir/processed_egs; do
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $d/storage ] ; then
    mkdir -p $d
    utils/create_split_dir.pl \
      /export/b0{3,4,5,6}/$USER/kaldi-data/egs/mini_librispeech-$(date +'%m_%d_%H_%M')/s5/$d/storage $d/storage
  fi
done


if [ $stage -le 17 ]; then
  echo "$0: about to dump raw egs."
  # Dump raw egs.
  steps/chain2/get_raw_egs.sh --cmd "$train_cmd" \
    --lang "default" \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --left-context $egs_left_context \
    --right-context $egs_right_context \
    --frame-subsampling-factor $frame_subsampling_factor \
    --alignment-subsampling-factor $frame_subsampling_factor \
    --frames-per-chunk ${chunk_width} \
    --online-ivector-dir ${train_ivector_dir} \
    ${train_data_dir} ${dir} ${lat_dir} ${dir}/raw_egs
fi

if [ $stage -le 18 ]; then
  echo "$0: about to process egs"
  steps/chain2/process_egs.sh  --cmd "$train_cmd" \
      --num-repeats 1 \
    ${dir}/raw_egs ${dir}/processed_egs
fi

if [ $stage -le 19 ]; then
  echo "$0: about to randomize egs"
  steps/chain2/randomize_egs.sh --frames-per-job 3000000 \
    ${dir}/processed_egs ${dir}/egs
fi

if [ $stage -le 20 ]; then
    echo "$0: Training pre-conditioning matrix"
    num_lda_jobs=`find ${dir}/egs/ -iname 'train.*.scp' | wc -l | cut -d ' ' -f2`
    steps/chain2/compute_preconditioning_matrix.sh --cmd "$train_cmd" \
        --nj $num_lda_jobs \
        $dir/configs/init.raw \
        $dir/egs \
        $dir || exit 1
fi

if [ $stage -le 21 ]; then
    echo "$0: Preparing initial acoustic model"
    if [ -f $dir/configs/init.config ]; then
            $train_cmd ${dir}/log/add_first_layer.log \
                    nnet3-init --srand=${srand} ${dir}/configs/init.raw \
                    ${dir}/configs/final.config ${dir}/init/default.raw || exit 1
    else
            $train_cmd ${dir}/log/init_model.log \
               nnet3-init --srand=${srand} ${dir}/configs/final.config ${dir}/init/default.raw || exit 1
    fi

    $train_cmd $dir/log/init_mdl.log \
        nnet3-am-init ${dir}/init/default_trans.mdl $dir/init/default.raw $dir/init/default.mdl || exit 1
fi

if [ $stage -le 22 ]; then
  echo "$0: about to train model"
  steps/chain2/train.sh \
    --stage $train_stage --cmd "$train_cmd" \
    --xent-regularize $xent_regularize --leaky-hmm-coefficient 0.1 \
    --max-param-change 2.0 \
    --num-jobs-initial 2 --num-jobs-final 5 \
    --minibatch-size 256,128,64 \
     $dir/egs $dir || exit 1;
fi

if [ $stage -le 23 ]; then
  # Note: it's not important to give mkgraph.sh the lang directory with the
  # matched topology (since it gets the topology file from the model).
  utils/mkgraph.sh \
    --self-loop-scale 1.0 data/lang_test_tgsmall \
    $tree_dir $tree_dir/graph_tgsmall || exit 1;
fi

if [ $stage -le 24 ]; then
  frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
  # Do the speaker-dependent decoding pass
  test_sets=dev_clean_2
  for data in $test_sets; do
    (
      nspk=$(wc -l <data/${data}_hires/spk2utt)
      steps/nnet3/decode.sh \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --extra-left-context $egs_left_context \
          --extra-right-context $egs_right_context \
          --extra-left-context-initial 0 \
          --extra-right-context-final 0 \
          --frames-per-chunk $frames_per_chunk \
          --nj $nspk --cmd "$decode_cmd"  --num-threads 4 \
          --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${data}_hires \
          $tree_dir/graph_tgsmall data/${data}_hires ${dir}/decode_tgsmall_${data} || exit 1
      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
        data/lang_test_{tgsmall,tglarge} \
       data/${data}_hires ${dir}/decode_{tgsmall,tglarge}_${data} || exit 1
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

exit 0;
