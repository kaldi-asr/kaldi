#!/bin/bash

# Copyright 2019 Srikanth Madikeri (Idiap Research Institute)
# 
# This script is a modification of local/chain/run_tdnn.sh adapted to the chain2 recipes.

# Set -e here so that we catch if any executable fails immediately
set -euo pipefail

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=0
decode_nj=10
train_set=train_clean_5
test_sets=dev_clean_2
gmm=tri3b
srand=0
nnet3_affix=

# The rest are configs specific to this script.  Most of the parameters
# are just hardcoded at this level, in the commands below.
affix=2c   # affix for the TDNN directory name
tree_affix=
train_stage=-10
get_egs_stage=-10


# training chunk-options
chunk_width=140
dropout_schedule='0,0@0.20,0.3@0.50,0'
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


# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

# if ! cuda-compiled; then
#   cat <<EOF && exit 1
# This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
# If you want to use GPUs (and have them), go to src/, and configure and make on a machine
# where "nvcc" is installed.
# EOF
# fi

if [ $stage -le 9 ]; then
    local/chain2/data_prep_common.sh  \
             --train-set $train_set \
             --gmm $gmm  || exit 1;
fi

# Problem: We have removed the "train_" prefix of our training set in
# the alignment directory names! Bad!
gmm_dir=exp/$gmm
ali_dir=exp/${gmm}_ali_${train_set}_sp
tree_dir=exp/chaina/tree_sp${tree_affix:+_$tree_affix}
lang=data/lang_chain
lat_dir=exp/chaina/${gmm}_${train_set}_sp_lats
dir=exp/chaina/tdnn${affix}_sp
train_data_dir=data/${train_set}_sp_hires
lores_train_data_dir=data/${train_set}_sp

for f in $gmm_dir/final.mdl $train_data_dir/feats.scp \
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
   if [ -f $tree_dir/final.mdl ]; then
     echo "$0: $tree_dir/final.mdl already exists, refusing to overwrite it."
     exit 1;
  fi
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
  input name=input dim=40
  # the first splicing is moved before the lda layer, so no splicing here
  fixed-affine-layer name=lda input=Append(-2,-1,0,1,2) affine-transform-file=$dir/configs/lda.mat
  relu-renorm-layer name=tdnn1 dim=512 input=Append(-2,-1,0,1,2)
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
  if [ $dir/init/default_trans.mdl ]; then # checking this because it may have been copied in a previous run of the same script
      copy-transition-model $tree_dir/final.mdl $dir/init/default_trans.mdl  || exit 1 &
  else
      echo "Keeping the old $dir/init/default_trans.mdl as it already exists."
  fi
fi
wait;

init_info=$dir/init/info.txt
if [ $stage -le 15 ]; then

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
    --left-context $egs_left_context \
    --right-context $egs_right_context \
    --frame-subsampling-factor $frame_subsampling_factor \
    --alignment-subsampling-factor $frame_subsampling_factor \
    --frames-per-chunk 140,100,160 \
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
    --stage $train_stage --cmd "$cuda_cmd" \
    --xent-regularize $xent_regularize --leaky-hmm-coefficient 0.1 \
    --max-param-change 2.0 \
    --num-jobs-initial 2 --num-jobs-final 5 \
     $dir/egs $dir
fi

if [ $stage -le 23 ]; then
  # Note: it's not important to give mkgraph.sh the lang directory with the
  # matched topology (since it gets the topology file from the model).
  utils/mkgraph.sh \
    --self-loop-scale 1.0 data/lang_test_tgsmall \
    $tree_dir $tree_dir/graph_tgsmall || exit 1;
fi

if [ $stage -le 24 ]; then
  # Do the speaker-dependent decoding pass
  test_sets=dev_clean_2
  for data in $test_sets; do
      nspk=$(wc -l <data/${data}_hires/spk2utt)
      steps/nnet3/decode.sh \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --extra-left-context $egs_left_context \
          --extra-right-context $egs_right_context \
          --frames-per-chunk 150 \
          --extra-left-context-initial 0 \
          --extra-right-context-final 0 \
          --nj $nspk --cmd "$decode_cmd"   \
          $tree_dir/graph_tgsmall data/${data}_hires ${dir}/decode_tgsmall_${data} || exit 1
      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
        data/lang_test_{tgsmall,tglarge} \
       data/${data}_hires ${dir}/decode_{tgsmall,tglarge}_${data} || exit 1
  done
fi

exit 0;
