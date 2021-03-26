#!/bin/bash

# Copyright 2019 Idiap Research Institute (Srikanth Madikeri)
# Apache 2.0.
# run_tdnn_7k.sh in local/chain but uses new kaldi recipe.

set -e

# configs for 'chain'
affix=chain2
stage=0
train_stage=-10
get_egs_stage=-10
speed_perturb=true
dir=exp/chain2/tdnn_8k  # Note: _sp will get added to this if $speed_perturb == true.
decode_iter=
decode_nj=96
decode_suff=

# The amount of extra left/right context we put in the egs.  Note: this could
# easily be zero, since we're not using a recurrent topology, but we put in a
# little extra context so that we have more room to play with the configuration
# without re-dumping egs.
egs_extra_left_context=5
egs_extra_right_context=5

# training options
frame_subsampling_factor=3
num_epochs=4
initial_effective_lrate=0.001
final_effective_lrate=0.0001
max_param_change=2.0
final_layer_normalize_target=0.5
num_jobs_initial=3
num_jobs_final=16
minibatch_size=128
frames_per_eg=150
remove_egs=true
common_egs_dir=exp/chain/tdnn_8k_chaina_v2_sp/egs
xent_regularize=0.1
srand=0
graph_dir=

test_online_decoding=false  # if true, it will run the last decoding stage.

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
# nnet3 setup, and you can skip them by setting "--stage 8" if you have already
# run those things.

suffix=
if [ "$speed_perturb" == "true" ]; then
  suffix=_sp
fi

dir=${dir}${affix:+_$affix}$suffix
train_set=train_nodup$suffix
gmm_dir=exp/tri4
ali_dir=exp/tri4_ali_nodup$suffix
treedir=exp/chain/tri5_7d_tree$suffix
lang=data/lang_chain_2y


# if we are using the speed-perturbed data we need to generate
# alignments for it.
local/nnet3/run_ivector_common.sh --stage $stage \
  --speed-perturb $speed_perturb \
  --generate-alignments $speed_perturb || exit 1;


# skipping this step
if [ $stage -le 9 ]; then
  # Get the alignments as lattices (gives the LF-MMI training more freedom).
  # use the same num-jobs as the alignments
  nj=$(cat exp/tri4_ali_nodup$suffix/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" data/$train_set \
    data/lang exp/tri4 exp/tri4_lats_nodup$suffix
  rm exp/tri4_lats_nodup$suffix/fsts.*.gz # save space
fi
lat_dir=exp/tri4_lats_nodup$suffix
train_data_dir=data/${train_set}_hires


if [ $stage -le 10 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  rm -rf $lang
  cp -r data/lang $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi

if [ $stage -le 11 ]; then
  # Build a tree using our new topology. This is the critically different
  # step compared with other recipes.
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor $frame_subsampling_factor \
      --context-opts "--context-width=2 --central-position=1" \
      --cmd "$train_cmd" 7000 data/$train_set $lang $ali_dir $treedir
fi

if [ $stage -le 12 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 dim=625
  relu-batchnorm-layer name=tdnn2 input=Append(-1,0,1) dim=625
  relu-batchnorm-layer name=tdnn3 input=Append(-1,0,1) dim=625
  relu-batchnorm-layer name=tdnn4 input=Append(-3,0,3) dim=625
  relu-batchnorm-layer name=tdnn5 input=Append(-3,0,3) dim=625
  relu-batchnorm-layer name=tdnn6 input=Append(-3,0,3) dim=625
  relu-batchnorm-layer name=tdnn7 input=Append(-3,0,3) dim=625

  ## adding the layers for chain branch
  relu-batchnorm-layer name=prefinal-chain input=tdnn7 dim=625 target-rms=0.5
  output-layer name=output include-log-softmax=false dim=$num_targets max-change=1.5
  output-layer name=output-default input=prefinal-chain include-log-softmax=false dim=$num_targets max-change=1.5

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  relu-batchnorm-layer name=prefinal-xent input=tdnn7 dim=625 target-rms=0.5
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5
  output-layer name=output-default-xent input=prefinal-xent dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5

EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
  if [ ! -f $dir/init/default_trans.mdl ]; then # checking this because it may have been copied in a previous run of the same script
      copy-transition-model $treedir/final.mdl $dir/init/default_trans.mdl  || exit 1 &
  else
      echo "Keeping the old $dir/init/default_trans.mdl as it already exists."
  fi
fi

init_info=$dir/init/info.txt
if [ $stage -le 13 ]; then

  if [ ! -f $dir/configs/ref.raw ]; then
      echo "Expected $dir/configs/ref.raw to exist"
      exit
  fi

  mkdir  -p $dir/init
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
if [ $stage -le 13 ]; then
  echo "$0: Making Phone LM and denominator and normalization FST"
  mkdir -p $dir/den_fsts/log

  # We may later reorganize this.
  cp $treedir/tree $dir/default.tree

  echo "$0: creating phone language-model"
  #TODO: check if we should use ali_dir or gmm_dir!
  $train_cmd $dir/den_fsts/log/make_phone_lm_default.log \
    chain-est-phone-lm --num-extra-lm-states=2000 \
       "ark:gunzip -c $ali_dir/ali.*.gz | ali-to-phones $gmm_dir/final.mdl ark:- ark:- |" \
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


if [ -z $common_egs_dir ]; then
    if [ $stage -le 14 ]; then
      echo "$0: about to dump raw egs."
      # Dump raw egs.
      steps/chain/get_raw_egs.sh --cmd "$train_cmd" \
        --lang "default" \
        --online-ivector-dir exp/nnet3/ivectors_${train_set} \
        --left-context $egs_left_context \
        --right-context $egs_right_context \
        --frame-subsampling-factor $frame_subsampling_factor \
        --alignment-subsampling-factor $frame_subsampling_factor \
        --frames-per-chunk $frames_per_eg \
        ${train_data_dir} ${dir} ${lat_dir} ${dir}/raw_egs
    fi

    if [ $stage -le 15 ]; then
      echo "$0: about to process egs"
      steps/chain/process_egs.sh  --cmd "$train_cmd" \
          --num-repeats 1 \
        ${dir}/raw_egs ${dir}/processed_egs
    fi

    if [ $stage -le 16 ]; then
      echo "$0: about to randomize egs"
      steps/chain/randomize_egs.sh --frames-per-job 1500000 \
        ${dir}/processed_egs ${dir}/egs
    fi
    common_egs_dir=$dir/egs
fi

if [ $stage -le 17 ]; then
    echo "$0: Training pre-conditioning matrix"
    num_lda_jobs=`find $common_egs_dir/ -iname 'train.*.scp' | wc -l | cut -d ' ' -f2`
    steps/chain/compute_preconditioning_matrix.sh --cmd "$train_cmd" \
        --nj $num_lda_jobs \
        $dir/configs/init.raw \
        $common_egs_dir \
        $dir || exit 1
fi

if [ $stage -le 18 ]; then
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

if [ $stage -le 19 ]; then
  echo "$0: about to train model"
  steps/chain/train2.sh \
    --stage $train_stage --cmd "$cuda_cmd" \
    --xent-regularize $xent_regularize --leaky-hmm-coefficient 0.1 \
    --initial-effective-lrate $initial_effective_lrate \
    --final-effective-lrate $final_effective_lrate \
    --max-param-change $max_param_change \
    --minibatch-size 128 \
    --l2-regularize 0.00005 \
    --num-jobs-initial $num_jobs_initial --num-jobs-final $num_jobs_final \
     $common_egs_dir $dir
fi


if [ $stage -le 20 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_sw1_tg $dir $dir/graph_sw1_tg
fi


graph_dir=$dir/graph_sw1_tg
iter_opts=
if [ ! -z $decode_iter ]; then
  iter_opts=" --iter $decode_iter "
fi
if [ $stage -le 21 ]; then
  rm $dir/.error 2>/dev/null || true
  for decode_set in eval2000; do
      (
      decode_nj=`wc -l data/${decode_set}_hires/spk2utt | cut -d ' ' -f1`
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $decode_nj --cmd "$decode_cmd" $iter_opts \
          --online-ivector-dir exp/nnet3/ivectors_${decode_set} \
          $graph_dir data/${decode_set}_hires \
          $dir/decode_${decode_set}${decode_iter:+_$decode_iter}${decode_suff}_sw1_tg || exit 1;
      if $has_fisher; then
          steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
            data/lang_sw1_{tg,fsh_fg} data/${decode_set}_hires \
            $dir/decode_${decode_set}${decode_iter:+_$decode_iter}${decode_suff}_sw1_{tg,fsh_fg} || exit 1;
      fi
      ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi

if $test_online_decoding && [ $stage -le 16 ]; then
  # note: if the features change (e.g. you add pitch features), you will have to
  # change the options of the following command line.
  steps/online/nnet3/prepare_online_decoding.sh \
       --mfcc-config conf/mfcc_hires.conf \
       $lang exp/nnet3/extractor $dir ${dir}_online

  rm $dir/.error 2>/dev/null || true
  for decode_set in train_dev eval2000; do
    (
      # note: we just give it "$decode_set" as it only uses the wav.scp, the
      # feature type does not matter.

      steps/online/nnet3/decode.sh --nj $decode_nj --cmd "$decode_cmd" \
          --acwt 1.0 --post-decode-acwt 10.0 \
         $graph_dir data/${decode_set}_hires \
         ${dir}_online/decode_${decode_set}${decode_iter:+_$decode_iter}_sw1_tg || exit 1;
      if $has_fisher; then
          steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
            data/lang_sw1_{tg,fsh_fg} data/${decode_set}_hires \
            ${dir}_online/decode_${decode_set}${decode_iter:+_$decode_iter}_sw1_{tg,fsh_fg} || exit 1;
      fi
    ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi


exit 0;

