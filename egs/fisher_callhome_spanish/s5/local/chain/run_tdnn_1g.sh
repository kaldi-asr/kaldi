#!/usr/bin/env bash

# 1g is like 1f but upgrading to a "resnet-style TDNN-F model", i.e.
#   with bypass resnet connections, and re-tuned.
# compute-wer --text --mode=present ark:exp/chain/multipsplice_tdnn/decode_fsp_train_test/scoring_kaldi/test_filt.txt ark,p:- 
# %WER 22.21 [ 8847 / 39831, 1965 ins, 2127 del, 4755 sub ]
# %SER 56.98 [ 3577 / 6278 ]
# Scored 6278 sentences, 0 not present in hyp.

# steps/info/chain_dir_info.pl  exp/chain/multipsplice_tdnn
# exp/chain/multipsplice_tdnn: num-iters=296 nj=1..2 num-params=8.2M dim=40+100->2489 combine=-0.170->-0.165 (over 8) xent:train/valid[196,295,final]=(-2.30,-1.93,-1.83/-2.24,-1.96,-1.86) logprob:train/valid[196,295,final]=(-0.208,-0.169,-0.164/-0.189,-0.161,-0.158)

set -e -o pipefail

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=0
nj=30
train_set=train
test_sets="test dev"
gmm=tri5a        # this is the source gmm-dir that we'll use for alignments; it
                 # should have alignments for the specified training data.
num_threads_ubm=32
nnet3_affix=       # affix for exp dirs, e.g. it was _cleaned in tedlium.

# Options which are not passed through to run_ivector_common.sh
affix=1g   #affix for TDNN+LSTM directory e.g. "1a" or "1b", in case we change the configuration.
common_egs_dir=
reporting_email=

# LSTM/chain options
train_stage=-10
xent_regularize=0.1
dropout_schedule='0,0@0.20,0.3@0.50,0'

# training chunk-options
chunk_width=140,100,160
# we don't need extra left/right context for TDNN systems.
chunk_left_context=0
chunk_right_context=0

# training options
srand=0
remove_egs=true

#decode options
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

if [ $stage -le 15 ]; then
echo "local/nnet3/run_ivector_common.sh \
  --stage $stage --nj $nj \
  --train-set $train_set --gmm $gmm \
  --num-threads-ubm $num_threads_ubm \
  --nnet3-affix "$nnet3_affix""

local/nnet3/run_ivector_common.sh \
  --stage $stage --nj $nj \
  --train-set $train_set --gmm $gmm \
  --num-threads-ubm $num_threads_ubm \
  --nnet3-affix "$nnet3_affix"

fi


gmm_dir=exp/${gmm}
ali_dir=exp/${gmm}_ali_${train_set}_sp
lat_dir=exp/tri5a_lats_nodup_sp
dir=exp/chain/multipsplice_tdnn
train_data_dir=data/${train_set}_sp_hires
train_ivector_dir=exp/nnet3/ivectors_train_sp_hires
lores_train_data_dir=data/${train_set}_sp

# note: you don't necessarily have to change the treedir name
# each time you do a new experiment-- only if you change the
# configuration in a way that affects the tree.
tree_dir=exp/chain/${gmm}_tree
# the 'lang' directory is created by this script.
# If you create such a directory with a non-standard topology
# you should probably name it differently.
lang=data/lang_${gmm}_chain

#for f in $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp \
#    $lores_train_data_dir/feats.scp $gmm_dir/final.mdl \
#    $ali_dir/ali.1.gz $gmm_dir/final.mdl; do
#  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
#done


if [ $stage -le 16 ]; then
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

if [ $stage -le 17 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --nj 100 --cmd "$train_cmd" ${lores_train_data_dir} \
    data/lang $gmm_dir $lat_dir
  rm $lat_dir/fsts.*.gz # save space
fi

if [ $stage -le 18 ]; then
  # Build a tree using our new topology.  We know we have alignments for the
  # speed-perturbed data (local/nnet3/run_ivector_common.sh made them), so use
  # those.  The num-leaves is always somewhat less than the num-leaves from
  # the GMM baseline.
   if [ -f $tree_dir/final.mdl ]; then
     echo "$0: $tree_dir/final.mdl already exists, refusing to overwrite it."
     exit 1;
  fi
  steps/nnet3/chain/build_tree.sh \
    --frame-subsampling-factor 3 \
    --context-opts "--context-width=2 --central-position=1" \
    --cmd "$train_cmd" 3500 ${lores_train_data_dir} \
    $lang $ali_dir $tree_dir
fi


if [ $stage -le 19 ]; then
  mkdir -p $dir
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)
  tdnn_opts="l2-regularize=0.01 dropout-proportion=0.0 dropout-per-dim-continuous=true"
  tdnnf_opts="l2-regularize=0.01 dropout-proportion=0.0 bypass-scale=0.66"
  linear_opts="l2-regularize=0.01 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.01"
  output_opts="l2-regularize=0.005"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-dropout-layer name=tdnn1 $tdnn_opts dim=1024
  tdnnf-layer name=tdnnf2 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=1
  tdnnf-layer name=tdnnf3 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=1
  tdnnf-layer name=tdnnf4 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=1
  tdnnf-layer name=tdnnf5 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=0
  tdnnf-layer name=tdnnf6 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf7 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf9 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf10 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf11 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf12 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf13 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  linear-component name=prefinal-l dim=192 $linear_opts


  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts big-dim=1024 small-dim=192
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts big-dim=1024 small-dim=192
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi


if [ $stage -le 20 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/wsj-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/chain/train.py --stage=$train_stage \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir $train_ivector_dir \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.0 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.srand $srand \
    --trainer.max-param-change 2.0 \
    --trainer.num-epochs 4 \
    --trainer.frames-per-iter 5000000 \
    --trainer.optimization.num-jobs-initial 1 \
    --trainer.optimization.num-jobs-final=2 \
    --trainer.optimization.initial-effective-lrate 0.0005 \
    --trainer.optimization.final-effective-lrate 0.00005 \
    --trainer.num-chunk-per-minibatch 128,64 \
    --trainer.optimization.momentum 0.0 \
    --egs.chunk-width $chunk_width \
    --egs.chunk-left-context 0 \
    --egs.chunk-right-context 0 \
    --egs.dir "$common_egs_dir" \
    --egs.opts "--frames-overlap-per-eg 0" \
    --cleanup.remove-egs $remove_egs \
    --use-gpu true \
    --feat-dir $train_data_dir \
    --tree-dir $tree_dir \
    --lat-dir exp/tri5a_lats_nodup_sp \
    --dir $dir  || exit 1;
fi

if [ $stage -le 21 ]; then
  # The reason we are using data/lang_test here, instead of $lang, is just to
  # emphasize that it's not actually important to give mkgraph.sh the
  # lang directory with the matched topology (since it gets the
  # topology file from the model).  So you could give it a different
  # lang directory, one that contained a wordlist and LM of your choice,
  # as long as phones.txt was compatible.
  #LM was trained only on Fisher Spanish train subset.

  utils/mkgraph.sh \
    --self-loop-scale 1.0 data/lang_test \
    $tree_dir $tree_dir/graph_fsp_train || exit 1;

fi

rnnlmdir=exp/rnnlm_lstm_tdnn_1b
if [ $stage -le 22 ]; then
  local/rnnlm/train_rnnlm.sh --dir $rnnlmdir || exit 1;
fi

if [ $stage -le 23 ]; then
  frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    (
      nspk=$(wc -l <data/${data}_hires/spk2utt)
      for lmtype  in fsp_train; do
        steps/nnet3/decode.sh \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --extra-left-context 0 --extra-right-context 0 \
          --extra-left-context-initial 0 \
          --extra-right-context-final 0 \
          --frames-per-chunk $frames_per_chunk \
          --nj $nspk --cmd "$decode_cmd"  --num-threads 4 \
          --online-ivector-dir exp/nnet3/ivectors_${data}_hires \
          $tree_dir/graph_${lmtype} data/${data}_hires ${dir}/decode_${lmtype}_${data} || exit 1;
      done
      bash local/rnnlm/lmrescore_nbest.sh 1.0 data/lang_test $rnnlmdir data/${data}_hires/ \
	      ${dir}/decode_${lmtype}_${data} $dir/decode_rnnLM_${lmtype}_${data} || exit 1;
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

exit 0;
