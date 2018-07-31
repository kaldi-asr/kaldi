#!/bin/bash


# by default, with cleanup
# please note that the language(s) was not selected for any particular reason (other to represent the various sizes of babel datasets)
# 304-lithuanian   | %WER 40.8 | 20041 61492 | 61.4 27.0 11.6 2.2 40.8 28.8 | -0.172 | exp/chain_cleaned/tdnn_lstm_bab1_sp/decode_dev10h.pem/score_11/dev10h.pem.ctm.sys
#                  num-iters=48 nj=2..12 num-params=10.1M dim=43+100->3273 combine=-0.190->-0.175
#                  xent:train/valid[31,47,final]=(-2.14,-1.88,-1.86/-2.36,-2.23,-2.23)
#                  logprob:train/valid[31,47,final]=(-0.189,-0.157,-0.155/-0.238,-0.233,-0.232)
# 206-zulu         | %WER 52.9 | 22805 52162 | 50.4 38.3 11.2 3.4 52.9 30.9 | -0.516 | exp/chain_cleaned/tdnn_lstm_bab1_sp/decode_dev10h.pem/score_12/dev10h.pem.ctm.sys
#                  num-iters=66 nj=2..12 num-params=10.1M dim=43+100->3274 combine=-0.223->-0.212
#                  xent:train/valid[43,65,final]=(-2.18,-1.95,-1.94/-2.40,-2.32,-2.31)
#                  logprob:train/valid[43,65,final]=(-0.224,-0.191,-0.189/-0.279,-0.281,-0.279)
# 104-pashto       | %WER 41.3 | 21825 101803 | 63.0 27.0 10.0 4.3 41.3 30.1 | -0.441 | exp/chain_cleaned/tdnn_lstm_bab1_sp/decode_dev10h.pem/score_10/dev10h.pem.ctm.sys
#                  num-iters=85 nj=2..12 num-params=10.1M dim=43+100->3328 combine=-0.195->-0.189
#                  xent:train/valid[55,84,final]=(-2.05,-1.84,-1.82/-2.33,-2.23,-2.22)
#                  logprob:train/valid[55,84,final]=(-0.198,-0.170,-0.168/-0.266,-0.260,-0.258)
set -e -o pipefail

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=17
nj=30
train_set=train_cleaned
gmm=tri5_cleaned  # the gmm for the target data
langdir=data/langp/tri5_ali
num_threads_ubm=12
nnet3_affix=_cleaned  # cleanup affix for nnet3 and chain dirs, e.g. _cleaned

# The rest are configs specific to this script.  Most of the parameters
# are just hardcoded at this level, in the commands below.
train_stage=-10
tree_affix=  # affix for tree directory, e.g. "a" or "b", in case we change the configuration.
tdnn_affix=_bab1  #affix for TDNN directory, e.g. "a" or "b", in case we change the configuration.
common_egs_dir=exp/chain_cleaned/tdnn_lstm_sp/egs  # you can set this to use previously dumped egs.
chunk_width=150,120,90,75

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

local/chain/run_ivector_common.sh --stage $stage \
                                  --nj $nj \
                                  --train-set $train_set \
                                  --gmm $gmm \
                                  --num-threads-ubm $num_threads_ubm \
                                  --nnet3-affix "$nnet3_affix"


gmm_dir=exp/$gmm
ali_dir=exp/${gmm}_ali_${train_set}_sp
tree_dir=exp/chain${nnet3_affix}/tree${tree_affix}
lat_dir=exp/chain${nnet3_affix}/${gmm}_${train_set}_sp_lats
dir=exp/chain${nnet3_affix}/tdnn_lstm${tdnn_affix}_sp
train_data_dir=data/${train_set}_sp_hires
lores_train_data_dir=data/${train_set}_sp
train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires


for f in $gmm_dir/final.mdl $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp \
    $lores_train_data_dir/feats.scp $ali_dir/ali.1.gz $gmm_dir/final.mdl; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done

if [ $stage -le 14 ]; then
  echo "$0: creating lang directory with one state per phone."
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  if [ -d data/lang_chain ]; then
    if [ data/lang_chain/L.fst -nt data/lang/L.fst ]; then
      echo "$0: data/lang_chain already exists, not overwriting it; continuing"
    else
      echo "$0: data/lang_chain already exists and seems to be older than data/lang..."
      echo " ... not sure what to do.  Exiting."
      exit 1;
    fi
  else
    cp -r $langdir data/lang_chain
    silphonelist=$(cat data/lang_chain/phones/silence.csl) || exit 1;
    nonsilphonelist=$(cat data/lang_chain/phones/nonsilence.csl) || exit 1;
    # Use our special topology... note that later on may have to tune this
    # topology.
    steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >data/lang_chain/topo
  fi
fi

if [ $stage -le 15 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --nj 100 --cmd "$train_cmd" ${lores_train_data_dir} \
    $langdir $gmm_dir $lat_dir
  rm $lat_dir/fsts.*.gz # save space
fi

if [ $stage -le 16 ]; then
  # Build a tree using our new topology.  We know we have alignments for the
  # speed-perturbed data (local/nnet3/run_ivector_common.sh made them), so use
  # those.
  if [ -f $tree_dir/final.mdl ]; then
    echo "$0: $tree_dir/final.mdl already exists, refusing to overwrite it."
    exit 1;
  fi
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --context-opts "--context-width=2 --central-position=1" \
      --leftmost-questions-truncate -1 \
      --cmd "$train_cmd" 4000 ${lores_train_data_dir} data/lang_chain $ali_dir $tree_dir
fi

xent_regularize=0.1
if [ $stage -le 17 ]; then
  mkdir -p $dir

  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')
  [ -z $num_targets ] && { echo "$0: error getting num-targets"; exit 1; }
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)
  lstm_opts="decay-time=20"
  label_delay=5

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=43 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-2,-1,0,1,2,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 dim=512
  relu-batchnorm-layer name=tdnn2 input=Append(-1,0,1) dim=512
  relu-batchnorm-layer name=tdnn3 input=Append(-1,0,1) dim=512

  # check steps/libs/nnet3/xconfig/lstm.py for the other options and defaults
  fast-lstmp-layer name=fastlstm1 cell-dim=512 recurrent-projection-dim=128 non-recurrent-projection-dim=128 delay=-3 $lstm_opts
  relu-batchnorm-layer name=tdnn4 input=Append(-3,0,3) dim=512
  relu-batchnorm-layer name=tdnn5 input=Append(-3,0,3) dim=512
  fast-lstmp-layer name=fastlstm2 cell-dim=512 recurrent-projection-dim=128 non-recurrent-projection-dim=128 delay=-3 $lstm_opts
  relu-batchnorm-layer name=tdnn6 input=Append(-3,0,3) dim=512
  relu-batchnorm-layer name=tdnn7 input=Append(-3,0,3) dim=512
  fast-lstmp-layer name=fastlstm3 cell-dim=512 recurrent-projection-dim=128 non-recurrent-projection-dim=128 delay=-3 $lstm_opts

  ## adding the layers for chain branch
  output-layer name=output input=fastlstm3 output-delay=$label_delay include-log-softmax=false dim=$num_targets max-change=1.5

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  output-layer name=output-xent input=fastlstm3 output-delay=$label_delay dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5

EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/

fi

if [ $stage -le 18 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/babel-$(date +'%m_%d_%H_%M')/s5d/$RANDOM/$dir/egs/storage $dir/egs/storage
  fi
  [ ! -d $dir/egs ] && mkdir -p $dir/egs/
  touch $dir/egs/.nodelete # keep egs around when that run dies.

 steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir $train_ivector_dir \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.dir "$common_egs_dir" \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width $chunk_width \
    --trainer.num-chunk-per-minibatch 128 \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs 4 \
    --trainer.optimization.num-jobs-initial 2 \
    --trainer.optimization.num-jobs-final 12 \
    --trainer.optimization.initial-effective-lrate 0.001 \
    --trainer.optimization.final-effective-lrate 0.0001 \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs true \
    --feat-dir $train_data_dir \
    --tree-dir $tree_dir \
    --lat-dir $lat_dir \
    --dir $dir
fi



if [ $stage -le 19 ]; then
  # Note: it might appear that this data/lang_chain directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/langp_test $dir $dir/graph
fi

exit 0
