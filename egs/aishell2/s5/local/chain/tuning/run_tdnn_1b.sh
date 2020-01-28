#!/usr/bin/env bash

# _1b is as _1a, but with pitch feats, i-vector and dropout schedule added, referenced from wsj

# basic info:
# steps/info/chain_dir_info.pl exp/chain/tdnn_1f_nopitch_ivec_sp/exp/chain/tdnn_1f_nopitch_ivec_sp/: num-iters=578 nj=2..8 num-params=19.3M dim=43+100->4520 combine=-0.082->-0.081 (over 6) xent:train/valid[384,577,final]=(-0.863,-0.752,-0.740/-0.901,-0.791,-0.784) logprob:train/valid[384,577,final]=(-0.083,-0.076,-0.075/-0.084,-0.077,-0.076)

# results:
# local/chain/compare_wer.sh exp/chain/tdnn_1f_nopitch_ivec_sp/
# Model                tdnn_1f_nopitch_ivec_sp
# Num. of params             19.3M
# WER(%)                     8.81
# Final train prob        -0.0749
# Final valid prob        -0.0756
# Final train prob (xent)   -0.7401
# Final valid prob (xent)   -0.7837

set -e

# configs for 'chain'
affix=all
stage=0
train_stage=-10
get_egs_stage=-10
dir=exp/chain/tdnn_1b  # Note: _sp will get added to this
decode_iter=

# training options
num_epochs=4
initial_effective_lrate=0.001
final_effective_lrate=0.0001
max_param_change=2.0
final_layer_normalize_target=0.5
num_jobs_initial=2
num_jobs_final=4
nj=15
minibatch_size=128
dropout_schedule='0,0@0.20,0.3@0.50,0'
frames_per_eg=150,110,90
remove_egs=true
common_egs_dir=
xent_regularize=0.1

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

dir=${dir}${affix:+_$affix}_sp
train_set=train
test_sets="dev test"
ali_dir=exp/tri3_ali
treedir=exp/chain/tri4_cd_tree_sp
lang=data/lang_chain

if [ $stage -le 5 ]; then
  mfccdir=mfcc_hires
  for datadir in ${train_set} ${test_sets}; do
  	utils/copy_data_dir.sh data/${datadir} data/${datadir}_hires
    utils/data/perturb_data_dir_volume.sh data/${datadir}_hires || exit 1;
	steps/make_mfcc_pitch.sh --mfcc-config conf/mfcc_hires.conf --pitch-config conf/pitch.conf \
      --nj $nj data/${datadir}_hires exp/make_mfcc/ ${mfccdir}
    steps/compute_cmvn_stats.sh data/${datadir}_hires exp/make_mfcc ${mfccdir}
    utils/data/limit_feature_dim.sh 0:39 data/${datadir}_hires data/${datadir}_hires_nopitch
    steps/compute_cmvn_stats.sh data/${datadir}_hires_nopitch exp/make_mfcc ${mfccdir}
  done
fi

# extract ivector from unified data using the trained
if [ $stage -le 6 ]; then
  echo "$0: computing a subset of data to train the diagonal UBM."
  # We'll use about a quarter of the data.
  mkdir -p exp/chain/diag_ubm_${affix}
  temp_data_root=exp/chain/diag_ubm_${affix}

  num_utts_total=$(wc -l < data/${train_set}_hires_nopitch/utt2spk)
  num_utts=$[$num_utts_total/4]
  utils/data/subset_data_dir.sh data/${train_set}_hires_nopitch \
    $num_utts ${temp_data_root}/${train_set}_subset

  echo "$0: computing a PCA transform from the hires data."
  steps/online/nnet2/get_pca_transform.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" \
    --max-utts 10000 --subsample 2 \
    --dim $(feat-to-dim scp:${temp_data_root}/${train_set}_subset/feats.scp -) \
    ${temp_data_root}/${train_set}_subset \
    exp/chain/pca_transform_${affix}

  echo "$0: training the diagonal UBM."
  # Use 512 Gaussians in the UBM.
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj $nj \
    --num-frames 700000 \
    --num-threads 8 \
    ${temp_data_root}/${train_set}_subset 512 \
    exp/chain/pca_transform_${affix} exp/chain/diag_ubm_${affix}

  echo "$0: training the iVector extractor"
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj $nj \
    data/${train_set}_hires_nopitch exp/chain/diag_ubm_${affix} \
    exp/chain/extractor_${affix} || exit 1;

  for datadir in ${train_set} ${test_sets}; do
    steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 data/${datadir}_hires_nopitch data/${datadir}_hires_nopitch_max2
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
      data/${datadir}_hires_nopitch_max2 exp/chain/extractor_${affix} exp/chain/ivectors_${datadir}_${affix} || exit 1;
  done
fi

if [ $stage -le 7 ]; then
  # Get the alignments as lattices (gives the LF-MMI training more freedom).
  # use the same num-jobs as the alignments
  nj=$(cat $ali_dir/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" data/$train_set \
    data/lang exp/tri3 exp/tri4_sp_lats
  rm exp/tri4_sp_lats/fsts.*.gz # save space
fi

if [ $stage -le 8 ]; then
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

if [ $stage -le 9 ]; then
  # Build a tree using our new topology. This is the critically different
  # step compared with other recipes.
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
    --context-opts "--context-width=2 --central-position=1" \
    --cmd "$train_cmd" 5000 data/$train_set $lang $ali_dir $treedir
fi

if [ $stage -le 10 ]; then
  echo "$0: creating neural net configs using the xconfig parser";
  feat_dim=$(feat-to-dim scp:data/${train_set}_hires/feats.scp -)
  num_targets=$(tree-info $treedir/tree | grep num-pdfs | awk '{print $2}')
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)
  opts="l2-regularize=0.002"
  linear_opts="orthonormal-constraint=1.0"
  output_opts="l2-regularize=0.0005 bottleneck-dim=256"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=$feat_dim name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-dropout-layer name=tdnn1 $opts dim=1280
  linear-component name=tdnn2l dim=256 $linear_opts input=Append(-1,0)
  relu-batchnorm-dropout-layer name=tdnn2 $opts input=Append(0,1) dim=1280
  linear-component name=tdnn3l dim=256 $linear_opts
  relu-batchnorm-dropout-layer name=tdnn3 $opts dim=1280
  linear-component name=tdnn4l dim=256 $linear_opts input=Append(-1,0)
  relu-batchnorm-dropout-layer name=tdnn4 $opts input=Append(0,1) dim=1280
  linear-component name=tdnn5l dim=256 $linear_opts
  relu-batchnorm-dropout-layer name=tdnn5 $opts dim=1280 input=Append(tdnn5l, tdnn3l)
  linear-component name=tdnn6l dim=256 $linear_opts input=Append(-3,0)
  relu-batchnorm-dropout-layer name=tdnn6 $opts input=Append(0,3) dim=1280
  linear-component name=tdnn7l dim=256 $linear_opts input=Append(-3,0)
  relu-batchnorm-dropout-layer name=tdnn7 $opts input=Append(0,3,tdnn6l,tdnn4l,tdnn2l) dim=1280
  linear-component name=tdnn8l dim=256 $linear_opts input=Append(-3,0)
  relu-batchnorm-dropout-layer name=tdnn8 $opts input=Append(0,3) dim=1280
  linear-component name=tdnn9l dim=256 $linear_opts input=Append(-3,0)
  relu-batchnorm-dropout-layer name=tdnn9 $opts input=Append(0,3,tdnn8l,tdnn6l,tdnn4l) dim=1280
  linear-component name=tdnn10l dim=256 $linear_opts input=Append(-3,0)
  relu-batchnorm-dropout-layer name=tdnn10 $opts input=Append(0,3) dim=1280
  linear-component name=tdnn11l dim=256 $linear_opts input=Append(-3,0)
  relu-batchnorm-dropout-layer name=tdnn11 $opts input=Append(0,3,tdnn10l,tdnn8l,tdnn6l) dim=1280
  linear-component name=prefinal-l dim=256 $linear_opts

  relu-batchnorm-layer name=prefinal-chain input=prefinal-l $opts dim=1280
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

  relu-batchnorm-layer name=prefinal-xent input=prefinal-l $opts dim=1280
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts

EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 11 ]; then
  #if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
  #  utils/create_split_dir.pl \
  #   /export/b0{5,6,7,8}/$USER/kaldi-data/egs/aishell-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  #fi

  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir exp/chain/ivectors_${train_set}_${affix} \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width $frames_per_eg \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.num-chunk-per-minibatch $minibatch_size \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.num-jobs-initial $num_jobs_initial \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.initial-effective-lrate $initial_effective_lrate \
    --trainer.optimization.final-effective-lrate $final_effective_lrate \
    --trainer.max-param-change $max_param_change \
    --cleanup.remove-egs $remove_egs \
    --feat-dir data/${train_set}_hires \
    --tree-dir $treedir \
    --lat-dir exp/tri4_sp_lats \
    --dir $dir  || exit 1;
fi

if [ $stage -le 12 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_test $dir $dir/graph
fi

graph_dir=$dir/graph
if [ $stage -le 13 ]; then
  for test_set in $test_sets; do
    nj=$(wc -l data/${test_set}_hires/spk2utt | awk '{print $1}')
    steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
      --nj $nj --cmd "$decode_cmd" \
      --online-ivector-dir exp/chain/ivectors_${test_set}_${affix} \
      $graph_dir data/${test_set}_hires $dir/decode_${test_set} || exit 1;
  done
fi

echo "local/chain/run_tdnn.sh succeeded"
exit 0;
