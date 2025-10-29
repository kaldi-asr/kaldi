#!/bin/bash

# 1k101 is as 1k100 but removing 3 layers and adding l2-regularize to attention layers
# Similar WER, but valid objf is slightly better, and fewer params.
# local/chain/compare_wer.sh exp/chain_online_cmn/tdnn1k_sp exp/chain_online_cmn/tdnn1k81_sp exp/chain_online_cmn/tdnn1k81b_sp exp/chain_online_cmn/tdnn1k93_sp exp/chain_online_cmn/tdnn1k100_sp exp/chain_online_cmn/tdnn1k101_sp
# System                tdnn1k_sp tdnn1k81_sp tdnn1k81b_sp tdnn1k93_sp tdnn1k100_sp tdnn1k101_sp
#WER dev_clean_2 (tgsmall)      10.61     10.68     10.59     10.74     11.11     11.19
#WER dev_clean_2 (tglarge)       7.35      7.19      7.28      7.46      7.75      7.82
# Final train prob        -0.0618   -0.0567   -0.0564   -0.0581   -0.0446   -0.0464
# Final valid prob        -0.0786   -0.0751   -0.0762   -0.0753   -0.0657   -0.0656
# Final train prob (xent)   -1.4308   -1.2992   -1.2932   -1.3216   -1.0546   -1.1210
# Final valid prob (xent)   -1.5418   -1.4271   -1.4268   -1.4474   -1.2080   -1.2639
# Num-params                 5207856   5212464   5212464             7087440   6272928

# 1k100 is as 1k93 but adding attention.
# 1k93 is as 1k81 but adding dropout.
# 1k81 is as 1k79 but reducing the dims for the time-stride=3 layers from 96 to 64
#   (See also 80 where both dims were reduced to 80).
# local/chain/compare_wer.sh exp/chain_online_cmn/tdnn1k_sp exp/chain_online_cmn/tdnn1k79_sp exp/chain_online_cmn/tdnn1k80_sp exp/chain_online_cmn/tdnn1k81_sp exp/chain_online_cmn/tdnn1k81b_sp
# System                tdnn1k_sp tdnn1k79_sp tdnn1k80_sp tdnn1k81_sp tdnn1k81b_sp
#WER dev_clean_2 (tgsmall)      10.61     10.53     10.54     10.58     10.59
#WER dev_clean_2 (tglarge)       7.35      7.28      7.23      7.17      7.28
# Final train prob        -0.0618   -0.0558   -0.0568   -0.0563   -0.0564
# Final valid prob        -0.0786   -0.0751   -0.0752   -0.0757   -0.0762
# Final train prob (xent)   -1.4308   -1.2822   -1.2951   -1.2989   -1.2932
# Final valid prob (xent)   -1.5418   -1.4095   -1.4219   -1.4303   -1.4268
# Num-params                 5207856   5802288   5138736   5212464   5212464

# local/chain/compare_wer.sh exp/chain_online_cmn/tdnn1k_sp exp/chain_online_cmn/tdnn1k79_sp exp/chain_online_cmn/tdnn1k80_sp
# System                tdnn1k_sp tdnn1k79_sp tdnn1k80_sp
#WER dev_clean_2 (tgsmall)      10.61     10.53     10.54
#WER dev_clean_2 (tglarge)       7.35      7.28      7.23
# Final train prob        -0.0618   -0.0558   -0.0568
# Final valid prob        -0.0786   -0.0751   -0.0752
# Final train prob (xent)   -1.4308   -1.2822   -1.2951
# Final valid prob (xent)   -1.5418   -1.4095   -1.4219
# Num-params                 5207856   5802288   5138736

# 1k79 is as 1k74 but with wider layer dim and narrower non-splicing layers.
# 1k74 is like 1k72 but with no-splice layers between the initial tdnnf layers,
#  and removing 2 layers.
# WER not better but promising objf
# local/chain/compare_wer.sh exp/chain_online_cmn/tdnn1k_sp exp/chain_online_cmn/tdnn1k70_sp exp/chain_online_cmn/tdnn1k71_sp exp/chain_online_cmn/tdnn1k72_sp exp/chain_online_cmn/tdnn1k74_sp
# System                tdnn1k_sp tdnn1k70_sp tdnn1k71_sp tdnn1k72_sp tdnn1k74_sp
#WER dev_clean_2 (tgsmall)      10.61     11.33     10.88     10.80     10.82
#WER dev_clean_2 (tglarge)       7.35      7.65      7.36      7.27      7.38
# Final train prob        -0.0618   -0.0667   -0.0646   -0.0582   -0.0587
# Final valid prob        -0.0786   -0.0813   -0.0807   -0.0778   -0.0765
# Final train prob (xent)   -1.4308   -1.5438   -1.5218   -1.3131   -1.3369
# Final valid prob (xent)   -1.5418   -1.6445   -1.6326   -1.4403   -1.4616
# Num-params                 5207856   5249584   5249584   5249584   5249584


# 1k72 is like 1k71 but with less l2-regularize (less by one third)
# 1k71 is like 1k70 but bypass-scale=0.8
# 1k70 is like 1k but with alternating context / no-context.

# 1k is like 1j, while it introduces 'apply-cmvn-online' that does
# cmn normalization both for i-extractor and TDNN input.

# local/chain/compare_wer.sh --online exp/chain/tdnn1j_sp exp/chain_online_cmn/tdnn1k_sp
# System                tdnn1j_sp tdnn1k_sp
#WER dev_clean_2 (tgsmall)      10.97     10.64
#             [online:]         10.97     10.62
#WER dev_clean_2 (tglarge)       7.57      7.17
#             [online:]          7.65      7.16
# Final train prob        -0.0623   -0.0618
# Final valid prob        -0.0793   -0.0793
# Final train prob (xent)   -1.4448   -1.4376
# Final valid prob (xent)   -1.5605   -1.5461
# Num-params                 5210944   5210944

# steps/info/chain_dir_info.pl exp/chain/tdnn1j_sp
# exp/chain/tdnn1j_sp: num-iters=34 nj=2..5 num-params=5.2M dim=40+100->2336 combine=-0.068->-0.064 (over 4) xent:train/valid[21,33,final]=(-1.65,-1.48,-1.44/-1.77,-1.58,-1.56) logprob:train/valid[21,33,final]=(-0.076,-0.068,-0.062/-0.091,-0.084,-0.079)

# steps/info/chain_dir_info.pl exp/chain_online_cmn/tdnn1k_sp
# exp/chain_online_cmn/tdnn1k_sp: num-iters=34 nj=2..5 num-params=5.2M dim=40+100->2336 combine=-0.067->-0.062 (over 5) xent:train/valid[21,33,final]=(-1.63,-1.47,-1.44/-1.73,-1.57,-1.55) logprob:train/valid[21,33,final]=(-0.074,-0.067,-0.062/-0.093,-0.085,-0.079)

# Set -e here so that we catch if any executable fails immediately
set -euo pipefail

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=0
decode_nj=10
train_set=train_clean_5
test_sets=dev_clean_2
gmm=tri3b
nnet3_affix=_online_cmn

# Setting 'online_cmvn' to true replaces 'apply-cmvn' by
# 'apply-cmvn-online' both for i-vector extraction and TDNN input.
# The i-vector extractor uses the config 'conf/online_cmvn.conf' for
# both the UBM and the i-extractor. The TDNN input is configured via
# '--feat.cmvn-opts' that is set to the same config, so we use the
# same cmvn for i-extractor and the TDNN input.
online_cmvn=true

# The rest are configs specific to this script.  Most of the parameters
# are just hardcoded at this level, in the commands below.
affix=1k101   # affix for the TDNN directory name
tree_affix=
train_stage=-10
get_egs_stage=-10
decode_iter=

# training options
# training chunk-options
chunk_width=140,100,160
common_egs_dir=
dropout_schedule='0,0@0.20,0.25@0.50,0'
xent_regularize=0.1

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
                                  --online-cmvn-iextractor $online_cmvn \
                                  --nnet3-affix "$nnet3_affix" || exit 1;

# Problem: We have removed the "train_" prefix of our training set in
# the alignment directory names! Bad!
gmm_dir=exp/$gmm
ali_dir=exp/${gmm}_ali_${train_set}_sp
tree_dir=exp/chain${nnet3_affix}/tree_sp${tree_affix:+_$tree_affix}
lang=data/lang_chain
lat_dir=exp/chain${nnet3_affix}/${gmm}_${train_set}_sp_lats
dir=exp/chain${nnet3_affix}/tdnn${affix}_sp
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


if [ $stage -le 13 ]; then
  mkdir -p $dir
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)

  tdnn_opts="l2-regularize=0.02 dropout-proportion=0.0 dropout-per-dim-continuous=true"
  tdnnf_opts="l2-regularize=0.02 dropout-proportion=0.0 bypass-scale=0.8"
  attention_opts="l2-regularize=0.02 num-heads=2 num-left-inputs=5 num-left-inputs-required=1 num-right-inputs=2 num-right-inputs-required=1 dropout-proportion=0.0 bypass-scale=0.8"
  linear_opts="l2-regularize=0.02 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.02"
  output_opts="l2-regularize=0.01"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # this takes the MFCCs and generates filterbank coefficients.  The MFCCs
  # are more compressible so we prefer to dump the MFCCs to disk rather
  # than filterbanks.
  idct-layer name=idct input=input dim=40 cepstral-lifter=22 affine-transform-file=$dir/configs/idct.mat
  batchnorm-component name=batchnorm0 input=idct
  spec-augment-layer name=spec-augment freq-max-proportion=0.5 time-zeroed-proportion=0.2 time-mask-max-frames=20

  delta-layer name=delta input=spec-augment
  no-op-component name=input2 input=Append(delta, Scale(0.4, ReplaceIndex(ivector, t, 0)))

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 $tdnn_opts dim=768 input=input2
  tdnnf-layer name=tdnnf2 $tdnnf_opts dim=768 bottleneck-dim=128 time-stride=1
  residual-attention-layer name=attention3 $attention_opts time-stride=1
  tdnnf-layer name=tdnnf4 $tdnnf_opts dim=768 bottleneck-dim=128 time-stride=0
  tdnnf-layer name=tdnnf5 $tdnnf_opts dim=768 bottleneck-dim=128 time-stride=1
  residual-attention-layer name=attention6 $attention_opts time-stride=1
  tdnnf-layer name=tdnnf7 $tdnnf_opts dim=768 bottleneck-dim=128 time-stride=0
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=768 bottleneck-dim=64 time-stride=3
  residual-attention-layer name=attention9 $attention_opts time-stride=3
  tdnnf-layer name=tdnnf10 $tdnnf_opts dim=768 bottleneck-dim=128 time-stride=0
  tdnnf-layer name=tdnnf11 $tdnnf_opts dim=768 bottleneck-dim=64 time-stride=3
  residual-attention-layer name=attention12 $attention_opts time-stride=3
  tdnnf-layer name=tdnnf13 $tdnnf_opts dim=768 bottleneck-dim=128 time-stride=0
  tdnnf-layer name=tdnnf14 $tdnnf_opts dim=768 bottleneck-dim=64 time-stride=3
  residual-attention-layer name=attention15 $attention_opts time-stride=3
  tdnnf-layer name=tdnnf16 $tdnnf_opts dim=768 bottleneck-dim=128 time-stride=0
  linear-component name=prefinal-l dim=192 $linear_opts

  ## adding the layers for chain branch
  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts small-dim=192 big-dim=768
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

  # adding the layers for xent branch
  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts small-dim=192 big-dim=768
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi


if [ $stage -le 14 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/mini_librispeech-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/chain/train.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.online-ivector-dir=$train_ivector_dir \
    --feat.cmvn-opts="--config=conf/online_cmvn.conf" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient=0.1 \
    --chain.l2-regularize=0.0 \
    --chain.apply-deriv-weights=false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=20 \
    --trainer.frames-per-iter=3000000 \
    --trainer.optimization.num-jobs-initial=2 \
    --trainer.optimization.num-jobs-final=5 \
    --trainer.optimization.initial-effective-lrate=0.002 \
    --trainer.optimization.final-effective-lrate=0.0002 \
    --trainer.num-chunk-per-minibatch=128,64 \
    --egs.chunk-width=$chunk_width \
    --egs.dir="$common_egs_dir" \
    --egs.opts="--frames-overlap-per-eg 0 --online-cmvn $online_cmvn" \
    --cleanup.remove-egs=$remove_egs \
    --use-gpu=true \
    --reporting.email="$reporting_email" \
    --feat-dir=$train_data_dir \
    --tree-dir=$tree_dir \
    --lat-dir=$lat_dir \
    --dir=$dir  || exit 1;
fi

if [ $stage -le 15 ]; then
  # Note: it's not important to give mkgraph.sh the lang directory with the
  # matched topology (since it gets the topology file from the model).
  utils/mkgraph.sh \
    --self-loop-scale 1.0 data/lang_test_tgsmall \
    $tree_dir $tree_dir/graph_tgsmall || exit 1;
fi

if [ $stage -le 16 ]; then
  frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    (
      nspk=$(wc -l <data/${data}_hires/spk2utt)
      steps/nnet3/decode.sh \
          --acwt 1.0 --post-decode-acwt 10.0 \
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

# Not testing the 'looped' decoding separately, because for
# TDNN systems it would give exactly the same results as the
# normal decoding.

if $test_online_decoding && [ $stage -le 17 ]; then
  # note: if the features change (e.g. you add pitch features), you will have to
  # change the options of the following command line.
  steps/online/nnet3/prepare_online_decoding.sh \
    --mfcc-config conf/mfcc_hires.conf \
    --online-cmvn-config conf/online_cmvn.conf \
    $lang exp/nnet3${nnet3_affix}/extractor ${dir} ${dir}_online

  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    (
      nspk=$(wc -l <data/${data}_hires/spk2utt)
      # note: we just give it "data/${data}" as it only uses the wav.scp, the
      # feature type does not matter.
      steps/online/nnet3/decode.sh \
        --acwt 1.0 --post-decode-acwt 10.0 \
        --nj $nspk --cmd "$decode_cmd" \
        $tree_dir/graph_tgsmall data/${data} ${dir}_online/decode_tgsmall_${data} || exit 1
      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
        data/lang_test_{tgsmall,tglarge} \
        data/${data}_hires ${dir}_online/decode_{tgsmall,tglarge}_${data} || exit 1
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi


exit 0;
