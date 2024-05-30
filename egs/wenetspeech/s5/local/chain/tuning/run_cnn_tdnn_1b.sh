#!/usr/bin/env bash
# Copyright 2021  ASLP, NWPU (Author: Hang Lyu)
#                 Mobvoi Inc (Author: Binbin Zhang)
# Apache 2.0

# 1b is as 1a but adding SpecAugment.

# Set -e here so that we catch if any executable fails immediately
set -euo pipefail

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=0
train_nj=50
decode_nj=50
train_set="train_l"
gmm=tri3b
nnet3_affix=_cleaned

# The rest are configs specific to this script.  Most of the parameters
# are just hardcoded at this level, in the commands below.
affix=_1b   # affix for the TDNN directory name
tree_affix=
train_stage=-10
get_egs_stage=-10
decode_iter=

# training options
chunk_width=150,110,100
srand=0
remove_egs=false
common_egs_dir=
reporting_email=
num_epochs=5
frames_per_iter=3000000
initial_effective_lrate=0.00015
final_effective_lrate=0.000015
num_jobs_initial=8
num_jobs_final=8
xent_regularize=0.1

# decode options
test_sets=""
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
# nnet3 setup, and you can skip them by setting "--stage 11" if you have already
# run those things.
#local/chain/run_ivector_common.sh --stage $stage \
#                                  --train-set $train_set \
#                                  --test-sets "$test_sets" \
#                                  --gmm $gmm \
#                                  --nnet3-affix "$nnet3_affix" || exit 1;

# We don't conduct the techniques about SpecAug and Ivector.
# To accommodate with Kaldi's customary nomenclature, we masquerade the
# 'train_set' as the 'train_set_sp' dataset.

# Prepare the hires mfcc features and alignment.
if [ $stage -le 10 ]; then
  ln -sf $train_set data/${train_set}_sp
  for part in ${train_set}_sp $test_sets; do
    utils/copy_data_dir.sh data/$part data/${part}_hires
    steps/make_mfcc.sh --nj $train_nj --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/${part}_hires || exit 1;
    steps/compute_cmvn_stats.sh data/${part}_hires || exit 1;
    utils/fix_data_dir.sh data/${part}_hires
  done
  steps/align_fmllr.sh --stage 0 --nj $train_nj --cmd "$train_cmd" \
    data/${train_set}_sp data/lang exp/${train_set}/$gmm \
    exp/${train_set}/${gmm}_ali_${train_set}_sp
fi


gmm_dir=exp/${train_set}/$gmm
ali_dir=exp/${train_set}/${gmm}_ali_${train_set}_sp
tree_dir=exp/${train_set}/chain${nnet3_affix}/tree_sp${tree_affix:+_$tree_affix}
lang=data/lang_chain
lat_dir=exp/${train_set}/chain${nnet3_affix}/${gmm}_${train_set}_sp_lats
dir=exp/${train_set}/chain${nnet3_affix}/cnn_tdnn${affix}_sp
train_data_dir=data/${train_set}_sp_hires
lores_train_data_dir=data/${train_set}_sp

for f in $gmm_dir/final.mdl $train_data_dir/feats.scp \
    $lores_train_data_dir/feats.scp $ali_dir/ali.1.gz; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done

# Please take this as a reference on how to specify all the options of
# local/chain/run_chain_common.sh
local/chain/run_chain_common.sh --stage $stage \
                                --gmm-dir $gmm_dir \
                                --ali-dir $ali_dir \
                                --lores-train-data-dir ${lores_train_data_dir} \
                                --lang $lang \
                                --lat-dir $lat_dir \
                                --num-leaves 9000 \
                                --tree-dir $tree_dir || exit 1;

if [ $stage -le 14 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)
  cnn_opts="l2-regularize=0.01"
  ivector_affine_opts="l2-regularize=0.0"
  tdnnf_first_opts="l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.0"
  tdnnf_opts="l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.75"
  linear_opts="l2-regularize=0.008 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.008"
  output_opts="l2-regularize=0.005"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=40 name=input

  # this takes the MFCCs and generates filterbank coefficients.  The MFCCs
  # are more compressible so we prefer to dump the MFCCs to disk rather
  # than filterbanks.
  idct-layer name=idct input=input dim=40 cepstral-lifter=22 affine-transform-file=$dir/configs/idct.mat
  batchnorm-component name=idct-batchnorm input=idct
  spec-augment-layer name=spec-augment freq-max-proportion=0.5 time-zeroed-proportion=0.2 time-mask-max-frames=20 include-in-init=true

  conv-relu-batchnorm-layer name=cnn1 $cnn_opts height-in=40 height-out=40 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=64 learning-rate-factor=0.333 max-change=0.25
  conv-relu-batchnorm-layer name=cnn2 $cnn_opts height-in=40 height-out=40 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=64
  conv-relu-batchnorm-layer name=cnn3 $cnn_opts height-in=40 height-out=20 height-subsample-out=2 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=128
  conv-relu-batchnorm-layer name=cnn4 $cnn_opts height-in=20 height-out=20 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=128
  conv-relu-batchnorm-layer name=cnn5 $cnn_opts height-in=20 height-out=10 height-subsample-out=2 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=256
  conv-relu-batchnorm-layer name=cnn6 $cnn_opts height-in=10 height-out=10 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=256

  # the first TDNN-F layer has no bypass (since dims don't match), and a larger bottleneck so the
  # information bottleneck doesn't become a problem.
  tdnnf-layer name=tdnnf7 $tdnnf_first_opts dim=1536 bottleneck-dim=256 time-stride=0
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf9 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf10 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf11 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf12 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf13 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf14 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf15 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf16 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  attention-relu-renorm-layer name=attention1 num-heads=30 value-dim=30 key-dim=15 time-stride=3 num-left-inputs=15 num-right-inputs=6
  tdnnf-layer name=tdnnf17 $tdnnf_opts dim=1560 bottleneck-dim=160 time-stride=3
  fast-lstmp-layer name=lstm1 cell-dim=1024 recurrent-projection-dim=256 non-recurrent-projection-dim=256 decay-time=20 delay=-3
  tdnnf-layer name=tdnnf18 $tdnnf_opts dim=512 bottleneck-dim=160 time-stride=3
  fast-lstmp-layer name=lstm2 cell-dim=1024 recurrent-projection-dim=256 non-recurrent-projection-dim=256 decay-time=20 delay=-3
  linear-component name=prefinal-l dim=256 $linear_opts

  ## adding the layers for chain branch
  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts small-dim=256 big-dim=1536
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

  # adding the layers for xent branch
  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts small-dim=256 big-dim=1536
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi


if [ $stage -le 15 ]; then
  steps/nnet3/chain/train.py --stage=$train_stage \
    --cmd="$train_cmd" \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient=0.1 \
    --chain.l2-regularize=0.0 \
    --chain.apply-deriv-weights=false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=$num_epochs \
    --trainer.frames-per-iter=$frames_per_iter \
    --trainer.optimization.num-jobs-initial=$num_jobs_initial \
    --trainer.optimization.num-jobs-final=$num_jobs_final \
    --trainer.optimization.initial-effective-lrate=$initial_effective_lrate \
    --trainer.optimization.final-effective-lrate=$final_effective_lrate \
    --trainer.num-chunk-per-minibatch=128,64 \
    --egs.chunk-width=$chunk_width \
    --egs.dir="$common_egs_dir" \
    --egs.opts="--frames-overlap-per-eg 0" \
    --egs.stage=$get_egs_stage \
    --egs.cmd="$egs_cmd" \
    --cleanup.remove-egs=$remove_egs \
    --use-gpu=true \
    --reporting.email="$reporting_email" \
    --feat-dir=$train_data_dir \
    --tree-dir=$tree_dir \
    --lat-dir=$lat_dir \
    --dir=$dir  || exit 1;
fi

graph_dir=$dir/graph
if [ $stage -le 16 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 --remove-oov data/lang_test $dir $graph_dir
fi

iter_opts=
if [ ! -z $decode_iter ]; then
  iter_opts=" --iter $decode_iter "
fi
if [ $stage -le 17 ]; then
  rm $dir/.error 2>/dev/null || true
  for part_set in $test_sets; do
      (
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $decode_nj --cmd "$decode_cmd" $iter_opts \
          $graph_dir data/${part_set}_hires $dir/decode_${part_set}${decode_iter:+_$decode_iter} || exit 1
      ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi

if $test_online_decoding && [ $stage -le 18 ]; then
  # note: if the features change (e.g. you add pitch features), you will have to
  # change the options of the following command line.
  steps/online/nnet3/prepare_online_decoding.sh \
       --mfcc-config conf/mfcc_hires.conf \
       $lang exp/nnet3${nnet3_affix}/extractor $dir ${dir}_online

  rm ${dir}_online/.error 2>/dev/null || true
  for part_set in $test_sets; do
    (
      nspk=$(wc -l <data/${part_set}_hires/spk2utt)
      # note: we just give it "data/${part_set}" as it only uses the wav.scp, the
      # feature type does not matter.
      steps/online/nnet3/decode.sh \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $nspk --cmd "$decode_cmd" \
          $graph_dir data/${part_set} ${dir}_online/decode_${part_set} || exit 1

    ) || touch ${dir}_online/.error &
  done
  wait
  if [ -f ${dir}_online/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi

if [ $stage -le 19 ]; then
  # decode with rnnlm
  # If an rnnlm has been provided, we should set the "stage" to 4 for testing.
  ./local/wenetspeech_run_rnnlm.sh --stage 0 \
    --train-stage -10 \
    --ngram-order 5 \
    --num-epoch 8 \
    --num-jobs-initial 1 \
    --num-jobs-final 8 \
    --words-per-split 400000 \
    --text data/corpus/lm_text \
    --ac-model-dir $dir \
    --test-sets "$test_sets" \
    --decode-iter "$decode_iter" \
    --lang data/lang_test \
    --dir exp/rnnlm
fi

exit 0;
