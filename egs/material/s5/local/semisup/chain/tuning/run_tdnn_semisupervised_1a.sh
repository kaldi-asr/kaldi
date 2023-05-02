#!/usr/bin/env bash

# Copyright 2017  Vimal Manohar
#           2019  Yiming Wang
# Apache 2.0

# This script is semi-supervised recipe with ~40 hours of supervised data
# and ~320 hours unsupervised data with naive splitting.
# Based on "Semi-Supervised Training of Acoustic Models using Lattice-Free MMI",
# Vimal Manohar, Hossein Hadian, Daniel Povey, Sanjeev Khudanpur, ICASSP 2018
# http://www.danielpovey.com/files/2018_icassp_semisupervised_mmi.pdf
# local/semisup/run_100k.sh shows how to call this.

# This version of script uses only supervised data for i-vector extractor
# training as against using the combined data.

# This script uses the same tree as that for the seed model.
# See the comments in the script about how to change these.

# Unsupervised set: eval1_2_3_segmented (320 hours)
# unsup_frames_per_eg=150
# Deriv weights: Lattice posterior of best path pdf
# Unsupervised weight: 1.0
# Weights for phone LM (supervised, unsupervised): 3,2
# LM for decoding unsupervised data: ngram
# Supervision: Naive split lattices

# output-0 and output-1 are for superivsed and unsupervised data respectively.

# Semi-supervised training            train_sup
# WER on dev                          18.70
# WER on test                         18.18
# Final output-0 train prob           -0.1345
# Final output-0 valid prob           -0.1547
# Final output-0 train prob (xent)    -1.3683
# Final output-0 valid prob (xent)    -1.4077
# Final output-1 train prob           -0.6856
# Final output-1 valid prob           -0.6815
# Final output-1 train prob (xent)    -1.1224
# Final output-1 valid prob (xent)    -1.2218

set -u -e -o pipefail

stage=0   # Start from -1 for supervised seed system training
nj=30
test_sets="dev"

exp_root=exp/semisup
affix=1a  # affix for semi-supervised chain system

# LSTM/chain options
train_stage=-10
xent_regularize=0.1

# we don't need extra left/right context for TDNN systems.
chunk_left_context=0
chunk_right_context=0
dropout_schedule='0,0@0.20,0.3@0.50,0'
num_epochs=12

# training options
srand=0
remove_egs=true

# Datasets -- Expects data/$supervised_set and data/$unsupervised_set to be
# present
supervised_set=train
unsupervised_set=eval1_2_3_segmented

# Input seed system
sup_chain_dir=exp/semisup/chain/tdnn_1a_sp  # supervised chain system
sup_lat_dir=exp/semisup/chain/tri3_${supervised_set}_sp_lats  # Seed model options
sup_tree_dir=exp/semisup/chain/tree_sp  # tree directory for supervised chain system
ivector_root_dir=exp/nnet3  # i-vector extractor root directory

# Semi-supervised options
supervision_weights=1.0,1.0   # Weights for supervised, unsupervised data egs.
                              # Can be used to scale down the effect of unsupervised data
                              # by using a smaller scale for it e.g. 1.0,0.3
lm_weights=3,2  # Weights on phone counts from supervised, unsupervised data for denominator FST creation

sup_egs_dir=   # Supply this to skip supervised egs creation
unsup_egs_dir=  # Supply this to skip unsupervised egs creation
unsup_egs_opts=  # Extra options to pass to unsupervised egs creation

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
if [ -f ./path.sh ]; then . ./path.sh; fi
. ./utils/parse_options.sh

# The following can be replaced with the versions that model
# UNK using phone LM. $sup_lat_dir should also ideally be changed.
unsup_decode_lang=data/lang_combined_test
test_lang=data/lang_combined_test

dir=$exp_root/chain/tdnn_semisup_${affix}

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

supervised_set_perturbed=${supervised_set}_sp

sup_ivector_dir=$ivector_root_dir/ivectors_${supervised_set_perturbed}_hires

graphdir=$sup_chain_dir/graph_combined

for f in data/${supervised_set_perturbed}/feats.scp \
  data/${supervised_set_perturbed}_hires/feats.scp \
  $ivector_root_dir/extractor/final.ie $sup_ivector_dir/ivector_online.scp \
  $sup_lat_dir/lat.1.gz $sup_tree_dir/ali.1.gz \
  $unsup_decode_lang/G.fst; do
  if [ ! -f $f ]; then
    echo "$0: Could not find file $f"
    exit 1
  fi
done

if [ $stage -le 1 ]; then
  if [ ! -f $graphdir/HCLG.fst ]; then
    utils/mkgraph.sh --self-loop-scale 1.0 $unsup_decode_lang $sup_chain_dir $graphdir
  fi
fi

# Prepare the speed-perturbed unsupervised data directory
if [ $stage -le 2 ]; then
  if [ -f data/${unsupervised_set}_sp_hires/feats.scp ]; then
    echo "$0: data/${unsupervised_set}_sp_hires/feats.scp exists. Remove it or re-run from next stage"
    exit 1
  fi

  utils/data/perturb_data_dir_speed_3way.sh data/$unsupervised_set data/${unsupervised_set}_sp_hires
  utils/data/perturb_data_dir_volume.sh data/${unsupervised_set}_sp_hires

  steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" \
    --mfcc-config conf/mfcc_hires.conf data/${unsupervised_set}_sp_hires || exit 1
fi
unsupervised_set_perturbed=${unsupervised_set}_sp

# Extract i-vectors for the unsupervised data
if [ $stage -le 3 ]; then
  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
    data/${unsupervised_set_perturbed}_hires data/${unsupervised_set_perturbed}_max2_hires

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
    data/${unsupervised_set_perturbed}_max2_hires $ivector_root_dir/extractor \
    $ivector_root_dir/ivectors_${unsupervised_set_perturbed}_hires || exit 1
fi

# Decode unsupervised data and write lattices in non-compact
# undeterminized format
# Set --skip-scoring to false in order to score the unsupervised data
if [ $stage -le 4 ]; then
  echo "$0: getting the decoding lattices for the unsupervised subset using the chain model at: $sup_chain_dir"
  steps/nnet3/decode_semisup.sh --num-threads 4 --nj $nj --cmd "$decode_cmd" \
            --acwt 1.0 --post-decode-acwt 10.0 --write-compact false --skip-scoring true \
            --extra-left-context $chunk_left_context --extra-right-context $chunk_right_context \
            --extra-left-context-initial 0 --extra-right-context-final 0 \
            --frames-per-chunk 150 \
            --online-ivector-dir $ivector_root_dir/ivectors_${unsupervised_set_perturbed}_hires \
            --scoring-opts "--min-lmwt 10 --max-lmwt 10" --word-determinize false \
            $graphdir data/${unsupervised_set_perturbed}_hires $sup_chain_dir/decode_${unsupervised_set_perturbed}
fi

# Get best path alignment and lattice posterior of best path alignment to be
# used as frame-weights in lattice-based training
if [ $stage -le 5 ]; then
  steps/best_path_weights.sh --cmd "${train_cmd}" --acwt 0.1 \
    data/${unsupervised_set_perturbed}_hires \
    $sup_chain_dir/decode_${unsupervised_set_perturbed} \
    $sup_chain_dir/best_path_${unsupervised_set_perturbed}
fi

frame_subsampling_factor=1
if [ -f $sup_chain_dir/frame_subsampling_factor ]; then
  frame_subsampling_factor=$(cat $sup_chain_dir/frame_subsampling_factor)
fi
cmvn_opts=$(cat $sup_chain_dir/cmvn_opts) || exit 1

diff $sup_tree_dir/tree $sup_chain_dir/tree || { echo "$0: $sup_tree_dir/tree and $sup_chain_dir/tree differ"; exit 1; }

# Uncomment the following lines if you need to build new tree using both
# supervised and unsupervised data. This may help if amount of
# supervised data used to train the seed system tree is very small.
# unsupervised data

# tree_affix=semisup
# treedir=$exp_root/chain/tree_sp_${tree_affix}
# if [ -f $treedir/final.mdl ]; then
#   echo "$0: $treedir/final.mdl exists. Remove it and run again."
#   exit 1
# fi
#
# if [ $stage -le 6 ]; then
#   # This is usually 3 for chain systems.
#   echo $frame_subsampling_factor > \
#     $sup_chain_dir/best_path_${unsupervised_set_perturbed}/frame_subsampling_factor
#
#   # This should be 1 if using a different source for supervised data alignments.
#   # However alignments in seed tree directory have already been sub-sampled.
#   echo $frame_subsampling_factor > \
#     $sup_tree_dir/frame_subsampling_factor
#
#   # Build a new tree using stats from both supervised and unsupervised data
#   steps/nnet3/chain/build_tree_multiple_sources.sh \
#     --use-fmllr false --context-opts "--context-width=2 --central-position=1" \
#     --frame-subsampling-factor $frame_subsampling_factor \
#     7000 $lang \
#     data/${supervised_set_perturbed} \
#     ${sup_tree_dir} \
#     data/${unsupervised_set_perturbed} \
#     $chaindir/best_path_${unsupervised_set_perturbed} \
#     $treedir || exit 1
# fi
#
# sup_tree_dir=$treedir   # Use the new tree dir for further steps

# Train denominator FST using phone alignments from
# supervised and unsupervised data
if [ $stage -le 7 ]; then
  steps/nnet3/chain/make_weighted_den_fst.sh --num-repeats $lm_weights --cmd "$train_cmd" \
    ${sup_tree_dir} ${sup_chain_dir}/best_path_${unsupervised_set_perturbed} \
    $dir
fi

if [ $stage -le 8 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $sup_tree_dir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)
  opts="l2-regularize=0.004 dropout-proportion=0.0 dropout-per-dim=true dropout-per-dim-continuous=true"
  linear_opts="orthonormal-constraint=-1.0 l2-regularize=0.004"
  output_opts="l2-regularize=0.002"

  mkdir -p $dir/configs

  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-dropout-layer name=tdnn1 $opts dim=1024
  linear-component name=tdnn2l0 dim=256 $linear_opts input=Append(-1,0)
  linear-component name=tdnn2l dim=256 $linear_opts input=Append(-1,0)
  relu-batchnorm-dropout-layer name=tdnn2 $opts input=Append(0,1) dim=1024
  linear-component name=tdnn3l dim=256 $linear_opts input=Append(-1,0)
  relu-batchnorm-dropout-layer name=tdnn3 $opts dim=1024 input=Append(0,1)
  linear-component name=tdnn4l0 dim=256 $linear_opts input=Append(-1,0)
  linear-component name=tdnn4l dim=256 $linear_opts input=Append(0,1)
  relu-batchnorm-dropout-layer name=tdnn4 $opts input=Append(0,1) dim=1024
  linear-component name=tdnn5l dim=256 $linear_opts
  relu-batchnorm-dropout-layer name=tdnn5 $opts dim=1024 input=Append(0, tdnn3l)
  linear-component name=tdnn6l0 dim=256 $linear_opts input=Append(-3,0)
  linear-component name=tdnn6l dim=256 $linear_opts input=Append(-3,0)
  relu-batchnorm-dropout-layer name=tdnn6 $opts input=Append(0,3) dim=1280
  linear-component name=tdnn7l0 dim=256 $linear_opts input=Append(-3,0)
  linear-component name=tdnn7l dim=256 $linear_opts input=Append(0,3)
  relu-batchnorm-dropout-layer name=tdnn7 $opts input=Append(0,3,tdnn6l,tdnn4l,tdnn2l) dim=1024
  linear-component name=tdnn8l0 dim=256 $linear_opts input=Append(-3,0)
  linear-component name=tdnn8l dim=256 $linear_opts input=Append(0,3)
  relu-batchnorm-dropout-layer name=tdnn8 $opts input=Append(0,3) dim=1280
  linear-component name=tdnn9l0 dim=256 $linear_opts input=Append(-3,0)
  linear-component name=tdnn9l dim=256 $linear_opts input=Append(-3,0)
  relu-batchnorm-dropout-layer name=tdnn9 $opts input=Append(0,3,tdnn8l,tdnn6l,tdnn5l) dim=1024
  linear-component name=tdnn10l0 dim=256 $linear_opts input=Append(-3,0)
  linear-component name=tdnn10l dim=256 $linear_opts input=Append(0,3)
  relu-batchnorm-dropout-layer name=tdnn10 $opts input=Append(0,3) dim=1280
  linear-component name=tdnn11l0 dim=256 $linear_opts input=Append(-3,0)
  linear-component name=tdnn11l dim=256 $linear_opts input=Append(-3,0)
  relu-batchnorm-dropout-layer name=tdnn11 $opts input=Append(0,3,tdnn10l,tdnn9l,tdnn7l) dim=1024
  linear-component name=prefinal-l dim=256 $linear_opts

  relu-batchnorm-layer name=prefinal-chain input=prefinal-l $opts dim=1280
  linear-component name=prefinal-chain-l dim=256 $linear_opts
  batchnorm-component name=prefinal-chain-batchnorm
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

  relu-batchnorm-layer name=prefinal-xent input=prefinal-l $opts dim=1280
  linear-component name=prefinal-xent-l dim=256 $linear_opts
  batchnorm-component name=prefinal-xent-batchnorm
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
 
  # We use separate outputs for supervised and unsupervised data
  # so we can properly track the train and valid objectives.

  output name=output-0 input=output.affine
  output name=output-1 input=output.affine

  output name=output-0-xent input=output-xent.log-softmax
  output name=output-1-xent input=output-xent.log-softmax
EOF

  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

# Get values for $model_left_context, $model_right_context
. $dir/configs/vars

left_context=$model_left_context
right_context=$model_right_context

egs_left_context=$(perl -e "print int($left_context + $frame_subsampling_factor / 2)")
egs_right_context=$(perl -e "print int($right_context + $frame_subsampling_factor / 2)")

if [ -z "$sup_egs_dir" ]; then
  sup_egs_dir=$dir/egs_${supervised_set_perturbed}
  frames_per_eg=$(cat $sup_chain_dir/egs/info/frames_per_eg)

  if [ $stage -le 9 ]; then
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $sup_egs_dir/storage ]; then
      utils/create_split_dir.pl \
       /export/b0{5,6,7,8}/$USER/kaldi-data/egs/material-$(date +'%m_%d_%H_%M')/s5/$sup_egs_dir/storage $sup_egs_dir/storage
    fi
    mkdir -p $sup_egs_dir/
    touch $sup_egs_dir/.nodelete # keep egs around when that run dies.

    echo "$0: generating egs from the supervised data"
    steps/nnet3/chain/get_egs.sh --cmd "$decode_cmd" \
               --left-context $egs_left_context --right-context $egs_right_context \
               --left-context-initial $egs_left_context --right-context-final $egs_right_context \
               --frame-subsampling-factor $frame_subsampling_factor \
               --alignment-subsampling-factor $frame_subsampling_factor \
               --frames-per-eg $frames_per_eg \
               --frames-per-iter 1500000 \
               --cmvn-opts "$cmvn_opts" \
               --online-ivector-dir $sup_ivector_dir \
               --generate-egs-scp true \
               data/${supervised_set_perturbed}_hires $dir \
               $sup_lat_dir $sup_egs_dir
  fi
else
  frames_per_eg=$(cat $sup_egs_dir/info/frames_per_eg)
fi

unsup_frames_per_eg=150  # Using a frames-per-eg of 150 for unsupervised data
                         # was found to be better than allowing smaller chunks
                         # (160,140,110,80) like for supervised system
lattice_lm_scale=0.5  # lm-scale for using the weights from unsupervised lattices when
                      # creating numerator supervision
lattice_prune_beam=4.0  # beam for pruning the lattices prior to getting egs
                        # for unsupervised data
tolerance=1   # frame-tolerance for chain training

unsup_lat_dir=${sup_chain_dir}/decode_${unsupervised_set_perturbed}
if [ -z "$unsup_egs_dir" ]; then
  unsup_egs_dir=$dir/egs_${unsupervised_set_perturbed}

  if [ $stage -le 10 ]; then
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $unsup_egs_dir/storage ]; then
      utils/create_split_dir.pl \
       /export/b0{5,6,7,8}/$USER/kaldi-data/egs/material-$(date +'%m_%d_%H_%M')/s5/$unsup_egs_dir/storage $unsup_egs_dir/storage
    fi
    mkdir -p $unsup_egs_dir
    touch $unsup_egs_dir/.nodelete # keep egs around when that run dies.

    cp $sup_chain_dir/final.mdl $unsup_lat_dir || exit 1;

    echo "$0: generating egs from the unsupervised data"
    steps/nnet3/chain/get_egs.sh \
      --cmd "$decode_cmd" --alignment-subsampling-factor 1 \
      --left-tolerance $tolerance --right-tolerance $tolerance \
      --left-context $egs_left_context --right-context $egs_right_context \
      --left-context-initial $egs_left_context --right-context-final $egs_right_context \
      --frames-per-eg $unsup_frames_per_eg --frames-per-iter 1500000 \
      --frame-subsampling-factor $frame_subsampling_factor \
      --cmvn-opts "$cmvn_opts" --lattice-lm-scale $lattice_lm_scale \
      --lattice-prune-beam "$lattice_prune_beam" \
      --deriv-weights-scp $sup_chain_dir/best_path_${unsupervised_set_perturbed}/weights.scp \
      --online-ivector-dir $ivector_root_dir/ivectors_${unsupervised_set_perturbed}_hires \
      --generate-egs-scp true $unsup_egs_opts \
      data/${unsupervised_set_perturbed}_hires $dir \
      $unsup_lat_dir $unsup_egs_dir
  fi
fi

comb_egs_dir=$dir/comb_egs
if [ $stage -le 11 ]; then
  steps/nnet3/chain/multilingual/combine_egs.sh --cmd "$train_cmd" \
    --block-size 128 \
    --lang2weight $supervision_weights 2 \
    $sup_egs_dir $unsup_egs_dir $comb_egs_dir
  touch $comb_egs_dir/.nodelete # keep egs around when that run dies.
fi

if [ $train_stage -le -4 ]; then
  # This is to skip stages of den-fst creation, which was already done.
  train_stage=-4
fi

if [ $stage -le 12 ]; then
  steps/nnet3/chain/train.py --stage $train_stage \
    --egs.dir="$comb_egs_dir" \
    --cmd="$decode_cmd" \
    --feat.online-ivector-dir=$sup_ivector_dir \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient=0.1 \
    --chain.l2-regularize=0.0 \
    --chain.apply-deriv-weights=true \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --egs.chunk-width=$frames_per_eg \
    --egs.chunk-left-context=$chunk_left_context \
    --egs.chunk-right-context=$chunk_right_context \
    --egs.chunk-left-context-initial=0 \
    --egs.chunk-right-context-final=0 \
    --trainer.num-chunk-per-minibatch=128,64 \
    --trainer.frames-per-iter=1500000 \
    --trainer.num-epochs=$num_epochs \
    --trainer.optimization.num-jobs-initial=2 \
    --trainer.optimization.num-jobs-final=12 \
    --trainer.optimization.initial-effective-lrate=0.001 \
    --trainer.optimization.final-effective-lrate=0.0001 \
    --trainer.optimization.momentum=0.0 \
    --trainer.max-param-change=2.0 \
    --cleanup.remove-egs=false \
    --feat-dir=data/${supervised_set_perturbed}_hires \
    --tree-dir=$sup_tree_dir \
    --lat-dir=$sup_lat_dir \
    --dir=$dir || exit 1;
fi

test_graph_dir=$dir/graph_combined
if [ $stage -le 13 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 ${test_lang} $dir $test_graph_dir
fi

if [ $stage -le 14 ]; then
  frames_per_chunk=150
  rm -f $dir/.error 2>/dev/null || true
  for data in $test_sets; do
      (
      nspk=$(wc -l <data/${data}_hires/spk2utt)
      steps/nnet3/decode.sh \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --extra-left-context $chunk_left_context --extra-right-context $chunk_right_context \
          --extra-left-context-initial 0 \
          --extra-right-context-final 0 \
          --frames-per-chunk $frames_per_chunk \
          --nj $nspk --cmd "$decode_cmd"  --num-threads 4 \
          --online-ivector-dir ${ivector_root_dir}/ivectors_${data}_hires \
          $test_graph_dir data/${data}_hires ${dir}/decode_${data} || touch $dir/.error
      ) &
  done
  wait;
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

exit 0;
