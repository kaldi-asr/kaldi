#!/bin/bash

set -u -e -o pipefail

stage=-2
train_stage=-100
nj=30
decode_nj=30
base_train_set=train_comb350k # for reference

unsupervised_set=train_unsup250k  # set this to your choice of unsupervised data
supervised_set=train_sup
semi_affix=350k  # affix relating train-set splitting proportion

tdnn_affix=_xxsup1a  # affix for the supervised chain-model directory
train_supervised_opts="--stage -10 --train-stage -10"

# Unsupervised options
decode_affix=_ex250k
egs_affix=_prun2_lmwt0_tol2  # affix for the egs that are generated from unsupervised data and for the comined egs dir
unsup_frames_per_eg=  # if empty will be equal to the supervised model's config -- you will need to change minibatch_size for comb training accordingly
lattice_lm_scale=0.0  # lm-scale for using the weights from unsupervised lattices
lattice_prune_beam=2.0  # If supplied will prune the lattices prior to getting egs for unsupervised data
left_tolerance=2
right_tolerance=2
graph_affix=   # can be used to decode the unsup data with another lm/graph
phone_insertion_penalty=

common_egs_dir=

# Semi-supervised options
comb_affix=_comb1b2  # affix for new chain-model directory trained on the combined supervised+unsupervised subsets
tree_affix=
xent_regularize=0.1
hidden_dim=725
minibatch_size=128

decode_iter=
# to tune:
# frames_per_eg for unsupervised

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

nnet3_affix=_semi${semi_affix}  # affix for nnet3 and chain dirs
decode_affix=${decode_affix}${graph_affix}

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

if [ $stage -le -2 ]; then
  echo "$0: chain training on the supervised subset data/${supervised_set}"
  local/chain/run_tdnn.sh $train_supervised_opts --remove-egs false \
                          --train-set $supervised_set \
                          --nnet3-affix $nnet3_affix --tdnn-affix $tdnn_affix
fi

if [ $stage -le -1 ] && [ ! -f exp/nnet3${nnet3_affix}/ivectors_${unsupervised_set}_hires/ivector_online.scp ]; then
  echo "$0: getting ivectors for the hires unsupervised data data/${unsupervised_set}_hires"
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj "$nj" \
            data/${unsupervised_set}_hires exp/nnet3${nnet3_affix}/extractor \
            exp/nnet3${nnet3_affix}/ivectors_${unsupervised_set}_hires
fi

chaindir=exp/chain${nnet3_affix}/tdnn${tdnn_affix}_sp

left_context=`cat $chaindir/egs/info/left_context`
right_context=`cat $chaindir/egs/info/right_context`
left_context_initial=`cat $chaindir/egs/info/left_context_initial`
right_context_final=`cat $chaindir/egs/info/right_context_final`
[ -z $unsup_frames_per_eg ] && unsup_frames_per_eg=`cat $chaindir/egs/info/frames_per_eg`
frame_subsampling_factor=`cat $chaindir/frame_subsampling_factor`
cmvn_opts=`cat $chaindir/cmvn_opts`

if [ -z "$common_egs_dir" ]; then
  if [ $stage -le 0 ]; then
    echo "$0: getting the decoding lattices for the unsupervised subset using the chain model at: $chaindir"
    graphdir=$chaindir/graph${graph_affix}
    if [ ! -f $graphdir/HCLG.fst ]; then
      utils/mkgraph.sh --self-loop-scale 1.0 data/lang_test${graph_affix} $chaindir $graphdir
    fi
    steps/nnet3/decode.sh --num-threads 4 --nj $decode_nj --cmd "$decode_cmd" \
              --acwt 1.0 --post-decode-acwt 10.0 \
              --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${unsupervised_set}_hires \
              --scoring-opts "--min-lmwt 5 " \
              $chaindir/graph${graph_affix} data/${unsupervised_set}_hires $chaindir/decode_${unsupervised_set}${decode_affix}
    ln -s ../final.mdl $chaindir/decode_${unsupervised_set}${decode_affix}/final.mdl || true
  fi

  unsup_egs_dir=$chaindir/unsup_egs${decode_affix}${egs_affix}

  if [ $stage -le 1 ]; then
    echo "$0: generating egs from the unsupervised data"
    steps/nnet3/chain/get_egs.sh --cmd "$decode_cmd" --alignment-subsampling-factor 1 \
               --left-tolerance $left_tolerance --right-tolerance $right_tolerance \
               --left-context $left_context --right-context $right_context \
               --left-context-initial $left_context_initial --right-context-final $right_context_final \
               --frames-per-eg $unsup_frames_per_eg --frames-per-iter 1500000 \
               --frame-subsampling-factor $frame_subsampling_factor \
               --cmvn-opts "$cmvn_opts" --lattice-lm-scale $lattice_lm_scale \
               --lattice-prune-beam "$lattice_prune_beam" \
               --phone-insertion-penalty "$phone_insertion_penalty" \
               --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${unsupervised_set}_hires \
               data/${unsupervised_set}_hires $chaindir \
               ${chaindir}/decode_${unsupervised_set}${decode_affix} $unsup_egs_dir
  fi

  sup_egs_dir=$chaindir/egs_scp
  unsup_egs_dir=$chaindir/unsup_egs${decode_affix}${egs_affix}_scp
  comb_egs_dir=$chaindir/comb_egs${decode_affix}${egs_affix}_multi
  if [ $stage -le 2 ]; then
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $comb_egs_dir/storage ]; then
      utils/create_split_dir.pl \
       /export/b0{5,6,7,8}/$USER/kaldi-data/egs/fisher_english-$(date +'%m_%d_%H_%M')/s5c/$comb_egs_dir $comb_egs_dir
    fi

    touch $comb_egs_dir/.nodelete # keep egs around when that run dies.

    steps/nnet3/multilingual/combine_egs.sh --cmd "$train_cmd" \
      --minibatch-size 128 --samples-per-iter 10000 \
      --lang2weight 1.0,0.5 --egs-prefix cegs. 2 \
      $sup_egs_dir $unsup_egs_dir $comb_egs_dir
  fi
  common_egs_dir=$comb_egs_dir
fi 

treedir=exp/chain${nnet3_affix}/tree_${tree_affix}
lat_dir=exp/chain${nnet3_affix}/tri5a_${supervised_set}_sp_lats  # not required since egs is given.
dir=exp/chain${nnet3_affix}/tdnn${tdnn_affix}${decode_affix}${egs_affix}${comb_affix}

if [ $stage -le 12 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 dim=$hidden_dim
  relu-batchnorm-layer name=tdnn2 input=Append(-1,0,1,2) dim=$hidden_dim
  relu-batchnorm-layer name=tdnn3 input=Append(-3,0,3) dim=$hidden_dim
  relu-batchnorm-layer name=tdnn4 input=Append(-3,0,3) dim=$hidden_dim
  relu-batchnorm-layer name=tdnn5 input=Append(-3,0,3) dim=$hidden_dim
  relu-batchnorm-layer name=tdnn6 input=Append(-6,-3,0) dim=$hidden_dim

  ## adding the layers for chain branch
  relu-batchnorm-layer name=prefinal-chain input=tdnn6 dim=$hidden_dim target-rms=0.5
  output-layer name=output-0 input=prefinal-chain include-log-softmax=false dim=$num_targets max-change=1.5
  output-layer name=output-1 input=prefinal-chain include-log-softmax=false dim=$num_targets max-change=1.5
  output-layer name=output input=prefinal-chain include-log-softmax=false dim=$num_targets max-change=1.5

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  relu-batchnorm-layer name=prefinal-xent input=tdnn6 dim=$hidden_dim target-rms=0.5
  output-layer name=output-0-xent dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5
  output-layer name=output-1-xent input=prefinal-xent dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5
EOF

  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
  cp $dir/configs/final.config{,.orig}

  cat $dir/configs/final.config.orig | \
    perl -pe 's/component=output-1.affine/component=output-0.affine/g; 
              s/component=output-1-xent.affine/component=output-0-xent.affine/g;' > \
    $dir/configs/final.config
fi

if [ $stage -le 13 ]; then
  steps/nnet3/chain/train.py --stage $train_stage \
    --egs.dir "$common_egs_dir" \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${supervised_set}_sp_hires \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize 0.1 \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights true \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width 150 \
    --trainer.num-chunk-per-minibatch $minibatch_size \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs 4 \
    --trainer.optimization.num-jobs-initial 3 \
    --trainer.optimization.num-jobs-final 16 \
    --trainer.optimization.initial-effective-lrate 0.001 \
    --trainer.optimization.final-effective-lrate 0.0001 \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs false \
    --feat-dir data/${supervised_set}_sp_hires \
    --tree-dir $treedir \
    --lat-dir $lat_dir \
    --dir $dir  || exit 1;
fi

graph_dir=$dir/graph
if [ $stage -le 14 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_test $dir $graph_dir
fi

decode_suff=
if [ $stage -le 15 ]; then
  iter_opts=
  if [ ! -z "$decode_iter" ]; then
    iter_opts=" --iter $decode_iter "
  else 
    decode_iter=final
  fi
  
  nnet3-copy --edits="remove-output-nodes name=output;rename-node old-name=output-0 new-name=output;" $dir/$decode_iter.mdl - | \
    nnet3-am-copy --set-raw-nnet=- $dir/$decode_iter.mdl $dir/${decode_iter}-output.mdl

  iter_opts=" --iter ${decode_iter}-output "
  for decode_set in dev test; do
      (
      num_jobs=`cat data/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $num_jobs --cmd "$decode_cmd" $iter_opts \
          --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${decode_set}_hires \
          $graph_dir data/${decode_set}_hires $dir/decode_${decode_set}${decode_iter:+_$decode_iter}${decode_suff} || exit 1;
      ) &
  done
fi
wait;
exit 0;
