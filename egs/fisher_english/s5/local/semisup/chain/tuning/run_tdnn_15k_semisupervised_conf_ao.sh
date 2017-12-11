#!/bin/bash

# This script is semi-supervised recipe with 15 hours of supervised data
# and 250 hours unsupervised data with naive splitting. 
# We use the combined data for i-vector extractor training.
# We use 4-gram LM trained on 1250 hours of data excluding the 250 hours
# unsupervised data to create LM for decoding. Rescoring is done with 
# a larger 4-gram LM.
# This script uses the same tree as that for the seed model.
# This script is similar to _an, but uses phone LM to model UNK.

# Unsupervised set: train_unsup100k_250k
# unsup_frames_per_eg=150
# Deriv weights: Lattice posterior of best path pdf
# Unsupervised weight: 1.0
# Weights for phone LM (supervised, unsupervised): 5,2
# LM for decoding unsupervised data: 4gram
# Supervision: Naive split lattices

set -u -e -o pipefail

stage=0   # Start from -1 for supervised seed system training
train_stage=-100
nj=40
decode_nj=40
exp=exp/semisup_15k

# Datasets -- Expects data/$supervised_set and data/$unsupervised_set to be 
# present
unsupervised_set=train_unsup100k_250k  # set this to your choice of unsupervised data
supervised_set=train_sup15k
semisup_train_set=semisup15k_250k

# Seed model options
nnet3_affix=_semi15k_250k    # affix for nnet3 and chain dir -- relates to i-vector used
tdnn_affix=7i  # affix for the supervised chain-model directory
tree_affix=bi_i  # affix for the tree of the supervised model
train_supervised_opts="--stage -10 --train-stage -10"

# Unsupervised options
decode_affix=   # affix for decoded lattices
egs_affix=  # affix for the egs that are generated from unsupervised data and for the comined egs dir
unsup_frames_per_eg=150  # if empty, will be equal to the supervised model's config
lattice_lm_scale=0.5  # lm-scale for using the weights from unsupervised lattices when creating numerator supervision
lattice_prune_beam=4.0  # If supplied, will prune the lattices prior to getting egs for unsupervised data
tolerance=1   # frame-tolerance for chain training
phone_insertion_penalty=

rescore_unsup_lattices=true  # Const ARPA rescoring with a bigger LM
unsup_rescoring_affix=big   # Affix for const ARPA lang dir

# Semi-supervised options
comb_affix=comb1ao  # affix for new chain-model directory trained on the combined supervised+unsupervised subsets
supervision_weights=1.0,1.0   # Weights for supervised, unsupervised data egs
lm_weights=5,2   # Weights on phone counts from supervised, unsupervised data for denominator FST creation

sup_egs_dir=   # Supply this to skip supervised egs creation
unsup_egs_dir=  # Supply this to skip unsupervised egs creation
unsup_egs_opts=  # Extra options to pass to unsupervised egs creation

# Neural network opts
apply_deriv_weights=true
xent_regularize=0.1
hidden_dim=725

decode_iter=  # Iteration to decode with 

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

egs_affix=${egs_affix}_prun${lattice_prune_beam}_lmwt${lattice_lm_scale}_tol${tolerance}

RANDOM=0

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

if [ $stage -le -1 ]; then
  echo "$0: chain training on the supervised subset data/${supervised_set}"
  local/semisup/chain/tuning/run_tdnn_15k_i.sh $train_supervised_opts \
                          --train-set $supervised_set \
                          --unsup-train-set $unsupervised_set \
                          --semisup-train-set $semisup_train_set \
                          --nnet3-affix "$nnet3_affix" --tdnn-affix "$tdnn_affix" \
                          --tree-affix "$tree_affix" --exp $exp || exit 1
fi

lang=data/lang_chain_unk
unsup_decode_lang=data/lang_poco_test_ex250k_unk
unsup_rescore_lang=${unsup_decode_lang}${unsup_rescoring_affix}
unsup_decode_graph_affix=_poco_ex250k_unk

test_lang=data/lang_poco_test_unk
test_graph_affix=_poco_unk

extractor=$exp/nnet3${nnet3_affix}/extractor  # i-vector extractor
chaindir=$exp/chain${nnet3_affix}/tdnn${tdnn_affix}_sp  # supervised seed model
graphdir=$chaindir/graph${unsup_decode_graph_affix}

decode_affix=${decode_affix}${unsup_decode_graph_affix}

if [ ! -f $graphdir/HCLG.fst ]; then
  utils/mkgraph.sh --self-loop-scale 1.0 $unsup_decode_lang $chaindir $graphdir
fi

if [ $stage -le 2 ]; then
  # Keep 10 hours out for development set
  utils/subset_data_dir.sh --speakers data/${unsupervised_set} 10000 data/${unsupervised_set}_10k
  utils/subset_data_dir.sh --speakers data/${unsupervised_set}_10k 5000 data/${unsupervised_set}_10k_calib_train
  utils/subset_data_dir.sh --utt-list <(utils/filter_scp.pl --exclude data/${unsupervised_set}_10k_calib_train/utt2spk data/${unsupervised_set}_10k/utt2spk) \
    data/${unsupervised_set}_10k data/${unsupervised_set}_10k_calib_dev
  utils/subset_data_dir.sh --utt-list <(utils/filter_scp.pl --exclude data/${unsupervised_set}_10k/utt2spk data/${unsupervised_set}/utt2spk) \
    data/${unsupervised_set} data/${unsupervised_set}_240k
fi

unsupervised_set=${unsupervised_set}_240k

for dset in $unsupervised_set; do
  if [ ! -f data/${dset}_sp_hires/feats.scp ]; then
    echo "$0: Could not find data/${dset}_sp_hires/feats.scp."
    echo "$0: Expected this to be created in stage -1."
    exit 1
  fi

  # Decode unsupervised data and write lattices in non-compact 
  # undeterminized format
  # Set --skip-scoring to false in order to score the unsupervised data
  if [ $stage -le 4 ]; then
    echo "$0: getting the decoding lattices for the unsupervised subset using the chain model at: $chaindir"
    steps/nnet3/decode.sh --num-threads 4 --nj $decode_nj --cmd "$decode_cmd" \
              --acwt 1.0 --post-decode-acwt 10.0 --write-compact false --skip-scoring true \
              --online-ivector-dir $exp/nnet3${nnet3_affix}/ivectors_${semisup_train_set}_sp_hires \
              --scoring-opts "--min-lmwt 10 --max-lmwt 10" --word-determinize false \
              $graphdir data/${dset}_sp_hires $chaindir/decode_${dset}_sp${decode_affix}
  fi

  # Rescore undeterminized lattices with larger LM
  if $rescore_unsup_lattices; then
    if [ $stage -le 5 ]; then
      steps/lmrescore_const_arpa_undeterminized.sh --cmd "$decode_cmd" \
        --acwt 0.1 --beam 8.0  --skip-scoring true \
        $unsup_decode_lang $unsup_rescore_lang \
        data/${dset}_sp_hires \
        $chaindir/decode_${dset}_sp${decode_affix} \
        $chaindir/decode_${dset}_sp${decode_affix}${unsup_rescoring_affix}
    fi

    ln -sf ../final.mdl $chaindir/decode_${dset}_sp${decode_affix}_${unsup_rescoring_affix}/final.mdl
  else
    ln -sf ../final.mdl $chaindir/decode_${dset}_sp${decode_affix}/final.mdl
  fi
done

if $rescore_unsup_lattices; then
  decode_affix=${decode_affix}_${unsup_rescoring_affix}
fi

# Get lattice posterior of best path alignment
if [ $stage -le 8 ]; then
  steps/best_path_weights.sh --cmd "${train_cmd}" --acwt 0.1 \
    data/${unsupervised_set}_sp_hires $lang \
    $chaindir/decode_${unsupervised_set}_sp${decode_affix} \
    $chaindir/best_path_${unsupervised_set}_sp${decode_affix}
fi

frame_subsampling_factor=1
if [ -f $chaindir/frame_subsampling_factor ]; then
  frame_subsampling_factor=`cat $chaindir/frame_subsampling_factor`
fi
cmvn_opts=`cat $chaindir/cmvn_opts` || exit 1

treedir=$exp/chain${nnet3_affix}/tree_${tree_affix}
if [ ! -f $treedir/final.mdl ]; then
  echo "$0: $treedir/final.mdl does not exist."
  exit 1
fi

diff $treedir/tree $chaindir/tree || { echo "$0: $treedir/tree and $chaindir/tree differ"; exit 1; }

dir=$exp/chain${nnet3_affix}/tdnn${tdnn_affix}${decode_affix}${egs_affix}${comb_affix:+_$comb_affix}


# Train denominator FST using phone alignments from 
# supervised and unsupervised data
if [ $stage -le 10 ]; then
  echo $frame_subsampling_factor > $chaindir/best_path_${unsupervised_set}_sp${decode_affix}/frame_subsampling_factor
  steps/nnet3/chain/make_weighted_den_fst.sh --num-repeats $lm_weights --cmd "$train_cmd" \
    ${treedir} ${chaindir}/best_path_${unsupervised_set}_sp${decode_affix} \
    $dir
fi

if [ $stage -le 11 ]; then
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
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5
  
  # We use separate outputs for supervised and unsupervised data
  # so we can properly track the train and valid objectives.

  output name=output-0 input=output.affine
  output name=output-1 input=output.affine

  output name=output-0-xent input=output-xent.log-softmax
  output name=output-1-xent input=output-xent.log-softmax
EOF

  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

. $dir/configs/vars

left_context=$model_left_context
right_context=$model_right_context
left_context_initial=$model_left_context
right_context_final=$model_right_context

left_context=`perl -e "print int($left_context + $frame_subsampling_factor / 2)"`
right_context=`perl -e "print int($right_context + $frame_subsampling_factor / 2)"`
left_context_initial=`perl -e "print int($left_context_initial + $frame_subsampling_factor / 2)"`
right_context_final=`perl -e "print int($right_context_final + $frame_subsampling_factor / 2)"`

supervised_set=${supervised_set}_sp
sup_lat_dir=$exp/chain${nnet3_affix}/tri3_${supervised_set}_unk_lats
if [ -z "$sup_egs_dir" ]; then
  sup_egs_dir=$dir/egs_${supervised_set}
  frames_per_eg=$(cat $chaindir/egs/info/frames_per_eg)

  if [ $stage -le 12 ]; then
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $sup_egs_dir/storage ]; then
      utils/create_split_dir.pl \
       /export/b0{5,6,7,8}/$USER/kaldi-data/egs/fisher_english-$(date +'%m_%d_%H_%M')/s5c/$sup_egs_dir/storage $sup_egs_dir/storage
    fi
    mkdir -p $sup_egs_dir/
    touch $sup_egs_dir/.nodelete # keep egs around when that run dies.

    echo "$0: generating egs from the supervised data"
    steps/nnet3/chain/get_egs.sh --cmd "$decode_cmd" \
               --left-context $left_context --right-context $right_context \
               --left-context-initial $left_context_initial --right-context-final $right_context_final \
               --frame-subsampling-factor $frame_subsampling_factor \
               --alignment-subsampling-factor 3 \
               --frames-per-eg $frames_per_eg \
               --frames-per-iter 1500000 \
               --cmvn-opts "$cmvn_opts" \
               --online-ivector-dir $exp/nnet3${nnet3_affix}/ivectors_${semisup_train_set}_sp_hires \
               --generate-egs-scp true \
               data/${supervised_set}_hires $dir \
               $sup_lat_dir $sup_egs_dir
  fi
else
  frames_per_eg=$(cat $sup_egs_dir/info/frames_per_eg)
fi

unsupervised_set=${unsupervised_set}_sp
unsup_lat_dir=${chaindir}/decode_${unsupervised_set}${decode_affix} 

if [ -z "$unsup_egs_dir" ]; then
  [ -z $unsup_frames_per_eg ] && [ ! -z "$frames_per_eg" ] && unsup_frames_per_eg=$frames_per_eg
  unsup_egs_dir=$dir/egs_${unsupervised_set}${decode_affix}${egs_affix}

  if [ $stage -le 13 ]; then
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $unsup_egs_dir/storage ]; then
      utils/create_split_dir.pl \
       /export/b0{5,6,7,8}/$USER/kaldi-data/egs/fisher_english-$(date +'%m_%d_%H_%M')/s5c/$unsup_egs_dir/storage $unsup_egs_dir/storage
    fi
    mkdir -p $unsup_egs_dir
    touch $unsup_egs_dir/.nodelete # keep egs around when that run dies.

    echo "$0: generating egs from the unsupervised data"
    steps/nnet3/chain/get_egs.sh \
               --cmd "$decode_cmd" --alignment-subsampling-factor 1 \
               --left-tolerance $tolerance --right-tolerance $tolerance \
               --left-context $left_context --right-context $right_context \
               --left-context-initial $left_context_initial --right-context-final $right_context_final \
               --frames-per-eg $unsup_frames_per_eg --frames-per-iter 1500000 \
               --frame-subsampling-factor $frame_subsampling_factor \
               --cmvn-opts "$cmvn_opts" --lattice-lm-scale $lattice_lm_scale \
               --lattice-prune-beam "$lattice_prune_beam" \
               --phone-insertion-penalty "$phone_insertion_penalty" \
               --deriv-weights-scp $chaindir/best_path_${unsupervised_set}${decode_affix}/weights.scp \
               --online-ivector-dir $exp/nnet3${nnet3_affix}/ivectors_${semisup_train_set}_sp_hires \
               --generate-egs-scp true $unsup_egs_opts \
               data/${unsupervised_set}_hires $dir \
               $unsup_lat_dir $unsup_egs_dir
  fi
fi

comb_egs_dir=$dir/${comb_affix}_egs${decode_affix}${egs_affix}_multi

if [ $stage -le 14 ]; then
  steps/nnet3/multilingual/combine_egs.sh --cmd "$train_cmd" \
    --minibatch-size 128 --frames-per-iter 1500000 \
    --lang2weight $supervision_weights --egs-prefix cegs. 2 \
    $sup_egs_dir $unsup_egs_dir $comb_egs_dir
  touch $comb_egs_dir/.nodelete # keep egs around when that run dies.
fi

if [ $train_stage -le -4 ]; then
  train_stage=-4
fi

if [ $stage -le 15 ]; then
  steps/nnet3/chain/train.py --stage $train_stage \
    --egs.dir "$comb_egs_dir" \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir $exp/nnet3${nnet3_affix}/ivectors_${semisup_train_set}_sp_hires \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights $apply_deriv_weights \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width $frames_per_eg \
    --trainer.num-chunk-per-minibatch 128 \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs 4 \
    --trainer.optimization.num-jobs-initial 3 \
    --trainer.optimization.num-jobs-final 16 \
    --trainer.optimization.initial-effective-lrate 0.001 \
    --trainer.optimization.final-effective-lrate 0.0001 \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs false \
    --feat-dir data/${supervised_set}_hires \
    --tree-dir $treedir \
    --lat-dir $sup_lat_dir \
    --dir $dir || exit 1;
fi

test_graph_dir=$dir/graph${test_graph_affix}
if [ $stage -le 17 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 ${test_lang} $dir $test_graph_dir
fi

if [ $stage -le 18 ]; then
  iter_opts=
  if [ ! -z $decode_iter ]; then
    nnet3-copy --edits="remove-output-nodes name=output;rename-node old-name=output-0 new-name=output" $dir/${decode_iter}.mdl - | \
      nnet3-am-copy --set-raw-nnet=- $dir/${decode_iter}.mdl $dir/${decode_iter}-output.mdl || exit 1
    iter_opts=" --iter ${decode_iter}-output "
  else
    nnet3-copy --edits="remove-output-nodes name=output;rename-node old-name=output-0 new-name=output" $dir/final.mdl - | \
      nnet3-am-copy --set-raw-nnet=- $dir/final.mdl $dir/final-output.mdl || exit 1
    iter_opts=" --iter final-output "
  fi

  for decode_set in dev test; do
      (
      num_jobs=`cat data/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $num_jobs --cmd "$decode_cmd" $iter_opts \
          --online-ivector-dir $exp/nnet3${nnet3_affix}/ivectors_${decode_set}_hires \
          $test_graph_dir data/${decode_set}_hires \
          $dir/decode${test_graph_affix}_${decode_set}${decode_iter:+_iter$decode_iter} || exit 1;
      ) &
  done
fi

wait;
exit 0;
