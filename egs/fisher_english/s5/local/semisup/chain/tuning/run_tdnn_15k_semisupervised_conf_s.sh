#!/bin/bash

# This script is same as _d, but uses a weight of 1.0 for unsupervised set.
# unsup_frames_per_eg=150
# Deriv weights: Lattice posterior of best path pdf
# Unsupervised weight: 1.0
# Weights for phone LM (supervised, unsupervises): 5,2
# LM for decoding unsupervised data: 4gram

set -u -e -o pipefail

stage=-2
train_stage=-100
nj=40
decode_nj=40
base_train_set=semisup15k_250k  # for reference
exp=exp/semisup_15k

unsupervised_set=train_unsup250k  # set this to your choice of unsupervised data
supervised_set=train_sup15k
semi_affix=semi15k_250k  # affix relating train-set splitting proportion

tdnn_affix=7b  # affix for the supervised chain-model directory
train_supervised_opts="--stage -10 --train-stage -10"

# Unsupervised options
decode_affix=_unphdet
egs_affix=  # affix for the egs that are generated from unsupervised data and for the comined egs dir
unsup_frames_per_eg=  # if empty will be equal to the supervised model's config -- you will need to change minibatch_size for comb training accordingly
lattice_lm_scale=0.5  # lm-scale for using the weights from unsupervised lattices
lattice_prune_beam=4.0  # If supplied will prune the lattices prior to getting egs for unsupervised data
tolerance=1
graph_affix=_ex250k   # can be used to decode the unsup data with another lm/graph
phone_insertion_penalty=

# Semi-supervised options
comb_affix=comb1s  # affix for new chain-model directory trained on the combined supervised+unsupervised subsets
supervision_weights=1.0,1.0
lm_weights=5,2
sup_egs_dir=   
unsup_egs_dir=
tree_affix=
unsup_egs_opts=
apply_deriv_weights=true


extra_left_context=0
extra_right_context=0

xent_regularize=0.1
hidden_dim=725
minibatch_size="150=128/300=64"
# to tune:
# frames_per_eg for unsupervised

decode_iter=
lang_test_suffix=

finetune_stage=-2
finetune_suffix=_finetune
finetune_iter=final
num_epochs_finetune=1
finetune_xent_regularize=0.1
finetune_opts="--chain.mmi-factor-schedule=0.05,0.05 --chain.smbr-factor-schedule=0.05,0.05"

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

nnet3_affix=_${semi_affix}  # affix for nnet3 and chain dirs
decode_affix=${decode_affix}${graph_affix}
egs_affix=${egs_affix}_prun${lattice_prune_beam}_lmwt${lattice_lm_scale}_tol${tolerance}
tree_affix=

RANDOM=0

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

if false && [ $stage -le 1 ]; then
  echo "$0: chain training on the supervised subset data/${supervised_set}"
  local/chain/run_tdnn_15k.sh $train_supervised_opts --remove-egs false \
                          --train-set $supervised_set --ivector-train-set $base_train_set \
                          --nnet3-affix $nnet3_affix --tdnn-affix $tdnn_affix --exp $exp
fi

extractor=$exp/nnet3${nnet3_affix}/extractor
chaindir=$exp/chain${nnet3_affix}/tdnn${tdnn_affix}_sp
graphdir=$chaindir/graph${graph_affix}
if [ ! -f $graphdir/HCLG.fst ]; then
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_test${graph_affix} $chaindir $graphdir
fi

if [ $stage -le 2 ]; then
  utils/subset_data_dir.sh --speakers data/${unsupervised_set} 10000 data/${unsupervised_set}_10k
  utils/subset_data_dir.sh --speakers data/${unsupervised_set}_10k 5000 data/${unsupervised_set}_10k_calib_train
  utils/subset_data_dir.sh --utt-list <(utils/filter_scp.pl --exclude data/${unsupervised_set}_10k_calib_train/utt2spk data/${unsupervised_set}_10k/utt2spk) \
    data/${unsupervised_set}_10k data/${unsupervised_set}_10k_calib_dev
  utils/subset_data_dir.sh --utt-list <(utils/filter_scp.pl --exclude data/${unsupervised_set}_10k/utt2spk data/${unsupervised_set}/utt2spk) \
    data/${unsupervised_set} data/${unsupervised_set}_240k
fi

unsupervised_set=${unsupervised_set}_240k

for dset in $unsupervised_set; do
  if [ $stage -le 3 ] && [ ! -f data/${dset}_sp_hires/feats.scp ]; then
    utils/data/perturb_data_dir_speed_3way.sh data/$dset data/${dset}_sp_hires_tmp
    utils/subset_data_dir.sh --utt-list data/${dset}_sp_hires_tmp/utt2spk \
      data/${base_train_set}_sp_hires data/${dset}_sp_hires
  fi

  if [ $stage -le 4 ]; then
    echo "$0: getting the decoding lattices for the unsupervised subset using the chain model at: $chaindir"
    steps/nnet3/decode.sh --num-threads 4 --nj $decode_nj --cmd "$decode_cmd" \
              --acwt 1.0 --post-decode-acwt 10.0 --write-compact false \
              --determinize-opts "--word-determinize=false" \
              --online-ivector-dir $exp/nnet3${nnet3_affix}/ivectors_${base_train_set}_sp_hires \
              --scoring-opts "--min-lmwt 10 --max-lmwt 10" --skip-scoring true \
              $graphdir data/${dset}_sp_hires $chaindir/decode_${dset}_sp${decode_affix}
  fi

  if [ $stage -le 5 ]; then
    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" --write-compact false \
      --read-determinized false --write-determinized false --skip-scoring true \
      data/lang_test${graph_affix} \
      data/lang_test${graph_affix}_fg data/${dset}_sp_hires \
      $chaindir/decode_${dset}_sp${decode_affix} \
      $chaindir/decode_${dset}_sp${decode_affix}_fg

    ln -s ../final.mdl $chaindir/decode_${dset}_sp${decode_affix}_fg/final.mdl || true
  fi
done

decode_affix=${decode_affix}_fg

if [ $stage -le 8 ]; then
  steps/best_path_weights.sh --cmd "${train_cmd}" --acwt 0.1 \
    data/${unsupervised_set}_sp_hires data/lang_chain \
    $chaindir/decode_${unsupervised_set}_sp${decode_affix} \
    $chaindir/best_path_${unsupervised_set}_sp${decode_affix}
fi

frame_subsampling_factor=1
if [ -f $chaindir/frame_subsampling_factor ]; then
  frame_subsampling_factor=`cat $chaindir/frame_subsampling_factor`
fi
cmvn_opts=`cat $chaindir/cmvn_opts` || exit 1

sup_ali_dir=$exp/tri3

treedir=$exp/chain${nnet3_affix}/tree_${tree_affix}
if [ $stage -le 9 ]; then
  if [ -f $treedir/final.mdl ]; then
    echo "$0: $treedir/final.mdl already exists. Remove it and try again."
    exit 1
  fi
fi

dir=$exp/chain${nnet3_affix}/tdnn${tdnn_affix}${decode_affix}${egs_affix}${comb_affix:+_$comb_affix}

if [ $stage -le 10 ]; then
  steps/subset_ali_dir.sh --cmd "$train_cmd" \
    data/${unsupervised_set} data/${unsupervised_set}_sp_hires \
    $chaindir/best_path_${unsupervised_set}_sp${decode_affix} \
    $chaindir/best_path_${unsupervised_set}${decode_affix}
  echo $frame_subsampling_factor > $chaindir/best_path_${unsupervised_set}${decode_affix}/frame_subsampling_factor

  steps/nnet3/chain/make_weighted_den_fst.sh --num-repeats $lm_weights --cmd "$train_cmd" \
    ${treedir} ${chaindir}/best_path_${unsupervised_set}${decode_affix} \
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
  
  output name=output-0 input=output.affine skip-in-init=true
  output name=output-1 input=output.affine skip-in-init=true

  output name=output-0-xent input=output-xent.log-softmax skip-in-init=true
  output name=output-1-xent input=output-xent.log-softmax skip-in-init=true
EOF

  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

. $dir/configs/vars

left_context=$[model_left_context + extra_left_context]
right_context=$[model_right_context + extra_right_context]
left_context_initial=$model_left_context
right_context_final=$model_right_context

left_context=`perl -e "print int($left_context + $frame_subsampling_factor / 2)"`
right_context=`perl -e "print int($right_context + $frame_subsampling_factor / 2)"`
left_context_initial=`perl -e "print int($left_context_initial + $frame_subsampling_factor / 2)"`
right_context_final=`perl -e "print int($right_context_final + $frame_subsampling_factor / 2)"`

supervised_set=${supervised_set}_sp
sup_lat_dir=$exp/chain${nnet3_affix}/tri3_${supervised_set}_lats
if [ -z "$sup_egs_dir" ]; then
  sup_egs_dir=$dir/egs_${supervised_set}
  frames_per_eg=$(cat $chaindir/egs/info/frames_per_eg)

  if [ $stage -le 12 ]; then
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $sup_egs_dir/storage ]; then
      utils/create_split_dir.pl \
       /export/b0{5,6,7,8}/$USER/kaldi-data/egs/fisher_english-$(date +'%m_%d_%H_%M')/s5c/$sup_egs_dir/storage $sup_egs_dir/storage
    fi
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
               --online-ivector-dir $exp/nnet3${nnet3_affix}/ivectors_${base_train_set}_sp_hires \
               --generate-egs-scp true \
               data/${supervised_set}_hires $dir \
               $sup_lat_dir $sup_egs_dir
  fi
else
  frames_per_eg=$(cat $sup_egs_dir/info/frames_per_eg)
fi

unsupervised_set=${unsupervised_set}_sp
unsup_lat_dir=${chaindir}/decode_${unsupervised_set}${decode_affix} 
[ -z $unsup_frames_per_eg ] && unsup_frames_per_eg=$frames_per_eg

if [ -z "$unsup_egs_dir" ]; then
  unsup_egs_dir=$dir/egs_${unsupervised_set}${decode_affix}${egs_affix}

  if [ $stage -le 13 ]; then
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $unsup_egs_dir/storage ]; then
      utils/create_split_dir.pl \
       /export/b0{5,6,7,8}/$USER/kaldi-data/egs/fisher_english-$(date +'%m_%d_%H_%M')/s5c/$unsup_egs_dir/storage $unsup_egs_dir/storage
    fi
    touch $unsup_egs_dir/.nodelete # keep egs around when that run dies.

    echo "$0: generating egs from the unsupervised data"
    steps/nnet3/chain/get_egs_split.sh --cmd "$decode_cmd" --alignment-subsampling-factor 1 \
               --left-tolerance $tolerance --right-tolerance $tolerance \
               --left-context $left_context --right-context $right_context \
               --left-context-initial $left_context_initial --right-context-final $right_context_final \
               --frames-per-eg $unsup_frames_per_eg --frames-per-iter 1500000 \
               --frame-subsampling-factor $frame_subsampling_factor \
               --cmvn-opts "$cmvn_opts" --lattice-lm-scale $lattice_lm_scale \
               --lattice-prune-beam "$lattice_prune_beam" \
               --phone-insertion-penalty "$phone_insertion_penalty" \
               --deriv-weights-scp $chaindir/best_path_${unsupervised_set}${decode_affix}/weights.scp \
               --online-ivector-dir $exp/nnet3${nnet3_affix}/ivectors_${base_train_set}_sp_hires \
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
    --feat.online-ivector-dir $exp/nnet3${nnet3_affix}/ivectors_${base_train_set}_sp_hires \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights $apply_deriv_weights \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width 150 \
    --trainer.num-chunk-per-minibatch "$minibatch_size" \
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
    --dir $dir  || exit 1;
fi

graph_dir=$dir/graph
if [ $stage -le 17 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_test $dir $graph_dir
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
          $graph_dir data/${decode_set}_hires $dir/decode${lang_test_suffix}_${decode_set}${decode_iter:+_iter$decode_iter} || exit 1;
      ) &
  done
fi

if ! $do_finetuning; then
  wait
  exit 0
fi

if [ $stage -le 19 ]; then
  mkdir -p ${dir}${finetune_suffix}
  
  for f in phone_lm.fst normalization.fst den.fst tree 0.trans_mdl cmvn_opts; do
    cp ${dir}/$f ${dir}${finetune_suffix} || exit 1
  done
  cp -r ${dir}/configs ${dir}${finetune_suffix} || exit 1

  nnet3-copy --edits="remove-output-nodes name=output;remove-output-nodes name=output-xent;rename-node old-name=output-0 new-name=output;rename-node old-name=output-0-xent new-name=output-xent" \
    $dir/${finetune_iter}.mdl ${dir}${finetune_suffix}/init.raw

  if [ $finetune_stage -le -1 ]; then
    finetune_stage=-1
  fi

  steps/nnet3/chain/train.py --stage $finetune_stage \
    --trainer.input-model ${dir}${finetune_suffix}/init.raw \
    --egs.dir "$sup_egs_dir" \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir $exp/nnet3${nnet3_affix}/ivectors_${base_train_set}_sp_hires \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" $finetune_opts \
    --chain.xent-regularize $finetune_xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights true \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width 150 \
    --trainer.num-chunk-per-minibatch "150=64/300=32" \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs $num_epochs_finetune \
    --trainer.optimization.num-jobs-initial 3 \
    --trainer.optimization.num-jobs-final 16 \
    --trainer.optimization.initial-effective-lrate 0.0001 \
    --trainer.optimization.final-effective-lrate 0.00001 \
    --trainer.max-param-change 2.0 \
    --trainer.optimization.do-final-combination false \
    --cleanup.remove-egs false \
    --feat-dir data/${supervised_set}_hires \
    --tree-dir $treedir \
    --lat-dir $sup_lat_dir \
    --dir ${dir}${finetune_suffix} || exit 1;
fi

dir=${dir}${finetune_suffix}

if [ $stage -le 20 ]; then
  for decode_set in dev test; do
      (
      num_jobs=`cat data/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $num_jobs --cmd "$decode_cmd" \
          --online-ivector-dir $exp/nnet3${nnet3_affix}/ivectors_${decode_set}_hires \
          $graph_dir data/${decode_set}_hires $dir/decode_${decode_set}${decode_iter:+_iter$decode_iter} || exit 1;
      ) &
  done
fi

wait;
exit 0;

