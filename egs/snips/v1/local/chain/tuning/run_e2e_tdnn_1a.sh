#!/bin/bash
# Copyright 2018-2020  Daniel Povey
#           2018-2020  Yiming Wang

set -e

# configs for 'chain'
stage=0
train_stage=-10
affix=1a
remove_egs=false
xent_regularize=0.1
online_cmvn=true

# training options
srand=0
num_epochs=3
num_jobs_initial=2
num_jobs_final=5
minibatch_size=150=128,64/300=100,64,32/600=50,32,16/1200=16,8
common_egs_dir=
dim=80
bn_dim=20
frames_per_iter=3000000
bs_scale=0.0
train_set=train_shorter_combined_spe2e
test_sets="dev eval"
wake_word="HeySnips"

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

lang=data/lang_e2e
lang_decode=data/lang_e2e_decode
tree_dir=exp/chain/e2e_tree  # it's actually just a trivial tree (no tree building)
dir=exp/chain/e2e_tdnn_${affix}

if [ $stage -le 0 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  if [ -d $lang ]; then
    if [ $lang/L.fst -nt data/lang/L.fst ]; then
      echo "$0: $lang already exists, not overwriting it; continuing"
    else
      echo "$0: $lang already exists and seems to be older than data/lang ..."
      echo " ... not sure what to do.  Exiting."
      exit 1;
    fi
  else
    cp -r data/lang $lang
    silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
    nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
    # Use our special topology... note that later on may have to tune this
    # topology.
    local/gen_topo.pl 4 1 $nonsilphonelist $silphonelist >$lang/topo
  fi
fi

if [ $stage -le 1 ]; then
  echo "$0: Creating an unnormalized phone language model for the denominator graph..."
  mkdir -p $tree_dir
  id_sil=`cat data/lang/phones.txt | grep "SIL" | awk '{print $2}'`
  id_word=`cat data/lang/phones.txt | grep "heysnips" | awk '{print $2}'`
  id_freetext=`cat data/lang/phones.txt | grep "freetext" | awk '{print $2}'`
  cat <<EOF > $tree_dir/phone_lm.txt
0 1 $id_sil $id_sil
0 5 $id_sil $id_sil
1 2 $id_word $id_word
2 3 $id_sil $id_sil
1 4 $id_freetext $id_freetext
4 5 $id_sil $id_sil
3 2.09
5 0.0
EOF
  fstcompile $tree_dir/phone_lm.txt $tree_dir/phone_lm.fst
  fstdeterminizestar $tree_dir/phone_lm.fst $tree_dir/phone_lm.fst.tmp
  mv $tree_dir/phone_lm.fst.tmp $tree_dir/phone_lm.fst
  steps/nnet3/chain/e2e/prepare_e2e.sh --nj 30 --cmd "$train_cmd" \
                                       data/${train_set}_hires $lang $tree_dir
fi

if [ $stage -le 2 ]; then
  echo "$0: creating neural net configs using the xconfig parser";
  num_targets=$(tree-info $tree_dir/tree | grep num-pdfs | awk '{print $2}')
  learning_rate_factor=$(python3 -c "print(0.5/$xent_regularize)")
  affine_opts="l2-regularize=0.01 dropout-proportion=0.0 dropout-per-dim=true dropout-per-dim-continuous=true"
  tdnnf_opts="l2-regularize=0.01 dropout-proportion=0.0 bypass-scale=0.66"
  linear_opts="l2-regularize=0.01 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.01"
  output_opts="l2-regularize=0.002"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor

  relu-batchnorm-dropout-layer name=tdnn1 input=Append(-2,-1,0,1,2) $affine_opts dim=$dim
  tdnnf-layer name=tdnnf2 $tdnnf_opts dim=$dim bottleneck-dim=$bn_dim time-stride=1
  tdnnf-layer name=tdnnf3 $tdnnf_opts dim=$dim bottleneck-dim=$bn_dim time-stride=1
  tdnnf-layer name=tdnnf4 $tdnnf_opts dim=$dim bottleneck-dim=$bn_dim time-stride=1
  tdnnf-layer name=tdnnf5 $tdnnf_opts dim=$dim bottleneck-dim=$bn_dim time-stride=1
  tdnnf-layer name=tdnnf6 $tdnnf_opts dim=$dim bottleneck-dim=$bn_dim time-stride=1
  tdnnf-layer name=tdnnf7 $tdnnf_opts dim=$dim bottleneck-dim=$bn_dim time-stride=1
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=$dim bottleneck-dim=$bn_dim time-stride=1
  tdnnf-layer name=tdnnf9 $tdnnf_opts dim=$dim bottleneck-dim=$bn_dim time-stride=0
  tdnnf-layer name=tdnnf10 $tdnnf_opts dim=$dim bottleneck-dim=$bn_dim time-stride=3
  tdnnf-layer name=tdnnf11 $tdnnf_opts dim=$dim bottleneck-dim=$bn_dim time-stride=3
  tdnnf-layer name=tdnnf12 $tdnnf_opts dim=$dim bottleneck-dim=$bn_dim time-stride=3
  tdnnf-layer name=tdnnf13 $tdnnf_opts dim=$dim bottleneck-dim=$bn_dim time-stride=3
  tdnnf-layer name=tdnnf14 $tdnnf_opts dim=$dim bottleneck-dim=$bn_dim time-stride=3
  tdnnf-layer name=tdnnf15 $tdnnf_opts dim=$dim bottleneck-dim=$bn_dim time-stride=3
  tdnnf-layer name=tdnnf16 $tdnnf_opts dim=$dim bottleneck-dim=$bn_dim time-stride=3
  tdnnf-layer name=tdnnf17 $tdnnf_opts dim=$dim bottleneck-dim=$bn_dim time-stride=3
  tdnnf-layer name=tdnnf18 $tdnnf_opts dim=$dim bottleneck-dim=$bn_dim time-stride=3
  tdnnf-layer name=tdnnf19 $tdnnf_opts dim=$dim bottleneck-dim=$bn_dim time-stride=3
  tdnnf-layer name=tdnnf20 $tdnnf_opts dim=$dim bottleneck-dim=$bn_dim time-stride=3
  linear-component name=prefinal-l dim=30 $linear_opts

  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts big-dim=$dim small-dim=30
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts big-dim=$dim small-dim=30
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF

  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs
fi

if [ $stage -le 3 ]; then
  # no need to store the egs in a shared storage because we always
  # remove them. Anyway, it takes only 5 minutes to generate them.

  steps/nnet3/chain/e2e/train_e2e.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.cmvn-opts="--config=conf/online_cmvn.conf" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient=0.1 \
    --chain.l2-regularize=0.0 \
    --chain.apply-deriv-weights=false \
    --chain.frame-subsampling-factor=3 \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=$num_epochs \
    --trainer.frames-per-iter $frames_per_iter \
    --trainer.optimization.num-jobs-initial $num_jobs_initial \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.initial-effective-lrate 0.00003 \
    --trainer.optimization.final-effective-lrate 0.000003 \
    --trainer.optimization.backstitch-training-scale $bs_scale \
    --trainer.num-chunk-per-minibatch $minibatch_size \
    --trainer.optimization.momentum=0.0 \
    --egs.dir "$common_egs_dir" \
    --egs.opts "--num-utts-subset 300 --online-cmvn $online_cmvn" \
    --cleanup.remove-egs=$remove_egs \
    --feat-dir data/${train_set}_hires \
    --tree-dir $tree_dir \
    --dir=$dir  || exit 1;
fi

if [ $stage -le 4 ]; then
  steps/online/nnet3/prepare_online_decoding.sh \
    --mfcc-config conf/mfcc_hires.conf \
    --online-cmvn-config conf/online_cmvn.conf \
    $lang ${dir} ${dir}_online

  rm $dir/.error 2>/dev/null || true
  for wake_word_cost in -2.5 -2.0 -1.5 -1.0 -0.5 0.0 0.5 1.0 1.5 2.0 2.5; do
    rm -rf $lang_decode
    utils/prepare_lang.sh --num-sil-states 1 --num-nonsil-states 4 --sil-prob 0.0 \
      --position-dependent-phones false \
      data/local/dict "<sil>" $lang_decode/temp $lang_decode

    sil_id=`cat $lang_decode/words.txt | grep "<sil>" | awk '{print $2}'`
    freetext_id=`cat $lang_decode/words.txt | grep "FREETEXT" | awk '{print $2}'`
    id=`cat $lang_decode/words.txt | grep $wake_word | awk '{print $2}'`
    mkdir -p $lang_decode/lm
    cat <<EOF > $lang_decode/lm/fst.txt
0 1 $sil_id $sil_id
0 4 $sil_id $sil_id 7.0
1 4 $freetext_id $freetext_id 0.0
4 0 $sil_id $sil_id
1 2 $id $id $wake_word_cost
2 0 $sil_id $sil_id
0
EOF
    fstcompile $lang_decode/lm/fst.txt $lang_decode/G.fst
    set +e
    fstisstochastic $lang_decode/G.fst
    set -e
    utils/validate_lang.pl $lang_decode
    cp $lang/topo $lang_decode/topo

    utils/lang/check_phones_compatible.sh \
      data/lang/phones.txt $lang_decode/phones.txt
    rm -rf $tree_dir/graph_online/HCLG.fst
    utils/mkgraph.sh \
      --self-loop-scale 1.0 $lang_decode \
      $dir $tree_dir/graph_online || exit 1;

    frames_per_chunk=150
    for data in $test_sets; do
      (
        nj=30
        steps/online/nnet3/decode_wake_word.sh \
          --beam 200 --acwt 1.0 \
          --wake-word $wake_word \
          --extra-left-context-initial 0 \
          --frames-per-chunk $frames_per_chunk \
          --nj $nj --cmd "$decode_cmd" \
          $tree_dir/graph_online data/${data}_hires ${dir}_online/decode_${data}_cost$wake_word_cost || exit 1
      ) || touch $dir/.error &
    done
    wait
    [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
  done
  for data in $test_sets; do
    echo "Results on $data set:"
    cat ${dir}_online/decode_${data}_cost*/scoring_kaldi/all_results
  done
 
fi

