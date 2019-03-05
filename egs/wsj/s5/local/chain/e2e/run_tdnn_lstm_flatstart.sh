#!/bin/bash
# Copyright    2017  Hossein Hadian

# This is a TDNN-LSTM recipe that performs chain training in a flat-start manner
# Unlike run_tdnn_flatstart.sh which is context-independent, this recipe uses
# a full trivial biphone context-dependency tree. This is because this recipe is
# meant for character-based (i.e. lexicon-free) modeling where context helps
# significantly.
# It does not use ivecors or other forms of speaker adaptation.
# It is called from run_e2e_char.sh

# Note: this script is configured to run as character-based, if you want
# to run it in phoneme mode, you'll need to change _char
# to _nosp everywhere.


# local/chain/compare_wer.sh exp/chain/e2e_tdnn_lstm_bichar_1a
# System                e2e_tdnn_lstm_bichar_1a
#WER dev93 (tgpr)                9.85
#WER dev93 (tg)                  9.32
#WER dev93 (big-dict,tgpr)       8.19
#WER dev93 (big-dict,fg)         7.27
#WER eval92 (tgpr)               6.89
#WER eval92 (tg)                 6.70
#WER eval92 (big-dict,tgpr)      5.14
#WER eval92 (big-dict,fg)        4.29
# Final train prob        -0.0610
# Final valid prob        -0.0836
# Final train prob (xent)
# Final valid prob (xent)
# Num-params                 9219188

# steps/info/chain_dir_info.pl exp/chain/e2e_tdnn_lstm_bichar_1a/
# exp/chain/e2e_tdnn_lstm_bichar_1a_nocmvn: num-iters=138 nj=2..5 num-params=9.2M dim=40->3444 combine=-1.211->-1.211 (over 3) logprob:train/valid[91,137,final]=(-0.079,-0.062,-0.061/-0.093,-0.084,-0.084)

set -e

# configs for 'chain'
stage=0
train_stage=-10
get_egs_stage=-10
affix=_1a
decode_iter=

# training options
num_epochs=4.5
num_jobs_initial=2
num_jobs_final=5
minibatch_size=150=128,64/300=64,32/600=32,16/1200=16,8
common_egs_dir=
l2_regularize=0.00001
dim=512
frames_per_iter=2500000
cmvn_opts="--norm-means=false --norm-vars=false"
train_set=train_si284_spe2e_hires
test_sets="test_dev93 test_eval92"

chunk_left_context=40
chunk_right_context=0
extra_left_context=50
extra_right_context=0

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

lang=data/lang_e2e_char
treedir=exp/chain/e2e_bichar_tree
dir=exp/chain/e2e_tdnn_lstm_bichar${affix}

if [ $stage -le 0 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  rm -rf $lang
  cp -r data/lang_char $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi

if [ $stage -le 1 ]; then
  echo "$0: Estimating a phone language model for the denominator graph..."
  mkdir -p $treedir/log
  $train_cmd $treedir/log/make_phone_lm.log \
             cat data/$train_set/text \| \
             steps/nnet3/chain/e2e/text_to_phones.py --between-silprob 0.1 \
             data/lang_char \| \
             utils/sym2int.pl -f 2- data/lang_char/phones.txt \| \
             chain-est-phone-lm --num-extra-lm-states=2000 \
             ark:- $treedir/phone_lm.fst
  steps/nnet3/chain/e2e/prepare_e2e.sh --nj 30 --cmd "$train_cmd" \
                                       --type biphone \
                                       --shared-phones true \
                                       data/$train_set $lang $treedir
fi

if [ $stage -le 2 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $treedir/tree | grep num-pdfs | awk '{print $2}')

  pdim=$[dim/4]
  npdim=$[dim/4]
  opts="l2-regularize=0.01"
  lstm_opts="l2-regularize=0.0025"
  output_opts="l2-regularize=0.0025"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig

  input dim=40 name=input

  relu-batchnorm-layer name=tdnn1 input=Append(-1,0,1) dim=$dim $opts
  relu-batchnorm-layer name=tdnn2 input=Append(-1,0,1) dim=$dim $opts
  relu-batchnorm-layer name=tdnn3 input=Append(-1,0,1) dim=$dim $opts

  # check steps/libs/nnet3/xconfig/lstm.py for the other options and defaults
  fast-lstmp-layer name=fastlstm1 cell-dim=$dim recurrent-projection-dim=$pdim non-recurrent-projection-dim=$npdim delay=-3 $lstm_opts
  relu-batchnorm-layer name=tdnn4 input=Append(-3,0,3) dim=$dim $opts
  relu-batchnorm-layer name=tdnn5 input=Append(-3,0,3) dim=$dim $opts
  fast-lstmp-layer name=fastlstm2 cell-dim=$dim recurrent-projection-dim=$pdim non-recurrent-projection-dim=$npdim delay=-3 $lstm_opts
  relu-batchnorm-layer name=tdnn6 input=Append(-3,0,3) dim=$dim $opts
  relu-batchnorm-layer name=tdnn7 input=Append(-3,0,3) dim=$dim $opts
  fast-lstmp-layer name=fastlstm3 cell-dim=$dim recurrent-projection-dim=$pdim non-recurrent-projection-dim=$npdim delay=-3 $lstm_opts

  output-layer name=output include-log-softmax=true dim=$num_targets max-change=1.5 $output_opts

EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 3 ]; then
  # no need to store the egs in a shared storage because we always
  # remove them. Anyway, it takes only 5 minutes to generate them.

  steps/nnet3/chain/e2e/train_e2e.py --stage $train_stage \
    --cmd "$train_cmd" \
    --feat.cmvn-opts "$cmvn_opts" \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize $l2_regularize \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "--normalize-egs true --num-train-egs-combine 800" \
    --egs.chunk-left-context $chunk_left_context \
    --egs.chunk-right-context $chunk_right_context \
    --egs.chunk-left-context-initial 0 \
    --egs.chunk-right-context-final 0 \
    --trainer.num-chunk-per-minibatch $minibatch_size \
    --trainer.frames-per-iter $frames_per_iter \
    --trainer.num-epochs $num_epochs \
    --trainer.deriv-truncate-margin 8 \
    --trainer.optimization.momentum 0 \
    --trainer.optimization.num-jobs-initial $num_jobs_initial \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.initial-effective-lrate 0.001 \
    --trainer.optimization.final-effective-lrate 0.0001 \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs true \
    --cleanup.preserve-model-interval 50 \
    --feat-dir data/${train_set} \
    --tree-dir $treedir \
    --dir $dir  || exit 1;
fi

if [ $stage -le 4 ]; then
  # The reason we are using data/lang here, instead of $lang, is just to
  # emphasize that it's not actually important to give mkgraph.sh the
  # lang directory with the matched topology (since it gets the
  # topology file from the model).  So you could give it a different
  # lang directory, one that contained a wordlist and LM of your choice,
  # as long as phones.txt was compatible.

  utils/lang/check_phones_compatible.sh \
    data/lang_char_test_tgpr/phones.txt $lang/phones.txt
  utils/mkgraph.sh \
    --self-loop-scale 1.0 data/lang_char_test_tgpr \
    $dir $treedir/graph_tgpr || exit 1;

  utils/lang/check_phones_compatible.sh \
    data/lang_char_test_bd_tgpr/phones.txt $lang/phones.txt
  utils/mkgraph.sh \
    --self-loop-scale 1.0 data/lang_char_test_bd_tgpr \
    $dir $treedir/graph_bd_tgpr || exit 1;
fi

if [ $stage -le 5 ]; then
  frames_per_chunk=150
  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    (
      data_affix=$(echo $data | sed s/test_//)
      nspk=$(wc -l <data/${data}_hires/spk2utt)
      for lmtype in tgpr bd_tgpr; do
        steps/nnet3/decode.sh \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --extra-left-context $chunk_left_context \
          --extra-right-context $chunk_right_context \
          --extra-left-context-initial 0 \
          --extra-right-context-final 0 \
          --frames-per-chunk $frames_per_chunk \
          --nj $nspk --cmd "$decode_cmd"  --num-threads 4 \
          $treedir/graph_${lmtype} data/${data}_hires ${dir}/decode_${lmtype}_${data_affix} || exit 1
      done
      steps/lmrescore.sh \
        --self-loop-scale 1.0 \
        --cmd "$decode_cmd" data/lang_char_test_{tgpr,tg} \
        data/${data}_hires ${dir}/decode_{tgpr,tg}_${data_affix} || exit 1
      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
        data/lang_char_test_bd_{tgpr,fgconst} \
       data/${data}_hires ${dir}/decode_${lmtype}_${data_affix}{,_fg} || exit 1
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

echo "Done. Date: $(date). Results:"
local/chain/compare_wer.sh $dir
