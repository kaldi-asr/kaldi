#!/bin/bash

# 1d is as 1c but introducing two non-splicing layers towards the beginning of
#   the network.

# local/chain/compare_wer.sh exp/chain/tdnn1c_sp exp/chain/tdnn1e3_sp
# System                tdnn1c_sp tdnn1e3_sp
#WER dev93 (tgpr)                7.31      6.95
#WER dev93 (tg)                  6.98      6.79
#WER dev93 (big-dict,tgpr)       5.17      5.20
#WER dev93 (big-dict,fg)         4.70      4.65
#WER eval92 (tgpr)               4.96      5.09
#WER eval92 (tg)                 4.82      4.59
#WER eval92 (big-dict,tgpr)      2.98      2.82
#WER eval92 (big-dict,fg)        2.66      2.52
# Final train prob        -0.0559   -0.0554
# Final valid prob        -0.0669   -0.0648
# Final train prob (xent)   -0.9369   -0.9134
# Final valid prob (xent)   -0.9838   -0.9476

# steps/info/chain_dir_info.pl exp/chain/tdnn1d_sp
# exp/chain/tdnn1d_sp: num-iters=102 nj=2..5 num-params=8.1M dim=40+100->2854 combine=-0.065->-0.062 xent:train/valid[67,101,final]=(-1.02,-0.904,-0.913/-1.02,-0.940,-0.948) logprob:train/valid[67,101,final]=(-0.069,-0.056,-0.055/-0.073,-0.065,-0.065)


set -e -o pipefail

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=0
nj=30
train=noisy
enhan=$1
mdir=$2
train_set=tr05_multi_${train}
test_sets="dt05_real_$enhan dt05_simu_$enhan et05_real_$enhan et05_simu_$enhan"
gmm=tri3b_tr05_multi_${train} # this is the source gmm-dir that we'll use for alignments; it
                              # should have alignments for the specified training data.
nnet3_affix=       # affix for exp dirs, e.g. it was _cleaned in tedlium.

# Options which are not passed through to run_ivector_common.sh
affix=1d  #affix for TDNN+LSTM directory e.g. "1a" or "1b", in case we change the configuration.
common_egs_dir=
reporting_email=

# training chunk-options
chunk_width=140,100,160
# we don't need extra left/right context for TDNN systems.
chunk_left_context=0
chunk_right_context=0

#decode options
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

if [ $# -ne 2 ]; then
  printf "\nUSAGE: %s <enhancement method> <model dir>\n\n" `basename $0`
  echo "First argument specifies a unique name for different enhancement method"
  echo "Second argument specifies acoustic and language model directory"
  exit 1;
fi

# check whether run_init is executed
if [ ! -d data/lang ]; then
  echo "error, execute local/run_init.sh, first"
  exit 1;
fi

# check whether run_init is executed
if [ ! -d exp/tri3b_tr05_multi_${train} ]; then
  echo "error, execute local/run_init.sh, first"
  exit 1;
fi

gmm_dir=$mdir/exp/${gmm}
ali_dir=$mdir/exp/${gmm}_ali_${train_set}_sp
lat_dir=$mdir/exp/chain${nnet3_affix}/${gmm}_${train_set}_sp_lats
dir=$mdir/exp/chain${nnet3_affix}/tdnn${affix}_sp
train_data_dir=$mdir/data/${train_set}_sp_hires
train_ivector_dir=$mdir/exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires
lores_train_data_dir=$mdir/data/${train_set}_sp

# note: you don't necessarily have to change the treedir name
# each time you do a new experiment-- only if you change the
# configuration in a way that affects the tree.
tree_dir=$mdir/exp/chain${nnet3_affix}/tree_a_sp
# the 'lang' directory is created by this script.
# If you create such a directory with a non-standard topology
# you should probably name it differently.
lang=$mdir/data/lang_chain

for f in $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp \
    $lores_train_data_dir/feats.scp $gmm_dir/final.mdl \
    $ali_dir/ali.1.gz $gmm_dir/final.mdl; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done

if [ $stage -le 12 ]; then
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

if [ $stage -le 14 ]; then
  # Build a tree using our new topology.  We know we have alignments for the
  # speed-perturbed data (local/nnet3/run_ivector_common.sh made them), so use
  # those.  The num-leaves is always somewhat less than the num-leaves from
  # the GMM baseline.
  if [ -f $tree_dir/final.mdl ]; then
     echo "$0: $tree_dir/final.mdl already exists, refusing to overwrite it."
  else
    steps/nnet3/chain/build_tree.sh \
      --frame-subsampling-factor 3 \
      --context-opts "--context-width=2 --central-position=1" \
      --cmd "$train_cmd" 3500 ${lores_train_data_dir} \
      $lang $ali_dir $tree_dir
  fi
fi

if [ $stage -le 17 ]; then
  # The reason we are using data/lang here, instead of $lang, is just to
  # emphasize that it's not actually important to give mkgraph.sh the
  # lang directory with the matched topology (since it gets the
  # topology file from the model).  So you could give it a different
  # lang directory, one that contained a wordlist and LM of your choice,
  # as long as phones.txt was compatible.

  utils/lang/check_phones_compatible.sh \
    data/lang_test_tgpr_5k/phones.txt $lang/phones.txt
  utils/mkgraph.sh \
    --self-loop-scale 1.0 data/lang_test_tgpr_5k \
    $tree_dir $tree_dir/graph_tgpr_5k || exit 1;
fi

if [ $stage -le 18 ]; then
  frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    (
      data_affix=$(echo $data | sed s/test_//)
      nspk=$(wc -l <data/${data}_hires/spk2utt)
      for lmtype in tgpr_5k; do
        steps/nnet3/decode.sh \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --extra-left-context 0 --extra-right-context 0 \
          --extra-left-context-initial 0 \
          --extra-right-context-final 0 \
          --frames-per-chunk $frames_per_chunk \
          --nj $nspk --cmd "$decode_cmd"  --num-threads 4 \
          --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${data}_hires \
          $tree_dir/graph_${lmtype} data/${data}_hires ${dir}/decode_${lmtype}_${data_affix} || exit 1
      done
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

# Not testing the 'looped' decoding separately, because for
# TDNN systems it would give exactly the same results as the
# normal decoding.

if $test_online_decoding && [ $stage -le 19 ]; then
  # note: if the features change (e.g. you add pitch features), you will have to
  # change the options of the following command line.
  steps/online/nnet3/prepare_online_decoding.sh \
    --mfcc-config conf/mfcc_hires.conf \
    $lang exp/nnet3${nnet3_affix}/extractor ${dir} ${dir}_online

  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    (
      data_affix=$(echo $data | sed s/test_//)
      nspk=$(wc -l <data/${data}_hires/spk2utt)
      # note: we just give it "data/${data}" as it only uses the wav.scp, the
      # feature type does not matter.
      for lmtype in tgpr bd_tgpr; do
        steps/online/nnet3/decode.sh \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $nspk --cmd "$decode_cmd" \
          $tree_dir/graph_${lmtype} data/${data} ${dir}_online/decode_${lmtype}_${data_affix} || exit 1
      done
      steps/lmrescore.sh \
        --self-loop-scale 1.0 \
        --cmd "$decode_cmd" data/lang_test_{tgpr,tg} \
        data/${data}_hires ${dir}_online/decode_{tgpr,tg}_${data_affix} || exit 1
      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
        data/lang_test_bd_{tgpr,fgconst} \
       data/${data}_hires ${dir}_online/decode_${lmtype}_${data_affix}{,_fg} || exit 1
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

# scoring
if [ $stage -le 20 ]; then
  # decoded results of enhanced speech using TDNN AMs trained with enhanced data
  local/chime4_calc_wers.sh exp/chain/tdnn1d_sp $enhan exp/chain/tree_a_sp/graph_tgpr_5k \
    > exp/chain/tdnn1d_sp/best_wer_$enhan.result
  head -n 15 exp/chain/tdnn1d_sp/best_wer_$enhan.result
fi


exit 0;
