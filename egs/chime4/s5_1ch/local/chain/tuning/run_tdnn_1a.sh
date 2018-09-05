#!/bin/bash

# This was modified from wsj/local/chain/tunning/run_tdnn_1e.sh to be
# used in Chime4.

#This is the result using all 6 channels:
# exp/chain/tdnn1a_sp/best_wer_blstm_gev.result
# -------------------
# best overall dt05 WER 4.34% (language model weight = 7)
# -------------------
# dt05_simu WER: 4.46% (Average), 4.12% (BUS), 5.29% (CAFE), 4.00% (PEDESTRIAN), 4.42% (STREET)
# -------------------
# dt05_real WER: 4.21% (Average), 4.94% (BUS), 4.07% (CAFE), 3.81% (PEDESTRIAN), 4.04% (STREET)
# -------------------
# et05_simu WER: 5.50% (Average), 4.78% (BUS), 5.86% (CAFE), 5.51% (PEDESTRIAN), 5.83% (STREET)
# -------------------
# et05_real WER: 5.78% (Average), 6.82% (BUS), 5.10% (CAFE), 5.70% (PEDESTRIAN), 5.51% (STREET)
# -------------------
# Final train prob        -0.080
# Final valid prob        -0.075
# Final train prob (xent) -1.38
# Final valid prob (xent) -1.31

# steps/info/chain_dir_info.pl exp/chain/tdnn1a_sp
# exp/chain/tdnn1a_sp: num-iters=137 nj=2..5 num-params=10.4M dim=40+100->2781 combine=-0.095->-0.091 xent:train/valid[90,136,final]=(-1.85,-1.42,-1.38/-1.68,-1.32,-1.31) logprob:train/valid[90,136,final]=(-0.135,-0.086,-0.080/-0.115,-0.078,-0.075)


set -e -o pipefail

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=1
nj=30
train=noisy
train_set=tr05_multi_${train}
gmm=tri3b_tr05_multi_${train} # this is the source gmm-dir that we'll use for alignments; it
                              # should have alignments for the specified training data.
num_threads_ubm=32
nnet3_affix=       # affix for exp dirs, e.g. it was _cleaned in tedlium.

# Options which are not passed through to run_ivector_common.sh
affix=1a  #affix for TDNN+LSTM directory e.g. "1a" or "1b", in case we change the configuration.
common_egs_dir=
reporting_email=

# LSTM/chain options
train_stage=-10
xent_regularize=0.1

# training chunk-options
chunk_width=140,100,160
# we don't need extra left/right context for TDNN systems.
chunk_left_context=0
chunk_right_context=0

# training options
srand=0
remove_egs=true

#decode options
test_online_decoding=false  # if true, it will run the last decoding stage.
decode_only=false # if true, it wouldn't train a model again and will only do decoding
# End configuration section.
echo "$0 $@"  # Print the command line for logging


. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

enhan=$1
test_sets="dt05_real_$enhan dt05_simu_$enhan et05_real_$enhan et05_simu_$enhan"

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
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

if $decode_only; then
  mdir=`pwd`
  # check ivector extractor
  if [ ! -d $mdir/exp/nnet3${nnet3_affix}/extractor ]; then
    echo "error, set $mdir correctly"
    exit 1;
  fi
  # check tdnn graph
  if [ ! -d $mdir/exp/chain${nnet3_affix}/tree_a_sp/graph_tgpr_5k ]; then
    echo "error, set $mdir correctly"
    exit 1;
  fi
  # check dir
  if [ ! -d $mdir/exp/chain${nnet3_affix}/tdnn${affix}_sp ]; then
    echo "error, set $mdir correctly"
    exit 1;
  fi

  # make ivector for dev and eval
  for datadir in ${test_sets}; do
    utils/copy_data_dir.sh data/$datadir data/${datadir}_hires
  done

  # extracting hires features
  for datadir in ${test_sets}; do
    steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/${datadir}_hires
    steps/compute_cmvn_stats.sh data/${datadir}_hires
    utils/fix_data_dir.sh data/${datadir}_hires
  done

  # extract iVectors for the test data, but in this case we don't need the speed
  # perturbation (sp).
  for data in ${test_sets}; do
    nspk=$(wc -l <data/${data}_hires/spk2utt)
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj "${nspk}" \
    data/${data}_hires exp/nnet3${nnet3_affix}/extractor \
    exp/nnet3${nnet3_affix}/ivectors_${data}_hires
  done
  # directly do decoding
  stage=18
else
  local/nnet3/run_ivector_common.sh \
    --stage $stage --nj $nj \
    --train-set "$train_set" --gmm $gmm \
    --test-sets "$test_sets" \
    --num-threads-ubm $num_threads_ubm \
    --nnet3-affix "$nnet3_affix"
fi

gmm_dir=exp/${gmm}
ali_dir=exp/${gmm}_ali_${train_set}_sp
lat_dir=exp/chain${nnet3_affix}/${gmm}_${train_set}_sp_lats
dir=exp/chain${nnet3_affix}/tdnn${affix}_sp
train_data_dir=data/${train_set}_sp_hires
train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires
lores_train_data_dir=data/${train_set}_sp

# note: you don't necessarily have to change the treedir name
# each time you do a new experiment-- only if you change the
# configuration in a way that affects the tree.
tree_dir=exp/chain${nnet3_affix}/tree_a_sp
# the 'lang' directory is created by this script.
# If you create such a directory with a non-standard topology
# you should probably name it differently.
lang=data/lang_chain

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

if [ $stage -le 13 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --nj 100 --cmd "$train_cmd" ${lores_train_data_dir} \
    data/lang $gmm_dir $lat_dir
  rm $lat_dir/fsts.*.gz # save space
fi

if [ $stage -le 14 ]; then
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


if [ $stage -le 15 ]; then
  mkdir -p $dir
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)
  opts="l2-regularize=0.01"
  output_opts="l2-regularize=0.005"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-2,-1,0,1,2,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 $opts dim=850
  relu-batchnorm-layer name=tdnn2 $opts dim=850 input=Append(-1,0,1)
  relu-batchnorm-layer name=tdnn3 $opts dim=850
  relu-batchnorm-layer name=tdnn4 $opts dim=850 input=Append(-1,0,1)
  relu-batchnorm-layer name=tdnn5 $opts dim=850
  relu-batchnorm-layer name=tdnn6 $opts dim=850 input=Append(-3,0,3)
  relu-batchnorm-layer name=tdnn7 $opts dim=850 input=Append(-3,0,3)
  relu-batchnorm-layer name=tdnn8 $opts dim=850 input=Append(-6,-3,0)

  ## adding the layers for chain branch
  relu-batchnorm-layer name=prefinal-chain $opts dim=850 target-rms=0.5
  output-layer name=output $output_opts include-log-softmax=false dim=$num_targets max-change=1.5

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  relu-batchnorm-layer name=prefinal-xent $opts input=tdnn8 dim=850 target-rms=0.5
  output-layer name=output-xent $output_opts dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi


if [ $stage -le 16 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/chime4-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi
  
  cat $train_data_dir/utt2uniq | awk -F' ' '{print $1}' > $train_data_dir/utt2uniq.tmp1
  cat $train_data_dir/utt2uniq | awk -F' ' '{print $2}' | sed -e 's/\....//g' | sed -e 's/\_CH.//g' | sed -e 's/\_enhan//g' > $train_data_dir/utt2uniq.tmp2
  paste -d" " $train_data_dir/utt2uniq.tmp1 $train_data_dir/utt2uniq.tmp2 > $train_data_dir/utt2uniq
  rm -rf $train_data_dir/utt2uniq.tmp{1,2}
  
  steps/nnet3/chain/train.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.online-ivector-dir=$train_ivector_dir \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient=0.1 \
    --chain.l2-regularize=0.00005 \
    --chain.apply-deriv-weights=false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=12 \
    --trainer.frames-per-iter=3000000 \
    --trainer.optimization.num-jobs-initial=2 \
    --trainer.optimization.num-jobs-final=12 \
    --trainer.optimization.initial-effective-lrate=0.005 \
    --trainer.optimization.final-effective-lrate=0.0005 \
    --trainer.optimization.shrink-value=1.0 \
    --trainer.num-chunk-per-minibatch=128,64 \
    --trainer.optimization.momentum=0.0 \
    --trainer.optimization.backstitch-training-scale=0.3 \
    --trainer.optimization.backstitch-training-interval=1 \
    --egs.chunk-width=$chunk_width \
    --egs.chunk-left-context=0 \
    --egs.chunk-right-context=0 \
    --egs.chunk-left-context-initial=0 \
    --egs.chunk-right-context-final=0 \
    --egs.dir="$common_egs_dir" \
    --egs.opts="--frames-overlap-per-eg 0" \
    --cleanup.remove-egs=$remove_egs \
    --use-gpu=true \
    --reporting.email="$reporting_email" \
    --feat-dir=$train_data_dir \
    --tree-dir=$tree_dir \
    --lat-dir=$lat_dir \
    --dir=$dir  || exit 1;
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
      utils/data/modify_speaker_info.sh --seconds-per-spk-max 200 \
        data/${data}_hires data/${data}_chunked
      
      data_affix=$(echo $data | sed s/test_//)
      nspk=$(wc -l <data/${data}_chunked/spk2utt)
      for lmtype in tgpr_5k; do
        steps/nnet3/decode.sh \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --extra-left-context 0 --extra-right-context 0 \
          --extra-left-context-initial 0 \
          --extra-right-context-final 0 \
          --frames-per-chunk $frames_per_chunk \
          --nj $nspk --cmd "$decode_cmd"  --num-threads 4 \
          --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${data}_hires \
          $tree_dir/graph_${lmtype} data/${data}_chunked ${dir}/decode_${lmtype}_${data_affix} || exit 1
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
      for lmtype in tgpr_5k; do
        steps/online/nnet3/decode.sh \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $nspk --cmd "$decode_cmd" \
          $tree_dir/graph_${lmtype} data/${data} ${dir}_online/decode_${lmtype}_${data_affix} || exit 1
      done
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

# scoring
if [ $stage -le 20 ]; then
  # decoded results of enhanced speech using TDNN AMs trained with enhanced data
  local/chime4_calc_wers.sh exp/chain/tdnn${affix}_sp $enhan exp/chain/tree_a_sp/graph_tgpr_5k \
    > exp/chain/tdnn${affix}_sp/best_wer_$enhan.result
  head -n 15 exp/chain/tdnn${affix}_sp/best_wer_$enhan.result
fi


exit 0;
