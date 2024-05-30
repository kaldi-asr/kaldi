#!/usr/bin/env bash


# run_tdnn_lfr_1a.sh is similar in configuration to run_tdnn_1c.sh, but it's a
# low-frame-rate system (see egs/swbd/s5c/local/nnet3/tuning/run_tdnn_lfr1c.sh
# for an example of such a system).


# by default, with cleanup:
# local/nnet3/run_tdnn_lfr.sh

# without cleanup:
# local/nnet3/run_tdnn_lfr.sh  --train-set train --gmm tri3 --nnet3-affix "" &


set -e -o pipefail

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=0
nj=30
decode_nj=30
min_seg_len=1.55
train_set=train_cleaned
gmm=tri3_cleaned  # this is the source gmm-dir for the data-type of interest; it
                  # should have alignments for the specified training data.
num_threads_ubm=32
nnet3_affix=_cleaned  # cleanup affix for exp dirs, e.g. _cleaned
tdnn_affix=1a  #affix for TDNN directory e.g. "a" or "b", in case we change the configuration.

# Options which are not passed through to run_ivector_common.sh
train_stage=-10
remove_egs=true
srand=0
reporting_email=dpovey@gmail.com
# set common_egs_dir to use previously dumped egs.
common_egs_dir=

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

local/nnet3/run_ivector_common.sh --stage $stage \
                                  --nj $nj \
                                  --min-seg-len $min_seg_len \
                                  --train-set $train_set \
                                  --gmm $gmm \
                                  --num-threads-ubm $num_threads_ubm \
                                  --nnet3-affix "$nnet3_affix"



gmm_dir=exp/${gmm}
ali_dir=exp/${gmm}_ali_${train_set}_sp_comb
dir=exp/nnet3${nnet3_affix}/tdnn_lfr${tdnn_affix}_sp
# note: you don't necessarily have to change the treedir name
# each time you do a new experiment-- only if you change the
# configuration in a way that affects the tree.
treedir=exp/nnet3${nnet3_affix}/tree_lfr_a_sp
# the 'lang' directory is created by this script; it's one
# suitable for a low-frame-rate system such as this one.
lang=data/lang_lfr_a

train_data_dir=data/${train_set}_sp_hires_comb
train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires_comb


for f in $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp \
    $ali_dir/ali.1.gz $gmm_dir/final.mdl; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done


if [ $stage -le 12 ]; then
  echo "$0: creating lang directory $lang with chain-type topology"
  rm -rf $lang
  cp -r data/lang $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi

if [ $stage -le 13 ]; then
  # Build a tree using our new topology and a reduced sampling rate.
  # We use 4000 leaves, which is a little less than the number used
  # in the baseline GMM system (5k) in this setup, since generally
  # LFR systems do best with somewhat fewer leaves.
  #
  # To get the stats to build the tree this script only uses every third frame,
  # but it dumps converted alignments that essentially have 3 different
  # frame-shifted versions of the alignment interpolated together; these can be
  # used without modification in getting labels for training.
  steps/nnet3/chain/build_tree.sh \
    --repeat-frames true --frame-subsampling-factor 3 \
    --cmd "$train_cmd" 4000 data/${train_set}_sp_comb \
    $lang $ali_dir $treedir
fi

if [ $stage -le 14 ]; then
  mkdir -p $dir
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-2,-1,0,1,2,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-renorm-layer name=tdnn1 dim=750
  relu-renorm-layer name=tdnn2 dim=750 input=Append(-1,0,1)
  relu-renorm-layer name=tdnn3 dim=750 input=Append(-1,0,1)
  relu-renorm-layer name=tdnn4 dim=750 input=Append(-3,0,3)
  relu-renorm-layer name=tdnn5 dim=750 input=Append(-6,-3,0)
  output-layer name=output dim=$num_targets max-change=1.5
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi



if [ $stage -le 15 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/tedlium-$(date +'%m_%d_%H_%M')/s5_r2/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/train_dnn.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.online-ivector-dir=$train_ivector_dir \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=3 \
    --trainer.samples-per-iter=400000 \
    --trainer.optimization.num-jobs-initial=2 \
    --trainer.optimization.num-jobs-final=12 \
    --trainer.optimization.initial-effective-lrate=0.0015 \
    --trainer.optimization.final-effective-lrate=0.00015 \
    --trainer.optimization.minibatch-size=256,128 \
    --egs.dir="$common_egs_dir" \
    --cleanup.remove-egs=$remove_egs \
    --use-gpu=true \
    --feat-dir=$train_data_dir \
    --ali-dir=$treedir \
    --lang=$lang \
    --reporting.email="$reporting_email" \
    --dir=$dir  || exit 1;
  echo 3 >$dir/frame_subsampling_factor
fi

if [ $stage -le 16 ]; then
  # The reason we are using data/lang here, instead of $lang, is just to
  # emphasize that it's not actually important to give mkgraph.sh the
  # lang directory with the matched topology (since it gets the
  # topology file from the model).  So you could give it a different
  # lang directory, one that contained a wordlist and LM of your choice,
  # as long as phones.txt was compatible.

  utils/lang/check_phones_compatible.sh data/lang/phones.txt $lang/phones.txt
  utils/mkgraph.sh --self-loop-scale 0.333 data/lang $dir $dir/graph
fi


if [ $stage -le 17 ]; then
  # note: for TDNNs, looped decoding gives exactly the same results
  # as regular decoding, so there is no point in testing it separately.
  # We use regular decoding because it supports multi-threaded (we just
  # didn't create the binary for that, for looped decoding, so far).
  rm $dir/.error || true 2>/dev/null
  for dset in dev test; do
   (
     steps/nnet3/decode.sh --acwt 0.333 --post-decode-acwt 3.0 --nj $decode_nj \
        --cmd "$decode_cmd"  --num-threads 4 \
        --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${dset}_hires \
      $dir/graph data/${dset}_hires ${dir}/decode_${dset} || exit 1
    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang data/lang_rescore \
       data/${dset}_hires ${dir}/decode_${dset} ${dir}/decode_${dset}_rescore || exit 1
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi


exit 0;
