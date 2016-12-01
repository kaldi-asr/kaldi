#!/bin/bash

# This scripts does nnet3-tdnn training and decoding 
# (just like nnet3/run_tdnn.sh) except that it uses chain topology and
# a frame subsampling factor of 3

# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call train_tdnn.sh with --gpu false,
# --num-threads 16 and --minibatch-size 128.

stage=0
affix=
train_stage=-10
has_fisher=true
speed_perturb=true
common_egs_dir=
reporting_email=
remove_egs=true
leftmost_questions_truncate=-1

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

suffix=
if [ "$speed_perturb" == "true" ]; then
  suffix=_sp
fi
dir=exp/nnet3/reduced_tdnn
dir=$dir${affix:+_$affix}
dir=${dir}$suffix
train_set=train_nodup$suffix
ali_dir=exp/tri4_ali_nodup$suffix
treedir=exp/nnet3/reduced_tdnn_tree$suffix
lang=data/lang_reduced

local/nnet3/run_ivector_common.sh --stage $stage \
	--speed-perturb $speed_perturb || exit 1;


if [ $stage -le 9 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  rm -rf $lang
  cp -r data/lang $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi

if [ $stage -le 10 ]; then
  # Build a tree using our new topology. This is the critically different
  # step compared with other recipes.
  steps/nnet3/chain/build_tree.sh --repeat-frames true --frame-subsampling-factor 3 \
      --leftmost-questions-truncate $leftmost_questions_truncate \
      --context-opts "--context-width=2 --central-position=1" \
      --cmd "$train_cmd" 7000 data/$train_set $lang $ali_dir $treedir
fi

if [ $stage -le 11 ]; then
  echo "$0: creating neural net configs";

  # create the config files for nnet initialization.
  # Note: --relu-dim-init is the dim for the first relu layer, and --relu-dim is
  # the dim for the last relu layer, and the dims of the layers in between increase
  # geometrically. The reason for this kind of parameters re-distribution is mainly
  # for time efficiency, since the lower layers are more computationally intensive.
  steps/nnet3/tdnn/make_configs.py  \
    --feat-dir data/${train_set}_hires \
    --ivector-dir exp/nnet3/ivectors_${train_set} \
    --ali-dir $treedir \
    --relu-dim-init 512 \
    --relu-dim-final 1024 \
    --splice-indexes "-2,-1,0,1,2 -1,1 -1,1 -1,1 -2,2 -2,2 -2,2 -4,4 0" \
    --use-presoftmax-prior-scale true \
   $dir/configs || exit 1;
fi


if [ $stage -le 12 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/swbd-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/train_dnn.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.online-ivector-dir exp/nnet3/ivectors_${train_set} \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --trainer.num-epochs 2 \
    --trainer.optimization.num-jobs-initial 3 \
    --trainer.optimization.num-jobs-final 16 \
    --trainer.optimization.initial-effective-lrate 0.0017 \
    --trainer.optimization.final-effective-lrate 0.00017 \
    --egs.dir "$common_egs_dir" \
    --cleanup.remove-egs $remove_egs \
    --cleanup.preserve-model-interval 100 \
    --use-gpu true \
    --feat-dir=data/${train_set}_hires \
    --ali-dir $treedir \
    --lang data/lang \
    --reporting.email="$reporting_email" \
    --dir=$dir  || exit 1;

fi

graph_dir=$dir/graph_sw1_tg
if [ $stage -le 13 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --left-biphone --self-loop-scale 1.0 data/lang_sw1_tg $dir $graph_dir
fi

if [ $stage -le 14 ]; then
  for decode_set in train_dev eval2000; do
    (
    num_jobs=`cat data/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
    steps/nnet3/decode.sh --acwt 0.333 --post-decode-acwt 3.0 --nj $num_jobs --cmd "$decode_cmd" \
        --online-ivector-dir exp/nnet3/ivectors_${decode_set} \
       $graph_dir data/${decode_set}_hires $dir/decode_${decode_set}_hires_sw1_tg || exit 1;
    if $has_fisher; then
	steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
          data/lang_sw1_{tg,fsh_fg} data/${decode_set}_hires \
	  $dir/decode_${decode_set}_hires_sw1_{tg,fsh_fg} || exit 1;
    fi
    ) &
  done
fi
wait;
exit 0;

