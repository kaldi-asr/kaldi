set -e

# configs for 'chain'
# this script is based on swbd's 6z script
affix=
stage=16
train_stage=-10
get_egs_stage=-10
dir=exp/chain/tdnn_6z  # Note: _sp will get added to this, which means "speed perturb".
decode_iter=

# TDNN options
# this script uses the new tdnn config generator so it needs a final 0 to reflect that the final layer input has no splicing
# training options
frames_per_eg=150
relu_dim=725
remove_egs=false
common_egs_dir=
xent_regularize=0.1
self_repair_scale=0.00001
max_wer=


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

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 16" if you have already
# run those things.

suffix=_sp
dir=${dir}${affix:+_$affix}
dir=${dir}$suffix
treedir=exp/chain/tri5_2y_tree$suffix
lang=data/lang_chain_2y
max_wer_opt=${max_wer:+" --max-wer $max_wer "}

local/chain/run_chain_common.sh --stage $stage \
                                --frames-per-eg $frames_per_eg \
                                $max_wer_opt \
                                --dir $dir \
                                --treedir $treedir \
                                --lang $lang || exit 1;

. $dir/vars
# sets the directory names where features, ivectors and lattices are stored
#train_data_dir
#train_ivector_dir
#lat_dir

################################### 

if [ $stage -le 16 ]; then
  echo "$0: creating neural net configs";
  # create the config files for nnet initialization
  repair_opts=${self_repair_scale:+" --self-repair-scale $self_repair_scale "}

  steps/nnet3/tdnn/make_configs.py $repair_opts \
    --feat-dir $train_data_dir \
    --ivector-dir $train_ivector_dir \
    --tree-dir $treedir \
    --relu-dim $relu_dim \
    --splice-indexes "-1,0,1 -1,0,1,2 -3,0,3 -3,0,3 -3,0,3 -6,-3,0 0" \
    --use-presoftmax-prior-scale false \
    --xent-regularize $xent_regularize \
    --xent-separate-forward-affine true \
    --include-log-softmax false \
    --final-layer-normalize-target 0.5 \
    $dir/configs || exit 1;
fi



if [ $stage -le 17 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/librispeech-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  fi

 touch $dir/egs/.nodelete # keep egs around when that run dies.

 steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir $train_ivector_dir \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width $frames_per_eg \
    --egs.dir "$common_egs_dir" \
    --trainer.num-chunk-per-minibatch 128 \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs 4 \
    --trainer.optimization.num-jobs-initial 3 \
    --trainer.optimization.num-jobs-final 16 \
    --trainer.optimization.initial-effective-lrate 0.001 \
    --trainer.optimization.final-effective-lrate 0.0001 \
    --trainer.max-param-change 2 \
    --cleanup.remove-egs $remove_egs \
    --feat-dir $train_data_dir \
    --tree-dir $treedir \
    --lat-dir $lat_dir \
    --dir $dir  || exit 1;

fi

if [ $stage -le 18 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 --remove-oov data/lang_test_tgsmall $dir $dir/graph_test_tgsmall
  # romove <UNK> from the graph
  fstrmsymbols --apply-to-output=true --remove-arcs=true "echo 3|" $dir/graph_test_tgsmall/HCLG.fst $dir/graph_test_tgsmall/HCLG.fst
fi

graph_dir=$dir/graph_test_tgsmall
if [ $stage -le 19 ]; then
  iter_opts=
  if [ ! -z $decode_iter ]; then
    iter_opts=" --iter $decode_iter "
  fi
  for decode_set in test_clean test_other dev_clean dev_other; do
      (
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj 50 --cmd "$decode_cmd" $iter_opts \
          --online-ivector-dir exp/nnet3/ivectors_${decode_set} \
          $graph_dir data/${decode_set}_hires $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_tgsmall || touch $dir/.error
      steps/lmrescore.sh --cmd "$decode_cmd" --self-loop-scale 1.0 data/lang_test_{tgsmall,tgmed} \
          data/${decode_set}_hires $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_{tgsmall,tgmed} || touch $dir/.error
      steps/lmrescore_const_arpa.sh \
          --cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
          data/${decode_set}_hires $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_{tgsmall,tglarge} || touch $dir/.error
      steps/lmrescore_const_arpa.sh \
          --cmd "$decode_cmd" data/lang_test_{tgsmall,fglarge} \
          data/${decode_set}_hires $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_{tgsmall,fglarge} || touch $dir/.error
      ) &
  done
fi
wait;
exit 0;
