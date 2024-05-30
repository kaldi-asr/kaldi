#!/usr/bin/env bash

# 7a inherits from 6z (which is a TDNN+ReLU-based network with various small
# bugs hopefully fixed now), and from 6r, which is our most-successful
# double-frame-rate system.  We're re-dumping the egs, because the egs used in
# 6r used right-tolerance=10, which turns out to have been a bug, and not a
# helpful one.

# it is not better than 6z.
# local/chain/compare_wer.sh 6v 6z 7a
#System                       6v        6z        7a
#WER on train_dev(tg)      15.00     15.18     15.05
#WER on train_dev(fg)      13.91     14.06     14.10
#WER on eval2000(tg)        17.2      17.2      17.3
#WER on eval2000(fg)        15.7      15.6      15.7
#Final train prob      -0.105012 -0.106268 -0.110288
#Final valid prob      -0.125877 -0.126726 -0.127071
#Final train prob (xent)      -1.54736   -1.4556  -1.59569
#Final valid prob (xent)      -1.57475  -1.50136  -1.62312

# 6z is as 6y, but fixing the right-tolerance in the scripts to default to 5 (as
# the default is in the code), rather than the previous script default value of
# 10 which I seem to have added to the script around Feb 9th.

# 6y is as 6w, but after fixing the config-generation script to use
# a higher learning-rate factor for the final xent layer (it was otherwise
# training too slowly).

# 6w is as 6v (a new tdnn-based recipe), but using 1.5 million not 1.2 million
# frames per iter (and of course re-dumping the egs).

# this is same as v2 script but with xent-regularization
# it has a different splicing configuration
set -e

# configs for 'chain'
affix=
stage=14
train_stage=-10
get_egs_stage=-10
speed_perturb=true
dir=exp/chain/tdnn_7a  # Note: _sp will get added to this if $speed_perturb == true.
decode_iter=

# TDNN options
# this script uses the new tdnn config generator so it needs a final 0 to reflect that the final layer input has no splicing
# smoothing options
self_repair_scale=0.00001
# training options
num_epochs=2 # use 2 not 4 epochs, as with the double-frame-rate input, we
             # shift the input data in double the number of distinct ways
             # on each epoch.
initial_effective_lrate=0.001
final_effective_lrate=0.0001
leftmost_questions_truncate=-1
max_param_change=2.0
final_layer_normalize_target=0.5
num_jobs_initial=3
num_jobs_final=16
minibatch_size=128
relu_dim=576
remove_egs=false
common_egs_dir=
xent_regularize=0.1



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
# nnet3 setup, and you can skip them by setting "--stage 8" if you have already
# run those things.

suffix=
if [ "$speed_perturb" == "true" ]; then
  suffix=_sp
fi

dir=${dir}${affix:+_$affix}$suffix
train_set=train_nodup$suffix
ali_dir=exp/tri4_ali_nodup$suffix
treedir=exp/chain/tri5_2y_tree$suffix
lang=data/lang_chain_2y


# if we are using the speed-perturbed data we need to generate
# alignments for it.
local/nnet3/run_ivector_common.sh --stage $stage \
  --speed-perturb $speed_perturb \
  --generate-alignments $speed_perturb || exit 1;


if [ $stage -le 9 ]; then
  # Get the alignments as lattices (gives the CTC training more freedom).
  # use the same num-jobs as the alignments
  nj=$(cat exp/tri4_ali_nodup$suffix/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" data/$train_set \
    data/lang exp/tri4 exp/tri4_lats_nodup$suffix
  rm exp/tri4_lats_nodup$suffix/fsts.*.gz # save space
fi


if [ $stage -le 10 ]; then
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

if [ $stage -le 11 ]; then
  # Build a tree using our new topology.
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --leftmost-questions-truncate $leftmost_questions_truncate \
      --cmd "$train_cmd" 9000 data/$train_set $lang $ali_dir $treedir
fi

# Generate double-frame-rate version of the data.
if [ $stage -le 12 ]; then
  mfccdir=mfcc
  for dataset in eval2000 train_dev; do  ## ${train_set}; do
    utils/copy_data_dir.sh data/$dataset data/${dataset}_hires_dbl
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 30 --mfcc-config conf/mfcc_hires_dbl.conf \
        data/${dataset}_hires_dbl exp/make_hires_dbl/$dataset $mfccdir;
    steps/compute_cmvn_stats.sh data/${dataset}_hires_dbl exp/make_hires_dbl/$dataset $mfccdir;
    utils/fix_data_dir.sh data/${dataset}_hires_dbl  # remove segments with problems
  done
fi

if [ $stage -le 13 ]; then
  for dataset in eval2000 train_dev ${train_set}; do
    mkdir -p exp/nnet3/ivectors_${dataset}_fake2
    cp exp/nnet3/ivectors_${dataset}/ivector_online.scp exp/nnet3/ivectors_${dataset}_fake2
    # verify that the old ivector_period was 10.
    [ $(cat exp/nnet3/ivectors_${dataset}/ivector_period) -eq 10 ] || exit 1
    echo 20 > exp/nnet3/ivectors_${dataset}_fake2/ivector_period
  done
fi

if [ $stage -le 14 ]; then
  echo "$0: creating neural net configs";
  if [ ! -z "$relu_dim" ]; then
    dim_opts="--relu-dim $relu_dim"
  else
    dim_opts="--pnorm-input-dim $pnorm_input_dim --pnorm-output-dim  $pnorm_output_dim"
  fi

  # create the config files for nnet initialization
  repair_opts=${self_repair_scale:+" --self-repair-scale-nonlinearity $self_repair_scale "}

  steps/nnet3/tdnn/make_configs.py \
    $repair_opts \
    --feat-dir data/${train_set}_hires_dbl \
    --ivector-dir exp/nnet3/ivectors_${train_set}_fake2 \
    --tree-dir $treedir \
    $dim_opts \
    --splice-indexes "-1,0,1 -2,0,2 -4,-2,0,2 -6,0,6 -6,0,6 -12,-6,0 0" \
    --use-presoftmax-prior-scale false \
    --xent-regularize $xent_regularize \
    --xent-separate-forward-affine true \
    --include-log-softmax false \
    --final-layer-normalize-target $final_layer_normalize_target \
    $dir/configs || exit 1;
fi



if [ $stage -le 15 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{7,11,12,13}/$USER/kaldi-data/egs/swbd-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  fi

 touch $dir/egs/.nodelete # keep egs around when that run dies.

 steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir exp/nnet3/ivectors_${train_set}_fake2 \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.frame-subsampling-factor 6 \
    --chain.alignment-subsampling-factor 3 \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width 300 \
    --trainer.num-chunk-per-minibatch $minibatch_size \
    --trainer.frames-per-iter 3000000 \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.num-jobs-initial $num_jobs_initial \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.initial-effective-lrate $initial_effective_lrate \
    --trainer.optimization.final-effective-lrate $final_effective_lrate \
    --trainer.max-param-change $max_param_change \
    --cleanup.remove-egs $remove_egs \
    --feat-dir data/${train_set}_hires_dbl \
    --tree-dir $treedir \
    --lat-dir exp/tri4_lats_nodup$suffix \
    --dir $dir  || exit 1;

 echo "0.005" > $dir/frame_shift # this lets the sclite decoding script know
                                 # what the frame shift was, in seconds.
fi

if [ $stage -le 16 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_sw1_tg $dir $dir/graph_sw1_tg
fi

decode_suff=sw1_tg
graph_dir=$dir/graph_sw1_tg
if [ $stage -le 17 ]; then
  iter_opts=
  if [ ! -z $decode_iter ]; then
    iter_opts=" --iter $decode_iter "
  fi
  for decode_set in train_dev eval2000; do
      (
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj 50 --cmd "$decode_cmd" $iter_opts \
          --online-ivector-dir exp/nnet3/ivectors_${decode_set}_fake2 \
          $graph_dir data/${decode_set}_hires_dbl $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_${decode_suff} || exit 1;
      if $has_fisher; then
          steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
            data/lang_sw1_{tg,fsh_fg} data/${decode_set}_hires_dbl \
            $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_sw1_{tg,fsh_fg} || exit 1;
      fi
      ) &
  done
fi
wait;
exit 0;
