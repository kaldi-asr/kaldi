#!/bin/bash

# This script (multi_condition/run_tdnn_7f.sh) is the reverberated version of
# tuning/run_tdnn_7f.sh. It reverberates the training data with room impulse responses
# which leads to better results.
# (The reverberation of data is done in multi_condition/run_ivector_common.sh)
# This script assumes a mixing of the original training data with its reverberated copy
# and results in a 2-fold training set. Thus the number of epochs is halved to
# keep the same training time. The model converges after 2 epochs of training,
# The WER doesn't change much with more epochs of training.
# local/chain/compare_wer.sh tuning/7f multi_condition/7f
# System                 tuning/7f  multi_condition/7f
# WER on train_dev(tg)      14.46     14.27
# WER on train_dev(fg)      13.23     13.16
# WER on eval2000(tg)        17.0      16.3
# WER on eval2000(fg)        15.4      14.6
# Final train prob     -0.0882071 -0.123325
# Final valid prob      -0.107545 -0.131798
# Final train prob (xent)      -1.26246   -1.6196
# Final valid prob (xent)      -1.35525  -1.60244


set -e

# configs for 'chain'
affix=
stage=1
train_stage=-10
get_egs_stage=-10
speed_perturb=true
dir=exp/chain/tdnn_7f  # Note: _sp will get added to this if $speed_perturb == true.
decode_iter=
ivector_dir=exp/nnet3_rvb
num_data_reps=1        # number of reverberated copies of data to generate
input_train_set=train_nodup


# TDNN options
# this script uses the new tdnn config generator so it needs a final 0 to reflect that the final layer input has no splicing
splice_indexes="-1,0,1 -1,0,1 -1,0,1 -3,0,3 -3,0,3 -6,0,6 0"
# smoothing options
self_repair_scale=0.00001
# training options
num_epochs=2
initial_effective_lrate=0.001
final_effective_lrate=0.0001
leftmost_questions_truncate=-1
max_param_change=2.0
final_layer_normalize_target=0.5
num_jobs_initial=3
num_jobs_final=16
minibatch_size=128
relu_dim=625
frames_per_eg=150
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

dir=${dir}${affix:+_$affix}${suffix}_rvb${num_data_reps}
clean_train_set=${input_train_set}${suffix}
train_set=${clean_train_set}_rvb${num_data_reps}
ali_dir=exp/tri4_ali_nodup$suffix
treedir=exp/chain/tri5_7d_tree$suffix
lang=data/lang_chain_2y
clean_lat_dir=exp/tri4_lats_nodup${suffix}
lat_dir=${clean_lat_dir}_rvb${num_data_reps}


# The data reverberation will be done in this script.
local/nnet3/multi_condition/run_ivector_common.sh --stage $stage \
  --input-data-dir ${input_train_set} \
  --ivector-dir $ivector_dir \
  --speed-perturb $speed_perturb \
  --num-data-reps $num_data_reps || exit 1;


if [ $stage -le 9 ]; then
  # Get the alignments as lattices (gives the LF-MMI training more freedom).
  # use the same num-jobs as the alignments
  nj=$(cat exp/tri4_ali_nodup${suffix}/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" data/${clean_train_set} \
    data/lang exp/tri4 $clean_lat_dir
  rm $clean_lat_dir/fsts.*.gz # save space


  # Create the lattices for the reverberated data
  # We use the lattices/alignments from the clean data for the reverberated data.
  mkdir -p $lat_dir/temp/
  lattice-copy "ark:gunzip -c $clean_lat_dir/lat.*.gz |" ark,scp:$lat_dir/temp/lats.ark,$lat_dir/temp/lats.scp

  # copy the lattices for the reverberated data
  rm -f $lat_dir/temp/combined_lats.scp
  touch $lat_dir/temp/combined_lats.scp
  # Here prefix "rev0_" represents the clean set, "rev1_" represents the reverberated set
  for i in `seq 0 $num_data_reps`; do
    cat $lat_dir/temp/lats.scp | sed -e "s/^/rev${i}_/" >> $lat_dir/temp/combined_lats.scp
  done
  sort -u $lat_dir/temp/combined_lats.scp > $lat_dir/temp/combined_lats_sorted.scp

  lattice-copy scp:$lat_dir/temp/combined_lats_sorted.scp "ark:|gzip -c >$lat_dir/lat.1.gz" || exit 1;
  echo "1" > $lat_dir/num_jobs

  # copy other files from original lattice dir
  for f in cmvn_opts final.mdl splice_opts tree; do
    cp $clean_lat_dir/$f $lat_dir/$f
  done
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
  # Build a tree using our new topology. This is the critically different
  # step compared with other recipes.
  # we build the tree using the clean alignments as we empirically found that this was better.
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --leftmost-questions-truncate $leftmost_questions_truncate \
      --context-opts "--context-width=2 --central-position=1" \
      --cmd "$train_cmd" 7000 data/${clean_train_set} $lang $ali_dir $treedir
fi

if [ $stage -le 12 ]; then
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
    --feat-dir data/${train_set}_hires \
    --ivector-dir $ivector_dir/ivectors_${train_set} \
    --tree-dir $treedir \
    $dim_opts \
    --splice-indexes "$splice_indexes" \
    --use-presoftmax-prior-scale false \
    --xent-regularize $xent_regularize \
    --xent-separate-forward-affine true \
    --include-log-softmax false \
    --final-layer-normalize-target $final_layer_normalize_target \
    $dir/configs || exit 1;
fi



if [ $stage -le 13 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/swbd-reverb-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir $ivector_dir/ivectors_${train_set} \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width $frames_per_eg \
    --trainer.num-chunk-per-minibatch $minibatch_size \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.num-jobs-initial $num_jobs_initial \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.initial-effective-lrate $initial_effective_lrate \
    --trainer.optimization.final-effective-lrate $final_effective_lrate \
    --trainer.max-param-change $max_param_change \
    --cleanup.remove-egs $remove_egs \
    --feat-dir data/${train_set}_hires \
    --tree-dir $treedir \
    --lat-dir $lat_dir \
    --dir $dir  || exit 1;

fi

if [ $stage -le 14 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_sw1_tg $dir $dir/graph_sw1_tg
fi

decode_suff=sw1_tg
graph_dir=$dir/graph_sw1_tg
if [ $stage -le 15 ]; then
  iter_opts=
  if [ ! -z $decode_iter ]; then
    iter_opts=" --iter $decode_iter "
  fi
  for decode_set in train_dev eval2000; do
      (
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj 50 --cmd "$decode_cmd" $iter_opts \
          --online-ivector-dir $ivector_dir/ivectors_${decode_set} \
          $graph_dir data/${decode_set}_hires $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_${decode_suff} || exit 1;
      if $has_fisher; then
          steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
            data/lang_sw1_{tg,fsh_fg} data/${decode_set}_hires \
            $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_sw1_{tg,fsh_fg} || exit 1;
      fi
      ) &
  done
fi
wait;
exit 0;
