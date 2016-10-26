#!/bin/bash

set -e

# configs for 'chain'
affix=
stage=1
train_stage=-10
get_egs_stage=-10
speed_perturb=true
dir=exp/chain/tdnn_7b  # Note: _sp will get added to this if $speed_perturb == true.
decode_iter=
iv_dir=exp/nnet3_rvb
num_data_reps=1        # number of reverberated copies of data to generate
clean_train_set=train_nodup

# TDNN options
# this script uses the new tdnn config generator so it needs a final 0 to reflect that the final layer input has no splicing
# smoothing options
pool_window=
pool_type='none'
pool_lpfilter_width=
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
remove_egs=true
common_egs_dir=
xent_regularize=0.1



# End configuration section.
echo "$0 $@"  # Print the command line for logging

. cmd.sh
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


if [ "$speed_perturb" == "true" ]; then
  suffix=_sp
fi

dir=${dir}${affix:+_$affix}${suffix}_rvb${num_data_reps}
train_set=${clean_train_set}${suffix}_rvb${num_data_reps}
lang=data/lang_chain_2y
treedir=exp/chain/tri5_2y_tree${suffix}
lat_dir=exp/tri4_lats_nodup${suffix}
rvb_lat_dir=${lat_dir}_rvb${num_data_reps}


# if we are using the speed-perturbed data we need to generate
# alignments for it.
# Also the data reverberation will be done in this script/
echo local/nnet3/multi_condition/run_ivector_common.sh --stage $stage \
  --clean-data-dir ${clean_train_set} \
  --iv-dir $iv_dir \
  --speed-perturb $speed_perturb \
  --num-data-reps $num_data_reps || exit 1;


if [ $stage -le 9 ]; then
  # Get the alignments as lattices (gives the CTC training more freedom).
  # use the same num-jobs as the alignments
  nj=$(cat exp/tri4_ali_nodup${suffix}/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" data/${clean_train_set}${suffix} \
    data/lang exp/tri4 $lat_dir
  rm $lat_dir/fsts.*.gz # save space


  # Create the lattices for the reverberated data
  mkdir -p $rvb_lat_dir/temp/
  lattice-copy "ark:gunzip -c $lat_dir/lat.*.gz |" ark,scp:$rvb_lat_dir/temp/lats.ark,$rvb_lat_dir/temp/lats.scp

  # copy the lattices for the reverberated data
  rm -f $rvb_lat_dir/temp/combined_lats.scp
  touch $rvb_lat_dir/temp/combined_lats.scp
  # Here prefix "rev0_" represents the clean set, "rev1_" represents the reverberated set
  for i in `seq 0 $num_data_reps`; do
    cat $rvb_lat_dir/temp/lats.scp | sed -e "s/^/rev${i}_/" >> $rvb_lat_dir/temp/combined_lats.scp
  done
  sort -u $rvb_lat_dir/temp/combined_lats.scp > $rvb_lat_dir/temp/combined_lats_sorted.scp

  lattice-copy scp:$rvb_lat_dir/temp/combined_lats_sorted.scp "ark:|gzip -c >$rvb_lat_dir/lat.1.gz" || exit 1;
  echo "1" > $rvb_lat_dir/num_jobs

  # copy other files from original lattice dir
  for f in cmvn_opts final.mdl splice_opts tree; do
    cp $lat_dir/$f $rvb_lat_dir/$f
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
  # Build a tree using our new topology.
  # we build the tree using clean features (data/train) rather than
  # the augmented features (data/train_rvb) to get better alignments

  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --leftmost-questions-truncate $leftmost_questions_truncate \
      --cmd "$train_cmd" 9000 data/train_nodup${suffix} $lang exp/tri4_ali_nodup${suffix} $treedir
fi

if [ $stage -le 12 ]; then
  echo "$0: creating neural net configs";
  if [ ! -z "$relu_dim" ]; then
    dim_opts="--relu-dim $relu_dim"
  else
    dim_opts="--pnorm-input-dim $pnorm_input_dim --pnorm-output-dim  $pnorm_output_dim"
  fi

  # create the config files for nnet initialization
  pool_opts=
  pool_opts=$pool_opts${pool_type:+" --pool-type $pool_type "}
  pool_opts=$pool_opts${pool_window:+" --pool-window $pool_window "}
  pool_opts=$pool_opts${pool_lpfilter_width:+" --pool-lpfilter-width $pool_lpfilter_width "}
  repair_opts=${self_repair_scale:+" --self-repair-scale $self_repair_scale "}

  steps/nnet3/tdnn/make_configs.py $pool_opts \
    $repair_opts \
    --feat-dir data/${train_set}_hires \
    --ivector-dir $iv_dir/ivectors_${train_set} \
    --tree-dir $treedir \
    $dim_opts \
    --splice-indexes "-1,0,1 -1,0,1,2 -3,0,3 -3,0,3 -3,0,3 -6,-3,0 0" \
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
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/swbd-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  fi

 touch $dir/egs/.nodelete # keep egs around when that run dies.

 steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir $iv_dir/ivectors_${train_set} \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
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
    --lat-dir $rvb_lat_dir \
    --dir $dir  || exit 1;
fi

if [ $stage -le 13 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_sw1_tg $dir $dir/graph_sw1_tg
fi

decode_suff=sw1_tg
graph_dir=$dir/graph_sw1_tg
if [ $stage -le 14 ]; then
  iter_opts=
  if [ ! -z $decode_iter ]; then
    iter_opts=" --iter $decode_iter "
  fi
  for decode_set in eval2000; do
      (
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj 30 --cmd "$decode_cmd" $iter_opts \
          --online-ivector-dir $iv_dir/ivectors_${decode_set} \
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
