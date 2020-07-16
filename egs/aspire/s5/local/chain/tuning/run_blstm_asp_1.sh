#!/usr/bin/env bash

set -e

# based on run_blstm_7b.sh
# but has lesser number of parameters (20 million)

# configs for 'chain'
affix=
stage=11 # assuming you already ran the xent systems and the chain tdnn system
train_stage=-10
get_egs_stage=-10
dir=exp/chain/blstm_asp1
decode_iter=

# training options
num_epochs=4
remove_egs=false
common_egs_dir=exp/chain/blstm_7b/egs
num_data_reps=3

min_seg_len=
frames_per_eg=150
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

dir=${dir}${affix:+_$affix}
ali_dir=exp/tri5a_rvb_ali
treedir=exp/chain/tri6_tree_11000
lang=data/lang_chain


# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 8" if you have already
# run those things.
local/nnet3/run_ivector_common.sh --stage $stage --num-data-reps 3|| exit 1;

if [ $stage -le 7 ]; then
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

if [ $stage -le 8 ]; then
  # Build a tree using our new topology.
  # we build the tree using clean features (data/train) rather than
  # the augmented features (data/train_rvb) to get better alignments

  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --leftmost-questions-truncate -1 \
      --cmd "$train_cmd" 11000 data/train $lang exp/tri5a $treedir
fi

if [ -z $min_seg_len ]; then
  min_seg_len=$(python -c "print ($frames_per_eg+5)/100.0")
fi

if [ $stage -le 9 ]; then
  [ -d data/train_rvb_min${min_seg_len}_hires ] && rm -rf data/train_rvb_min${min_seg_len}_hires
  steps/cleanup/combine_short_segments.py --minimum-duration $min_seg_len \
    --input-data-dir data/train_rvb_hires \
    --output-data-dir data/train_rvb_min${min_seg_len}_hires

  #extract ivectors for the new data
  steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 \
    data/train_rvb_min${min_seg_len}_hires data/train_rvb_min${min_seg_len}_hires_max2
  ivectordir=exp/nnet3/ivectors_train_min${min_seg_len}
  if [[ $(hostname -f) == *.clsp.jhu.edu ]]; then # this shows how you can split across multiple file-systems.
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/aspire/s5/$ivectordir/storage $ivectordir/storage
  fi

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 200 \
    data/train_rvb_min${min_seg_len}_hires_max2 \
    exp/nnet3/extractor $ivectordir || exit 1;

 # combine the non-hires features for alignments/lattices
  [ -d data/train_min${min_seg_len} ] && rm -r data/train_min${min_seg_len};
  utt_prefix="THISISUNIQUESTRING_"
  spk_prefix="THISISUNIQUESTRING_"
  utils/copy_data_dir.sh --spk-prefix "$spk_prefix" --utt-prefix "$utt_prefix" \
    data/train data/train_temp_for_lats
  steps/cleanup/combine_short_segments.py --minimum-duration $min_seg_len \
                   --input-data-dir data/train_temp_for_lats \
                   --output-data-dir data/train_min${min_seg_len}
fi

if [ $stage -le 10 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  nj=200
  lat_dir=exp/tri5a_min${min_seg_len}_lats
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" data/train_min${min_seg_len} \
    data/lang exp/tri5a $lat_dir
  rm -f $lat_dir/fsts.*.gz # save space

  rvb_lat_dir=exp/tri5a_rvb_min${min_seg_len}_lats
  mkdir -p $rvb_lat_dir/temp/
  lattice-copy "ark:gunzip -c $lat_dir/lat.*.gz |" ark,scp:$rvb_lat_dir/temp/lats.ark,$rvb_lat_dir/temp/lats.scp

  # copy the lattices for the reverberated data
  rm -f $rvb_lat_dir/temp/combined_lats.scp
  touch $rvb_lat_dir/temp/combined_lats.scp
  for i in `seq 1 $num_data_reps`; do
    cat $rvb_lat_dir/temp/lats.scp | sed -e "s/THISISUNIQUESTRING/rev${i}/g" >> $rvb_lat_dir/temp/combined_lats.scp
  done
  sort -u $rvb_lat_dir/temp/combined_lats.scp > $rvb_lat_dir/temp/combined_lats_sorted.scp

  lattice-copy scp:$rvb_lat_dir/temp/combined_lats_sorted.scp "ark:|gzip -c >$rvb_lat_dir/lat.1.gz" || exit 1;
  echo "1" > $rvb_lat_dir/num_jobs

  # copy other files from original lattice dir
  for f in cmvn_opts final.mdl splice_opts tree; do
    cp $lat_dir/$f $rvb_lat_dir/$f
  done
fi

if [ $stage -le 11 ]; then
  echo "$0: creating neural net configs";

  steps/nnet3/lstm/make_configs.py  \
    --feat-dir data/train_rvb_hires \
    --ivector-dir exp/nnet3/ivectors_train_min${min_seg_len} \
    --tree-dir $treedir \
    --splice-indexes="-2,-1,0,1,2 0 0" \
    --lstm-delay=" [-3,3] [-3,3] [-3,3] " \
    --xent-regularize 0.1 \
    --include-log-softmax false \
    --num-lstm-layers 3 \
    --cell-dim 1024 \
    --hidden-dim 1024 \
    --recurrent-projection-dim 128 \
    --non-recurrent-projection-dim 128 \
    --label-delay 0 \
    --self-repair-scale-nonlinearity 0.00001 \
    --self-repair-scale-clipgradient 1.0 \
   $dir/configs || exit 1;

fi

if [ $stage -le 12 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/aspire-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  fi

  touch $dir/egs/.nodelete # keep egs around when that run dies.

  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir exp/nnet3/ivectors_train_min${min_seg_len} \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize 0.1 \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.num-chunk-per-minibatch 64 \
    --trainer.max-param-change 1.414 \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width $frames_per_eg \
    --egs.dir "$common_egs_dir" \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.num-jobs-initial 3 \
    --trainer.optimization.num-jobs-final 16 \
    --trainer.optimization.initial-effective-lrate 0.001 \
    --trainer.optimization.final-effective-lrate 0.0001 \
    --trainer.optimization.shrink-value 0.99 \
    --trainer.optimization.momentum 0.0 \
    --trainer.deriv-truncate-margin 8 \
    --cleanup.remove-egs $remove_egs \
    --feat-dir data/train_rvb_min${min_seg_len}_hires \
    --tree-dir $treedir \
    --lat-dir exp/tri5a_rvb_min${min_seg_len}_lats \
    --dir $dir  || exit 1;
fi

if [ $stage -le 13 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_pp_test $dir $dir/graph_pp
fi


if [ $stage -le 14 ]; then
  extra_left_context=$[$chunk_left_context+10]
  extra_right_context=$[$chunk_right_context+10]
  # %WER 26.8 | 2120 27220 | 80.2 11.7 8.1 7.0 26.8 76.5 | -0.804 | exp/chain/blstm_asp_1/decode_dev_aspire_whole_uniformsegmented_win10_over5_v7_iterfinal_pp_fg/score_9/penalty_0.0/

  local/multi_condition/prep_test_aspire.sh --stage 4 --decode-num-jobs 30  --affix "v7" \
   --extra-left-context $extra_left_context \
   --extra-right-context $extra_right_context \
   --frames-per-chunk $chunk_width \
   --acwt 1.0 --post-decode-acwt 10.0 \
   --window 10 --overlap 5 \
   --sub-speaker-frames 6000 --max-count 75 --ivector-scale 0.75 \
   --pass2-decode-opts "--min-active 1000" \
   dev_aspire data/lang $dir/graph_pp $dir
fi
exit 0;


#%WER 29.7 | 2120 27214 | 78.0 14.3 7.6 7.7 29.7 78.2 | -0.765 | exp/chain/blstm_asp_1/decode_dev_aspire_whole_uniformsegmented_win10_over5_v7_iter300_pp_fg/score_9/penalty_0.0/ctm.filt.filt.sys
#%WER 29.0 | 2120 27214 | 78.0 13.1 9.0 7.0 29.0 78.0 | -0.778 | exp/chain/blstm_asp_1/decode_dev_aspire_whole_uniformsegmented_win10_over5_v7_iter400_pp_fg/score_9/penalty_0.25/ctm.filt.filt.sys
#%WER 28.0 | 2120 27217 | 78.9 12.8 8.2 6.9 28.0 77.7 | -0.819 | exp/chain/blstm_asp_1/decode_dev_aspire_whole_uniformsegmented_win10_over5_v7_iter500_pp_fg/score_9/penalty_0.25/ctm.filt.filt.sys
#%WER 28.1 | 2120 27214 | 79.0 12.8 8.2 7.2 28.1 77.2 | -0.807 | exp/chain/blstm_asp_1/decode_dev_aspire_whole_uniformsegmented_win10_over5_v7_iter600_pp_fg/score_9/penalty_0.25/ctm.filt.filt.sys
#%WER 27.1 | 2120 27218 | 79.8 12.2 7.9 7.0 27.1 76.4 | -0.814 | exp/chain/blstm_asp_1/decode_dev_aspire_whole_uniformsegmented_win10_over5_v7_iter800_pp_fg/score_9/penalty_0.25/ctm.filt.filt.sys
#%WER 26.5 | 2120 27220 | 80.6 12.3 7.0 7.1 26.5 76.1 | -0.804 | exp/chain/blstm_asp_1/decode_dev_aspire_whole_uniformsegmented_win10_over5_v7_iter1300_pp_fg/score_9/penalty_0.0/ctm.filt.filt.sys
