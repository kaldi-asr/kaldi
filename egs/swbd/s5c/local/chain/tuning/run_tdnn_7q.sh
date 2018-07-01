#!/bin/bash

# 7q is as 7p but a modified topology with resnet-style skip connections, more layers,
#  skinnier bottlenecks, removing the 3-way splicing and skip-layer splicing,
#  and re-tuning the learning rate and l2 regularize.  The configs are
#  standardized and substantially simplified.  There isn't any advantage in WER
#  on this setup; the advantage of this style of config is that it also works
#  well on smaller datasets, and we adopt this style here also for consistency.

# local/chain/compare_wer_general.sh --rt03 tdnn7p_sp tdnn7q_sp
# System                tdnn7p_sp tdnn7q_sp [rerun:tdnn7q_sp]
# WER on train_dev(tg)      11.80    11.77    11.93
# WER on train_dev(fg)      10.77     10.85    10.95
# WER on eval2000(tg)        14.4     14.5    14.5
# WER on eval2000(fg)        13.0     13.0    13.1
# WER on rt03(tg)            17.5     17.5    17.4
# WER on rt03(fg)            15.3     15.3    15.2
# Final train prob         -0.057     -0.057   -0.058
# Final valid prob         -0.069      -0.071   -0.072
# Final train prob (xent)        -0.886    -0.886   -0.885
# Final valid prob (xent)       -0.9005    -0.8979   -0.8977
# Num-parameters               22865188  21356836  21356836

# 7p10q is as 7p10m but with larger frames-per-iter (1.5->5million), a bit more
#  than double the l2-regularize on non-final layers, and a bit less than half
#  the final learning rate.  (A change similar to this worked well on tedlium).
#
# 7p10m is as 7p10l but making several changes (I'll start tuning and experimenting again,
#  as it's too many changes to know what made the difference).
#  - Reducing num-chunk-per-minibatch from 128 to 64.
#  - Reducing num-epochs from 8 to 6.
#  - Using tdnnf6-layer, which lacks the internal splicing.
#  - Changing the extra time-stride=1 layer introduced in 7p10l to time-stride=3.
# It's a bit worse; seems to be underfitting.
# local/chain/compare_wer_general.sh --rt03 tdnn7p_sp tdnn7p10l_sp tdnn7p10m_sp
# System                tdnn7p_sp tdnn7p10l_sp tdnn7p10m_sp
# WER on train_dev(tg)      11.80     11.73     11.73
# WER on train_dev(fg)      10.77     10.79     10.85
# WER on eval2000(tg)        14.4      14.3      14.6
# WER on eval2000(fg)        13.0      13.1      13.2
# WER on rt03(tg)            17.5      17.5      17.7
# WER on rt03(fg)            15.3      15.2      15.5
# Final train prob         -0.057    -0.053    -0.056
# Final valid prob         -0.069    -0.066    -0.071
# Final train prob (xent)        -0.886    -0.822    -0.867
# Final valid prob (xent)       -0.9005   -0.8526   -0.8828
# Num-parameters               22865188  22315300  21356836

# 7p10l is as 7p10g but adding two more layers: one with stride 1, one with stride 3.
# 7p10g is as 7p10b but adding two more layers.
# It does seem to help-- and results now match 10p.
# local/chain/compare_wer_general.sh --rt03 tdnn7p_sp tdnn7p10b_sp tdnn7p10g_sp
# System                tdnn7p_sp tdnn7p10b_sp tdnn7p10g_sp
# WER on train_dev(tg)      11.80     11.82     11.80
# WER on train_dev(fg)      10.77     10.98     10.83
# WER on eval2000(tg)        14.4      14.4      14.4
# WER on eval2000(fg)        13.0      13.1      13.1
# WER on rt03(tg)            17.5      17.7      17.5
# WER on rt03(fg)            15.3      15.5      15.2
# Final train prob         -0.057    -0.056    -0.053
# Final valid prob         -0.069    -0.070    -0.067
# Final train prob (xent)        -0.886    -0.895    -0.843
# Final valid prob (xent)       -0.9005   -0.9311   -0.8815
# Num-parameters               22865188  17295652  19805476


# local/chain/compare_wer_general.sh --rt03 tdnn7p_sp tdnn7p10b_sp tdnn7p10g_sp
# System                tdnn7p_sp tdnn7p10b_sp tdnn7p10g_sp
# WER on train_dev(tg)      11.80     11.82     11.80
# WER on train_dev(fg)      10.77     10.98     10.83
# WER on eval2000(tg)        14.4      14.4      14.4
# WER on eval2000(fg)        13.0      13.1      13.1
# WER on rt03(tg)            17.5      17.7      17.5
# WER on rt03(fg)            15.3      15.5      15.2
# Final train prob         -0.057    -0.056    -0.053
# Final valid prob         -0.069    -0.070    -0.067
# Final train prob (xent)        -0.886    -0.895    -0.843
# Final valid prob (xent)       -0.9005   -0.9311   -0.8815
# Num-parameters               22865188  17295652  19805476

# 7p10b is as 7p10 but increasing the size: 1024->1536, 128->192.
#  Because I think it will exhaust memory, running only on bigger-memory machines.
# It's better than 7p10 but no better than the original 7p.
# local/chain/compare_wer_general.sh --rt03 tdnn7p_sp tdnn7p10_sp tdnn7p10b_sp
# System                tdnn7p_sp tdnn7p10_sp tdnn7p10b_sp
# WER on train_dev(tg)      11.80     12.17     11.82
# WER on train_dev(fg)      10.77     11.37     10.98
# WER on eval2000(tg)        14.4      15.1      14.4
# WER on eval2000(fg)        13.0      13.6      13.1
# WER on rt03(tg)            17.5      18.4      17.7
# WER on rt03(fg)            15.3      15.9      15.5
# Final train prob         -0.057    -0.062    -0.056
# Final valid prob         -0.069    -0.075    -0.070
# Final train prob (xent)        -0.886    -0.984    -0.895
# Final valid prob (xent)       -0.9005   -1.0033   -0.9311
# Num-parameters               22865188   9926436  17295652

# 7p10 is as 7p but like the run_tdnn_1f10r2 experiment in tedlium (naming may
# change)... it uses 'tdnnf-layer' and bottleneck dim of 128.

# 7p is as 7o but adding the option "--constrained false" to --egs.opts.
# This is the new 'unconstrained egs' code where it uses the e2e examples.
# This leads to ~40% speed-up in egs generation.
#
#
# local/chain/compare_wer_general.sh --rt03 tdnn7o_sp tdnn7p_sp
# System                tdnn7o_sp tdnn7p_sp
# WER on train_dev(tg)      11.74     11.75
# WER on train_dev(fg)      10.69     10.83
# WER on eval2000(tg)        14.6      14.1
# WER on eval2000(fg)        13.1      12.8
# WER on rt03(tg)            17.5      17.3
# WER on rt03(fg)            15.4      15.0
# Final train prob         -0.070    -0.055
# Final valid prob         -0.084    -0.069
# Final train prob (xent)        -0.883    -0.872
# Final valid prob (xent)       -0.9110   -0.9020
# Num-parameters               22865188  22886776

# steps/info/chain_dir_info.pl exp/chain/tdnn7q_sp
# exp/chain/tdnn7q_sp: num-iters=394 nj=3..16 num-params=21.4M dim=40+100->6034 combine=-0.057->-0.057 (over 8) xent:train/valid[261,393,final]=(-1.19,-0.886,-0.885/-1.22,-0.904,-0.898) logprob:train/valid[261,393,final]=(-0.089,-0.059,-0.058/-0.101,-0.072,-0.072)

set -e

# configs for 'chain'
stage=0
train_stage=-10
get_egs_stage=-10
speed_perturb=true
affix=7q
if [ -e data/rt03 ]; then maybe_rt03=rt03; else maybe_rt03= ; fi

decode_iter=
decode_nj=50

# training options
frames_per_eg=150,110,100
remove_egs=false
common_egs_dir=
xent_regularize=0.1
dropout_schedule='0,0@0.20,0.5@0.50,0'

test_online_decoding=false  # if true, it will run the last decoding stage.

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

suffix=
$speed_perturb && suffix=_sp
dir=exp/chain/tdnn${affix}${suffix}

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

train_set=train_nodup$suffix
ali_dir=exp/tri4_ali_nodup$suffix
treedir=exp/chain/tri5_7d_tree$suffix
lang=data/lang_chain_2y


# if we are using the speed-perturbed data we need to generate
# alignments for it.
local/nnet3/run_ivector_common.sh --stage $stage \
  --speed-perturb $speed_perturb \
  --generate-alignments $speed_perturb || exit 1;


if [ $stage -le 9 ]; then
  # Get the alignments as lattices (gives the LF-MMI training more freedom).
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
  # Build a tree using our new topology. This is the critically different
  # step compared with other recipes.
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --context-opts "--context-width=2 --central-position=1" \
      --cmd "$train_cmd" 7000 data/$train_set $lang $ali_dir $treedir
fi

if [ $stage -le 12 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)
  affine_opts="l2-regularize=0.01 dropout-proportion=0.0 dropout-per-dim=true dropout-per-dim-continuous=true"
  tdnnf_opts="l2-regularize=0.01 dropout-proportion=0.0 bypass-scale=0.66"
  linear_opts="l2-regularize=0.01 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.01"
  output_opts="l2-regularize=0.002"

  mkdir -p $dir/configs

  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-dropout-layer name=tdnn1 $affine_opts dim=1536
  tdnnf-layer name=tdnnf2 $tdnnf_opts dim=1536 bottleneck-dim=192 time-stride=1
  tdnnf-layer name=tdnnf3 $tdnnf_opts dim=1536 bottleneck-dim=192 time-stride=1
  tdnnf-layer name=tdnnf4 $tdnnf_opts dim=1536 bottleneck-dim=192 time-stride=1
  tdnnf-layer name=tdnnf5 $tdnnf_opts dim=1536 bottleneck-dim=192 time-stride=0
  tdnnf-layer name=tdnnf6 $tdnnf_opts dim=1536 bottleneck-dim=192 time-stride=3
  tdnnf-layer name=tdnnf7 $tdnnf_opts dim=1536 bottleneck-dim=192 time-stride=3
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=1536 bottleneck-dim=192 time-stride=3
  tdnnf-layer name=tdnnf9 $tdnnf_opts dim=1536 bottleneck-dim=192 time-stride=3
  tdnnf-layer name=tdnnf10 $tdnnf_opts dim=1536 bottleneck-dim=192 time-stride=3
  tdnnf-layer name=tdnnf11 $tdnnf_opts dim=1536 bottleneck-dim=192 time-stride=3
  tdnnf-layer name=tdnnf12 $tdnnf_opts dim=1536 bottleneck-dim=192 time-stride=3
  tdnnf-layer name=tdnnf13 $tdnnf_opts dim=1536 bottleneck-dim=192 time-stride=3
  tdnnf-layer name=tdnnf14 $tdnnf_opts dim=1536 bottleneck-dim=192 time-stride=3
  tdnnf-layer name=tdnnf15 $tdnnf_opts dim=1536 bottleneck-dim=192 time-stride=3
  linear-component name=prefinal-l dim=256 $linear_opts

  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts big-dim=1536 small-dim=256
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts big-dim=1536 small-dim=256
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 13 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/swbd-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  fi

#    --cmd "queue.pl --config /home/dpovey/queue_conly.conf" \


  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$train_cmd" \
    --feat.online-ivector-dir exp/nnet3/ivectors_${train_set} \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.0 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0 --constrained false" \
    --egs.chunk-width $frames_per_eg \
    --trainer.num-chunk-per-minibatch 64 \
    --trainer.frames-per-iter 5000000 \
    --trainer.num-epochs 6 \
    --trainer.optimization.num-jobs-initial 3 \
    --trainer.optimization.num-jobs-final 16 \
    --trainer.optimization.initial-effective-lrate 0.00025 \
    --trainer.optimization.final-effective-lrate 0.000025 \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs $remove_egs \
    --feat-dir data/${train_set}_hires \
    --tree-dir $treedir \
    --lat-dir exp/tri4_lats_nodup$suffix \
    --dir $dir  || exit 1;

fi

if [ $stage -le 14 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_sw1_tg $dir $dir/graph_sw1_tg
fi


graph_dir=$dir/graph_sw1_tg
iter_opts=
if [ ! -z $decode_iter ]; then
  iter_opts=" --iter $decode_iter "
fi
if [ $stage -le 15 ]; then
  rm $dir/.error 2>/dev/null || true
  for decode_set in train_dev eval2000 $maybe_rt03; do
      (
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $decode_nj --cmd "$decode_cmd" $iter_opts \
          --online-ivector-dir exp/nnet3/ivectors_${decode_set} \
          $graph_dir data/${decode_set}_hires \
          $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_sw1_tg || exit 1;
      if $has_fisher; then
          steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
            data/lang_sw1_{tg,fsh_fg} data/${decode_set}_hires \
            $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_sw1_{tg,fsh_fg} || exit 1;
      fi
      ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi

if $test_online_decoding && [ $stage -le 16 ]; then
  # note: if the features change (e.g. you add pitch features), you will have to
  # change the options of the following command line.
  steps/online/nnet3/prepare_online_decoding.sh \
       --mfcc-config conf/mfcc_hires.conf \
       $lang exp/nnet3/extractor $dir ${dir}_online

  rm $dir/.error 2>/dev/null || true
  for decode_set in train_dev eval2000 $maybe_rt03; do
    (
      # note: we just give it "$decode_set" as it only uses the wav.scp, the
      # feature type does not matter.

      steps/online/nnet3/decode.sh --nj $decode_nj --cmd "$decode_cmd" \
          --acwt 1.0 --post-decode-acwt 10.0 \
         $graph_dir data/${decode_set}_hires \
         ${dir}_online/decode_${decode_set}${decode_iter:+_$decode_iter}_sw1_tg || exit 1;
      if $has_fisher; then
          steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
            data/lang_sw1_{tg,fsh_fg} data/${decode_set}_hires \
            ${dir}_online/decode_${decode_set}${decode_iter:+_$decode_iter}_sw1_{tg,fsh_fg} || exit 1;
      fi
    ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi


exit 0;
