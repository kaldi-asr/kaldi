#!/bin/bash

# based on 7q, and implemented the basic DFSMN under LF-MMI

# System                tdnn7q_sp dfsmn1a_sp
# WER on train_dev(tg)      12.08     12.41
# WER on train_dev(fg)      11.15     11.35
# WER on eval2000(tg)        14.1      14.3
# WER on eval2000(fg)        12.8      13.0
# WER on rt03(tg)            17.5      17.9
# WER on rt03(fg)            15.3      15.6
# Final train prob         -0.055    -0.064
# Final valid prob         -0.072    -0.076
# Final train prob (xent)        -0.875    -0.942
# Final valid prob (xent)       -0.9064   -0.9547
# Num-parameters               18725244  17943676



# steps/info/chain_dir_info.pl exp/chain/dfsmn1a_sp
# exp/chain/dfsmn1a_sp: num-iters=295 nj=3..16 num-params=17.9M dim=40+100->6078 combine=-0.064->-0.064 (over 2) xent:train/valid[195,294,final]=(-1.29,-0.936,-0.942/-1.28,-0.954,-0.955) logprob:train/valid[195,294,final]=(-0.100,-0.064,-0.064/-0.108,-0.077,-0.076)
set -e

# configs for 'chain'
stage=12
train_stage=-10
get_egs_stage=-10
speed_perturb=true
affix=1a
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
dir=exp/chain/dfsmn${affix}${suffix}

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
  relu-batchnorm-dropout-layer name=tdnn1 input=Append(-1,0,1) dim=1536 l2-regularize=0.01 dropout-proportion=0.0 dropout-per-dim-continuous=true
  linear-component name=tdnn1l dim=256 orthonormal-constraint=-1.0 l2-regularize=0.01

  blocksum-layer name=dfsmn1_projection_pre dim=256  input=Append(-2,-1,0,1,2) 
  relu-batchnorm-dropout-layer name=dfsmn1_inter dim=1536 input=Sum(dfsmn1_projection_pre, tdnn1l) l2-regularize=0.01 dropout-proportion=0.0 dropout-per-dim-continuous=true
  linear-component name=dfsmn1_projection dim=256  orthonormal-constraint=-1.0 l2-regularize=0.01
  
  blocksum-layer name=dfsmn2_projection_pre dim=256  input=Append(-2,-1,0,1,2) 
  relu-batchnorm-dropout-layer name=dfsmn2_inter dim=1536 input=Sum(dfsmn2_projection_pre, dfsmn1_projection_pre, dfsmn1_projection) l2-regularize=0.01 dropout-proportion=0.0 dropout-per-dim-continuous=true
  linear-component name=dfsmn2_projection dim=256  orthonormal-constraint=-1.0 l2-regularize=0.01
  
  blocksum-layer name=dfsmn3_projection_pre dim=256  input=Append(-2,-1,0,1,2) 
  relu-batchnorm-dropout-layer name=dfsmn3_inter dim=1536 input=Sum(dfsmn3_projection_pre, dfsmn2_projection_pre, dfsmn2_projection) l2-regularize=0.01 dropout-proportion=0.0 dropout-per-dim-continuous=true
  linear-component name=dfsmn3_projection dim=256  orthonormal-constraint=-1.0 l2-regularize=0.01

  blocksum-layer name=dfsmn4_projection_pre dim=256  input=Append(-2,-1,0,1,2) 
  relu-batchnorm-dropout-layer name=dfsmn4_inter dim=1536 input=Sum(dfsmn4_projection_pre, dfsmn3_projection_pre, dfsmn3_projection) l2-regularize=0.01 dropout-proportion=0.0 dropout-per-dim-continuous=true
  linear-component name=dfsmn4_projection dim=256  orthonormal-constraint=-1.0 l2-regularize=0.01

  blocksum-layer name=dfsmn5_projection_pre dim=256  input=Append(-2,-1,0,1,2)  
  relu-batchnorm-dropout-layer name=dfsmn5_inter dim=1536 input=Sum(dfsmn5_projection_pre, dfsmn4_projection_pre, dfsmn4_projection) l2-regularize=0.01 dropout-proportion=0.0 dropout-per-dim-continuous=true
  linear-component name=dfsmn5_projection dim=256  orthonormal-constraint=-1.0 l2-regularize=0.01

  blocksum-layer name=dfsmn6_projection_pre dim=256  input=Append(-2,-1,0,1,2)  
  relu-batchnorm-dropout-layer name=dfsmn6_inter dim=1536 input=Sum(dfsmn6_projection_pre, dfsmn5_projection_pre, dfsmn5_projection) l2-regularize=0.01 dropout-proportion=0.0 dropout-per-dim-continuous=true
  linear-component name=dfsmn6_projection dim=256  orthonormal-constraint=-1.0 l2-regularize=0.01

  blocksum-layer name=dfsmn7_projection_pre dim=256  input=Append(-2,-1,0,1,2) 
  relu-batchnorm-dropout-layer name=dfsmn7_inter dim=1536 input=Sum(dfsmn7_projection_pre, dfsmn6_projection_pre, dfsmn6_projection) l2-regularize=0.01 dropout-proportion=0.0 dropout-per-dim-continuous=true
  linear-component name=dfsmn7_projection dim=256  orthonormal-constraint=-1.0 l2-regularize=0.01

  blocksum-layer name=dfsmn8_projection_pre dim=256  input=Append(-2,-1,0,1,2) 
  relu-batchnorm-dropout-layer name=dfsmn8_inter dim=1536 input=Sum(dfsmn8_projection_pre, dfsmn7_projection_pre, dfsmn7_projection) l2-regularize=0.01 dropout-proportion=0.0 dropout-per-dim-continuous=true
  linear-component name=dfsmn8_projection dim=256  orthonormal-constraint=-1.0 l2-regularize=0.01

  blocksum-layer name=dfsmn9_projection_pre dim=256  input=Append(-3,0,3)  
  relu-batchnorm-dropout-layer name=dfsmn9_inter dim=1536 input=Sum(dfsmn9_projection_pre, dfsmn8_projection_pre, dfsmn8_projection) l2-regularize=0.01 dropout-proportion=0.0 dropout-per-dim-continuous=true
  linear-component name=dfsmn9_projection dim=256  orthonormal-constraint=-1.0 l2-regularize=0.01

  blocksum-layer name=dfsmn10_projection_pre dim=256  input=Append(-3,0,3) 
  relu-batchnorm-dropout-layer name=dfsmn10_inter dim=1536 input=Sum(dfsmn10_projection_pre, dfsmn9_projection_pre, dfsmn9_projection) l2-regularize=0.01 dropout-proportion=0.0 dropout-per-dim-continuous=true
  linear-component name=dfsmn10_projection dim=256  orthonormal-constraint=-1.0 l2-regularize=0.01

  blocksum-layer name=dfsmn11_projection_pre dim=256  input=Append(-3,0,3) 
  relu-batchnorm-dropout-layer name=dfsmn11_inter dim=1536 input=Sum(dfsmn11_projection_pre, dfsmn10_projection_pre, dfsmn10_projection) l2-regularize=0.01 dropout-proportion=0.0 dropout-per-dim-continuous=true
  linear-component name=dfsmn11_projection dim=256  orthonormal-constraint=-1.0 l2-regularize=0.01
  
  blocksum-layer name=dfsmn12_projection_pre dim=256  input=Append(-3,0,3)  
  relu-batchnorm-dropout-layer name=dfsmn12_inter dim=1536 input=Sum(dfsmn12_projection_pre, dfsmn11_projection_pre, dfsmn11_projection) l2-regularize=0.01 dropout-proportion=0.0 dropout-per-dim-continuous=true
  linear-component name=dfsmn12_projection dim=256  orthonormal-constraint=-1.0 l2-regularize=0.01
  
  blocksum-layer name=dfsmn13_projection_pre dim=256  input=Append(-3,0,3)  
  relu-batchnorm-dropout-layer name=dfsmn13_inter dim=1536 input=Sum(dfsmn13_projection_pre, dfsmn12_projection_pre, dfsmn12_projection) l2-regularize=0.01 dropout-proportion=0.0 dropout-per-dim-continuous=true
  linear-component name=dfsmn13_projection dim=256  orthonormal-constraint=-1.0 l2-regularize=0.01
  
  blocksum-layer name=dfsmn14_projection_pre dim=256  input=Append(-3,0,3)  
  relu-batchnorm-dropout-layer name=dfsmn14_inter dim=1536 input=Sum(dfsmn14_projection_pre, dfsmn13_projection_pre, dfsmn13_projection) l2-regularize=0.01 dropout-proportion=0.0 dropout-per-dim-continuous=true
  linear-component name=dfsmn14_projection dim=256  orthonormal-constraint=-1.0 l2-regularize=0.01
  
  blocksum-layer name=dfsmn15_projection_pre dim=256  input=Append(-3,0,3)  
  relu-batchnorm-dropout-layer name=dfsmn15_inter dim=1536 input=Sum(dfsmn15_projection_pre, dfsmn14_projection_pre, dfsmn14_projection) l2-regularize=0.01 dropout-proportion=0.0 dropout-per-dim-continuous=true
  linear-component name=dfsmn15_projection dim=256  orthonormal-constraint=-1.0 l2-regularize=0.01


  prefinal-layer name=prefinal-chain input=dfsmn15_projection $prefinal_opts big-dim=1536 small-dim=256
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

  prefinal-layer name=prefinal-xent input=dfsmn15_projection $prefinal_opts big-dim=1536 small-dim=256
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
    --trainer.frames-per-iter 2000000 \
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
