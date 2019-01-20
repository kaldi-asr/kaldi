#!/bin/bash


# Not working well yet (WER should be closer to 12%.  Need to check for bugs).

#a09:s5: grep WER exp/chaina/tdnn1b_sp/decode_dev_clean_2_tgsmall.si/wer_* | utils/best_wer.sh
#%WER 20.12 [ 4052 / 20138, 394 ins, 569 del, 3089 sub ] exp/chaina/tdnn1b_sp/decode_dev_clean_2_tgsmall.si/wer_10_0.0
#a09:s5: grep WER exp/chaina/tdnn1b_sp/decode_dev_clean_2_tgsmall/wer_* | utils/best_wer.sh
#%WER 18.13 [ 3652 / 20138, 297 ins, 613 del, 2742 sub ] exp/chaina/tdnn1b_sp/decode_dev_clean_2_tgsmall/wer_13_0.0

# Set -e here so that we catch if any executable fails immediately
set -euo pipefail

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=0
decode_nj=10
train_set=train_clean_5
test_sets=dev_clean_2
gmm=tri3b
srand=0
nnet3_affix=

# The rest are configs specific to this script.  Most of the parameters
# are just hardcoded at this level, in the commands below.
affix=1b   # affix for the TDNN directory name
tree_affix=
train_stage=-10
get_egs_stage=-10


# training chunk-options
chunk_width=140
dropout_schedule='0,0@0.20,0.3@0.50,0'
xent_regularize=0.1
bottom_subsampling_factor=3
frame_subsampling_factor=3
langs="default"  # list of language names

# The amount of extra left/right context we put in the egs.  Note: this could
# easily be zero, since we're not using a recurrent topology, but we put in a
# little extra context so that we have more room to play with the configuration
# without re-dumping egs.
egs_extra_left_context=5
egs_extra_right_context=5

# The number of chunks (of length: see $chunk_width above) that we group
# together for each "speaker" (actually: pseudo-speaker, since we may have
# to group multiple speaker together in some cases).
chunks_per_group=4


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

local/chaina/data_prep_common.sh --stage $stage \
                                 --train-set $train_set \
                                 --gmm $gmm  || exit 1;

# Problem: We have removed the "train_" prefix of our training set in
# the alignment directory names! Bad!
gmm_dir=exp/$gmm
ali_dir=exp/${gmm}_ali_${train_set}_sp
tree_dir=exp/chaina/tree_sp${tree_affix:+_$tree_affix}
lang=data/lang_chain
lat_dir=exp/chaina/${gmm}_${train_set}_sp_lats
dir=exp/chaina/tdnn${affix}_sp
train_data_dir=data/${train_set}_sp_hires
lores_train_data_dir=data/${train_set}_sp

for f in $gmm_dir/final.mdl $train_data_dir/feats.scp \
    $lores_train_data_dir/feats.scp $ali_dir/ali.1.gz; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done

if [ $stage -le 10 ]; then
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

if [ $stage -le 11 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --nj 75 --cmd "$train_cmd" ${lores_train_data_dir} \
    data/lang $gmm_dir $lat_dir
  rm $lat_dir/fsts.*.gz # save space
fi

if [ $stage -le 12 ]; then
  # Build a tree using our new topology.  We know we have alignments for the
  # speed-perturbed data (local/nnet3/run_ivector_common.sh made them), so use
  # those.  The num-leaves is always somewhat less than the num-leaves from
  # the GMM baseline.
  # This will be a two-level tree (with the smaller number of leaves specified
  # by the '--num-clusters' option); this is needed by the adaptation framework
  # search below for 'tree.map'
   if [ -f $tree_dir/final.mdl ]; then
     echo "$0: $tree_dir/final.mdl already exists, refusing to overwrite it."
     exit 1;
  fi
   steps/nnet3/chain/build_tree.sh \
     --num-clusters 200 \
     --frame-subsampling-factor ${frame_subsampling_factor} \
     --context-opts "--context-width=2 --central-position=1" \
     --cmd "$train_cmd" 3500 ${lores_train_data_dir} \
     $lang $ali_dir $tree_dir
fi


# $dir/configs will contain xconfig and config files for the initial
# models.  It's a scratch space used by this script but not by
# scripts called from here.
mkdir -p $dir/configs/
# $dir/init will contain the initial models
mkdir -p $dir/init/

l2=0.03
tdnn_opts="l2-regularize=0.03 dropout-proportion=0.0 dropout-per-dim-continuous=true"
tdnnf_opts="l2-regularize=0.03 dropout-proportion=0.0 bypass-scale=0.66"
linear_opts="l2-regularize=0.03 orthonormal-constraint=-1.0"
prefinal_opts="l2-regularize=0.03"
output_opts="l2-regularize=0.015"
num_leaves=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')
learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)

if [ $stage -le 13 ]; then
  echo "$0: creating top neural net using the xconfig parser";

  cat <<EOF > $dir/configs/bottom.xconfig
  input dim=40 name=input

  batchnorm-component name=input-batchnorm

  relu-batchnorm-dropout-layer name=tdnn1 $tdnn_opts dim=768 input=Append(-1,0,1)
  tdnnf-layer name=tdnnf2 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=1
  tdnnf-layer name=tdnnf3 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=1
  tdnnf-layer name=tdnnf4 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=1
  tdnnf-layer name=tdnnf5 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=0
  # this 'batchnorm-layer' has an affine component but no nonlinearlity
  linear-component name=linear_bottleneck dim=256 l2-regularize=$l2
  batchnorm-component name=linear_bottleneck_bn
  output name=output input=linear_bottleneck_bn
EOF
  steps/nnet3/xconfig_to_config.py --xconfig-file $dir/configs/bottom.xconfig \
                                   --config-file-out $dir/configs/bottom.config
  nnet3-init --srand=$srand $dir/configs/bottom.config $dir/init/bottom.raw
fi

if [ $stage -le 14 ]; then
  echo "$0: creating adaptation model/transform"

  # note: 'default' corresponds to the language name (we use 'default' since this
  # is not really a multilingual setup.
  # Note: the bottleneck dimension of 256 specified in the bottom.nnet must match
  # with the dimension of this transform (256).
  cat <<EOF | nnet3-adapt --binary=false init - $tree_dir/tree.map $dir/init/default.ada
AppendTransform num-transforms=6
  NoOpTransform dim=64
  MeanOnlyTransform dim=64
  FmllrTransform dim=32
  FmllrTransform dim=32
  FmllrTransform dim=32
  FmllrTransform dim=32
EOF

  # check the dimensions match
  transform_dim=$(nnet3-adapt info $dir/init/default.ada | grep '^dim' | awk -F= '/^dim/ { print $2; }')
  bottom_output_dim=$(nnet3-info $dir/init/bottom.raw | grep 'output-node name=output ' | perl -ane 'm/dim=(\d+)/ && print $1;')
  if ! [ "$transform_dim" -eq "$bottom_output_dim" ]; then
    echo "$0: expected dim of transform to equal output-dim of bottom nnet, got '$transform_dim' != '$bottom_output_dim'"
    exit 1
  fi
fi


if [ $stage -le 15 ]; then

  # Note: we'll use --bottom-subsampling-factor=3, so all time-strides for the
  # top network should be interpreted at the 30ms frame subsampling rate.

  echo "$0: creating top model"
  cat <<EOF > $dir/configs/default.xconfig
  input name=input dim=256
  linear-component $linear_opts name=linear_from_input dim=768
  tdnnf-layer name=tdnnf1 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=1
  tdnnf-layer name=tdnnf2 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=1
  tdnnf-layer name=tdnnf3 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=1
  tdnnf-layer name=tdnnf4 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=1
  tdnnf-layer name=tdnnf5 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=1
  tdnnf-layer name=tdnnf6 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=1
  tdnnf-layer name=tdnnf7 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=1
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=768 bottleneck-dim=96 time-stride=1
  linear-component name=prefinal-l dim=192 $linear_opts

  # adding the output layer for chain branch
  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts small-dim=192 big-dim=768
  output-layer name=output include-log-softmax=false dim=$num_leaves $output_opts
  # .. and its speaker-independent version
  prefinal-layer name=prefinal-chain-si input=prefinal-l $prefinal_opts small-dim=192 big-dim=768
  output-layer name=output-si include-log-softmax=false dim=$num_leaves $output_opts

  # adding the output layer for xent branch
  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts small-dim=192 big-dim=768
  output-layer name=output-xent dim=$num_leaves learning-rate-factor=$learning_rate_factor $output_opts
  # .. and its speaker-independent version
  prefinal-layer name=prefinal-xent-si input=prefinal-l $prefinal_opts small-dim=192 big-dim=768
  output-layer name=output-si-xent dim=$num_leaves learning-rate-factor=$learning_rate_factor $output_opts
EOF
  steps/nnet3/xconfig_to_config.py --xconfig-file $dir/configs/default.xconfig \
                                   --config-file-out $dir/configs/default.config
  nnet3-init --srand=$srand $dir/configs/default.config - | \
     nnet3-am-init $tree_dir/final.mdl - $dir/init/default.mdl
fi


if [ $stage -le 16 ]; then
  # Work out the model's total effective left and right context (in the
  # feature frame-sampling rate).
  # The following script is equivalent to doing something like the
  # following:
  # cat > $dir/init/info.txt <<EOF
  # langs default
  # frame_subsampling_factor 3
  # bottom_subsampling_factor 3
  # model_left_context 22
  # model_right_context 22
  # EOF
  #
  # note: $langs is "default"
  steps/chaina/get_model_context.sh \
        --frame-subsampling-factor $frame_subsampling_factor \
        --bottom-subsampling-factor $bottom_subsampling_factor \
       --langs "$langs" $dir/init/ $dir/init/info.txt
fi


if [ $stage -le 17 ]; then
  # Make phone LM and denominator and normalization FST
  mkdir -p $dir/den_fsts/log

  # We may later reorganize this.
  cp $tree_dir/tree $dir/default.tree

  echo "$0: creating phone language-model"
  $cmd $dir/den_fsts/log/make_phone_lm_default.log \
    chain-est-phone-lm --num-extra-lm-states=2000 \
       "ark:gunzip -c $gmm_dir/ali.*.gz | ali-to-phones $gmm_dir/final.mdl ark:- ark:- |" \
       $dir/den_fsts/default.phone_lm.fst

  echo "$0: creating denominator FST"
  $cmd $dir/den_fsts/log/make_den_fst.log \
     chain-make-den-fst $dir/default.tree $dir/init/default.mdl $dir/den_fsts/default.phone_lm.fst \
     $dir/den_fsts/default.den.fst $dir/den_fsts/default.normalization.fst || exit 1;
fi


model_left_context=$(awk '/^model_left_context/ {print $2;}' $dir/init/info.txt)
model_right_context=$(awk '/^model_right_context/ {print $2;}' $dir/init/info.txt)
# Note: we add frame_subsampling_factor/2 so that we can support the frame
# shifting that's done during training, so if frame-subsampling-factor=3, we
# train on the same egs with the input shifted by -1,0,1 frames.  This is done
# via the --frame-shift option to nnet3-chain-copy-egs in the script.
egs_left_context=$[model_left_context+(frame_subsampling_factor/2)+egs_extra_left_context]
egs_right_context=$[model_right_context+(frame_subsampling_factor/2)+egs_extra_right_context]

for d in $dir/raw_egs $dir/processed_egs; do
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $d/storage ] ; then
    mkdir -p $d
    utils/create_split_dir.pl \
      /export/b0{3,4,5,6}/$USER/kaldi-data/egs/mini_librispeech-$(date +'%m_%d_%H_%M')/s5/$d/storage $d/storage
  fi
done


if [ $stage -le 18 ]; then
  echo "$0: about to dump raw egs."
  # Dump raw egs.
  steps/chaina/get_raw_egs.sh --cmd "$cmd" \
    --lang "default" \
    --left-context $egs_left_context \
    --right-context $egs_right_context \
    --frame-subsampling-factor $frame_subsampling_factor \
    --alignment-subsampling-factor $frame_subsampling_factor \
    --frames-per-chunk 150 \
    ${train_data_dir} ${dir} ${lat_dir} ${dir}/raw_egs
fi

if [ $stage -le 19 ]; then
  echo "$0: about to process egs"
  steps/chaina/process_egs.sh  --cmd "$cmd" \
    --chunks-per-group ${chunks_per_group} ${dir}/raw_egs ${dir}/processed_egs
fi

if [ $stage -le 20 ]; then
  echo "$0: about to randomize egs"
  steps/chaina/randomize_egs.sh --frames-per-job 3000000 \
    ${dir}/processed_egs ${dir}/egs
fi

if [ $stage -le 21 ]; then
  echo "$0: about to train model"
  steps/chaina/train.sh \
    --stage $train_stage --cmd "$cmd" \
    --xent-regularize $xent_regularize --leaky-hmm-coefficient 0.1 \
    --dropout-schedule "$dropout_schedule" \
    --num-jobs-initial 2 --num-jobs-final 4 \
     $dir/egs $dir

fi


if [ $stage -le 22 ]; then
  # Dump the bottom-nnet outputs for this data.
  test_sets=dev_clean_2
  for data in $test_sets; do
    steps/chaina/compute_embeddings.sh data/${data}_hires $dir/final $dir/data/final/${data}
  done
fi

if [ $stage -le 23 ]; then
  # Note: it's not important to give mkgraph.sh the lang directory with the
  # matched topology (since it gets the topology file from the model).
  utils/mkgraph.sh \
    --self-loop-scale 1.0 data/lang_test_tgsmall \
    $tree_dir $tree_dir/graph_tgsmall || exit 1;
fi

if [ $stage -le 24 ]; then
  # Do the speaker-independent decoding pass
  test_sets=dev_clean_2
  for data in $test_sets; do
    steps/chaina/decode_si.sh --cmd "$cmd" --nj 10 --num-threads 4 \
        data/${data}_hires $tree_dir/graph_tgsmall\
        $dir/final $dir/data/final/${data} \
        $dir/decode_${data}_tgsmall.si
  done
fi

if [ $stage -le 25 ]; then
  # Do the speaker-dependent decoding pass
  test_sets=dev_clean_2
  for data in $test_sets; do
    steps/chaina/decode.sh --cmd "$cmd" --num-threads 4 \
        data/${data}_hires $tree_dir/graph_tgsmall\
        $dir/final $dir/data/final/${data} \
        $dir/decode_${data}_tgsmall.si $dir/decode_${data}_tgsmall
  done
fi


exit 0;


  # Work out the model
  # The following script is equivalent to doing something like the
  # following:
  # cat > $dir/init/info.txt <<EOF
  # langs default
  # frame_subsampling_factor 3
  # bottom_subsampling_factor 3
  # model_left_context 22
  # model_right_context 22
  # EOF
  #
  # note: $langs is "default"
  steps/chaina/get_model_context.sh \
        --frame-subsampling-factor=$frame_subsampling_factor \
        --bottom-subsampling-factor=$bottom_subsampling_factor \
       --langs="$langs" $dir/init/ > $dir/init/info.txt
fi

if [ $stage -le 14 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/mini_librispeech-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/chain/train.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.online-ivector-dir=$train_ivector_dir \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient=0.1 \
    --chain.l2-regularize=0.0 \
    --chain.apply-deriv-weights=false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=20 \
    --trainer.frames-per-iter=3000000 \
    --trainer.optimization.num-jobs-initial=2 \
    --trainer.optimization.num-jobs-final=5 \
    --trainer.optimization.initial-effective-lrate=0.002 \
    --trainer.optimization.final-effective-lrate=0.0002 \
    --trainer.num-chunk-per-minibatch=128,64 \
    --egs.chunk-width=$chunk_width \
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

if [ $stage -le 15 ]; then
  # Note: it's not important to give mkgraph.sh the lang directory with the
  # matched topology (since it gets the topology file from the model).
  utils/mkgraph.sh \
    --self-loop-scale 1.0 data/lang_test_tgsmall \
    $tree_dir $tree_dir/graph_tgsmall || exit 1;
fi

if [ $stage -le 16 ]; then
  frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    (
      nspk=$(wc -l <data/${data}_hires/spk2utt)
      steps/nnet3/decode.sh \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --frames-per-chunk $frames_per_chunk \
          --nj $nspk --cmd "$decode_cmd"  --num-threads 4 \
          --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${data}_hires \
          $tree_dir/graph_tgsmall data/${data}_hires ${dir}/decode_tgsmall_${data} || exit 1
      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
        data/lang_test_{tgsmall,tglarge} \
       data/${data}_hires ${dir}/decode_{tgsmall,tglarge}_${data} || exit 1
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

# Not testing the 'looped' decoding separately, because for
# TDNN systems it would give exactly the same results as the
# normal decoding.

if $test_online_decoding && [ $stage -le 17 ]; then
  # note: if the features change (e.g. you add pitch features), you will have to
  # change the options of the following command line.
  steps/online/nnet3/prepare_online_decoding.sh \
    --mfcc-config conf/mfcc_hires.conf \
    $lang exp/nnet3${nnet3_affix}/extractor ${dir} ${dir}_online

  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    (
      nspk=$(wc -l <data/${data}_hires/spk2utt)
      # note: we just give it "data/${data}" as it only uses the wav.scp, the
      # feature type does not matter.
      steps/online/nnet3/decode.sh \
        --acwt 1.0 --post-decode-acwt 10.0 \
        --nj $nspk --cmd "$decode_cmd" \
        $tree_dir/graph_tgsmall data/${data} ${dir}_online/decode_tgsmall_${data} || exit 1
      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
        data/lang_test_{tgsmall,tglarge} \
       data/${data}_hires ${dir}_online/decode_{tgsmall,tglarge}_${data} || exit 1
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi


exit 0;
