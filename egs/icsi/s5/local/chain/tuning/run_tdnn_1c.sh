#!/bin/bash

# this one is based on the best AMI/s5b chain model so far (as of June 2020)
# small differences in #params between conditions are due to condition-specific trees

# Mic ihm
# local/chain/compare_wer.sh exp/ihm/chain_1a/tdnn_b_sp exp/ihm/chain_1a/tdnn_c_sp
# System                tdnn_b_sp tdnn_c_sp
#WER dev       20.3      16.5
#WER eval       12.8      10.8
# Final train prob        -0.0563   -0.0467
# Final valid prob        -0.1135   -0.1026
# Final train prob (xent)   -1.0886   -0.7411
# Final valid prob (xent)   -1.2383   -0.8850
# Num-params                 7899958  33996936

#Mic mdm4
# local/chain/compare_wer.sh exp/mdm4/chain_1a/tdnn_b_sp_ihmali exp/mdm4/chain_1a/tdnn_c_sp_ihmali/
# System                tdnn_b_sp_ihmali tdnn_c_sp_ihmali
#WER dev       28.7      24.5
#WER eval       26.3      22.6
# Final train prob        -0.1160   -0.0827
# Final valid prob        -0.2102   -0.2075
# Final train prob (xent)   -1.7933   -1.1783
# Final valid prob (xent)   -2.2102   -1.8152
# Num-params                 7928822  34005144

# Mic sdm4
# local/chain/compare_wer.sh exp/sdm4/chain_1a/tdnn_b_sp_ihmali exp/sdm4/chain_1a/tdnn_c_sp_ihmali/
# System                tdnn_b_sp_ihmali tdnn_c_sp_ihmali
#WER dev       30.4      26.1
#WER eval       27.4      23.8
# Final train prob        -0.1279   -0.0885
# Final valid prob        -0.2283   -0.2179
# Final train prob (xent)   -1.9735   -1.2759
# Final valid prob (xent)   -2.3826   -1.9211
# Num-params                 7936038  34005144

set -e -o pipefail
# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=0
mic=ihm
nj=30
use_ihm_ali=false
train_set=train
min_seg_len=1.55
gmm=tri3  # the gmm for the target data
ihm_gmm=tri3  # the gmm for the IHM system (if --use-ihm-ali true).
nnet3_affix=_1a  # cleanup affix for nnet3 and chain dirs, e.g. _cleaned
num_epochs=15
remove_egs=true

# The rest are configs specific to this script.  Most of the parameters
# are just hardcoded at this level, in the commands below.
train_stage=-10
tree_affix=  # affix for tree directory, e.g. "a" or "b", in case we change the configuration.
tdnn_affix=_c  #affix for TDNN directory, e.g. "a" or "b", in case we change the configuration.
common_egs_dir=  # you can set this to use previously dumped egs.
dropout_schedule='0,0@0.20,0.5@0.50,0'

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

local/nnet3/run_ivector_common.sh --stage $stage \
                                  --mic $mic \
                                  --nj $nj \
                                  --train-set $train_set \
                                  --gmm $gmm \
                                  --nnet3-affix "$nnet3_affix"

if $use_ihm_ali; then
  gmm_dir=exp/ihm/${ihm_gmm}
  ali_dir=exp/${mic}/${ihm_gmm}_ali_${train_set}_sp_ihmdata
  lores_train_data_dir=data/$mic/${train_set}_ihmdata_sp
  tree_dir=exp/$mic/chain${nnet3_affix}/tree_bi${tree_affix}_ihmdata
  lat_dir=exp/$mic/chain${nnet3_affix}/${gmm}_${train_set}_sp_lats_ihmdata
  dir=exp/$mic/chain${nnet3_affix}/tdnn${tdnn_affix}_sp_ihmali
  # note: the distinction between when we use the 'ihmdata' suffix versus
  # 'ihmali' is pretty arbitrary.
else
  gmm_dir=exp/${mic}/$gmm
  ali_dir=exp/${mic}/${gmm}_ali_${train_set}_sp
  lores_train_data_dir=data/$mic/${train_set}_sp
  tree_dir=exp/$mic/chain${nnet3_affix}/tree_bi${tree_affix}
  lat_dir=exp/$mic/chain${nnet3_affix}/${gmm}_${train_set}_sp_lats
  dir=exp/$mic/chain${nnet3_affix}/tdnn${tdnn_affix}_sp
fi

train_data_dir=data/$mic/${train_set}_sp_hires
train_ivector_dir=exp/$mic/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires
final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-7

if $use_ihm_ali; then
  # the lores features (and alignments) for matched scenario are extracted anyway
  # by local/nnet3/run_ivector_common.sh for ivector training.
  # We only re-extract them from ihm data for mic in [sdmX, mdmX], if the ihm
  # alignments were explicitly requested to be used in either sdmX or mdmX scenarios
  local/prepare_parallel_train_data.sh $mic
  # Note: the first stage of the following script is stage 7
  local/nnet3/prepare_lores_feats.sh --stage $stage \
                                   --mic $mic \
                                   --nj $nj \
                                   --use-ihm-ali $use_ihm_ali \
                                   --train-set $train_set
fi

for f in $gmm_dir/final.mdl $lores_train_data_dir/feats.scp \
   $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done

# note, we only need to get alignments if using ihm alignments, otherwise
# those were already generated in run_ivector_common.sh for $mic of choice
if $use_ihm_ali && [ $stage -le 11 ]; then
  if [ -f $ali_dir/ali.1.gz ]; then
    echo "$0: alignments in $ali_dir appear to already exist.  Please either remove them "
    echo " ... or use a later --stage option."
    exit 1
  fi
  echo "$0: aligning perturbed, short-segment-combined ${maybe_ihm}data"
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    ${lores_train_data_dir} data/lang $gmm_dir $ali_dir
fi

[ ! -f $ali_dir/ali.1.gz ] && echo  "$0: expected $ali_dir/ali.1.gz to exist" && exit 1

if [ $stage -le 12 ]; then
  echo "$0: creating lang directory with one state per phone."
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  if [ -d data/lang_chain ]; then
    if [ data/lang_chain/L.fst -nt data/lang/L.fst ]; then
      echo "$0: data/lang_chain already exists, not overwriting it; continuing"
    else
      echo "$0: data/lang_chain already exists and seems to be older than data/lang..."
      echo " ... not sure what to do.  Exiting."
      exit 1;
    fi
  else
    cp -r data/lang data/lang_chain
    silphonelist=$(cat data/lang_chain/phones/silence.csl) || exit 1;
    nonsilphonelist=$(cat data/lang_chain/phones/nonsilence.csl) || exit 1;
    # Use our special topology... note that later on may have to tune this
    # topology.
    steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >data/lang_chain/topo
  fi
fi

if [ $stage -le 13 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --nj 35 --cmd "$train_cmd" ${lores_train_data_dir} \
    data/lang $gmm_dir $lat_dir
  rm $lat_dir/fsts.*.gz # save space
fi

if [ $stage -le 14 ]; then
  # Build a tree using our new topology.  We know we have alignments for the
  # speed-perturbed data (local/nnet3/run_ivector_common.sh made them), so use
  # those.
  if [ -f $tree_dir/final.mdl ]; then
    echo "$0: $tree_dir/final.mdl already exists, refusing to overwrite it."
    exit 1;
  fi
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --context-opts "--context-width=2 --central-position=1" \
      --leftmost-questions-truncate -1 \
      --cmd "$train_cmd" 4200 ${lores_train_data_dir} data/lang_chain $ali_dir $tree_dir
fi

xent_regularize=0.1

if [ $stage -le 15 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)
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
  relu-batchnorm-dropout-layer name=tdnn1 $affine_opts dim=2136
  tdnnf-layer name=tdnnf2 $tdnnf_opts dim=2136 bottleneck-dim=210 time-stride=1
  tdnnf-layer name=tdnnf3 $tdnnf_opts dim=2136 bottleneck-dim=210 time-stride=1
  tdnnf-layer name=tdnnf4 $tdnnf_opts dim=2136 bottleneck-dim=210 time-stride=1
  tdnnf-layer name=tdnnf5 $tdnnf_opts dim=2136 bottleneck-dim=210 time-stride=0
  tdnnf-layer name=tdnnf6 $tdnnf_opts dim=2136 bottleneck-dim=210 time-stride=3
  tdnnf-layer name=tdnnf7 $tdnnf_opts dim=2136 bottleneck-dim=210 time-stride=3
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=2136 bottleneck-dim=210 time-stride=3
  tdnnf-layer name=tdnnf9 $tdnnf_opts dim=2136 bottleneck-dim=210 time-stride=3
  tdnnf-layer name=tdnnf10 $tdnnf_opts dim=2136 bottleneck-dim=210 time-stride=3
  tdnnf-layer name=tdnnf11 $tdnnf_opts dim=2136 bottleneck-dim=210 time-stride=3
  tdnnf-layer name=tdnnf12 $tdnnf_opts dim=2136 bottleneck-dim=210 time-stride=3
  tdnnf-layer name=tdnnf13 $tdnnf_opts dim=2136 bottleneck-dim=210 time-stride=3
  tdnnf-layer name=tdnnf14 $tdnnf_opts dim=2136 bottleneck-dim=210 time-stride=3
  tdnnf-layer name=tdnnf15 $tdnnf_opts dim=2136 bottleneck-dim=210 time-stride=3
  linear-component name=prefinal-l dim=512 $linear_opts

  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts big-dim=2136 small-dim=512
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts big-dim=2136 small-dim=512
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts

EOF

  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/

fi

if [ $stage -le 16 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/icsi-$(date +'%m_%d_%H_%M')/s5b/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir $train_ivector_dir \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.dropout-schedule $dropout_schedule \
    --egs.dir "$common_egs_dir" \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width 150 \
    --trainer.num-chunk-per-minibatch 32 \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.num-jobs-initial 4 \
    --trainer.optimization.num-jobs-final 4 \
    --trainer.optimization.initial-effective-lrate 0.001 \
    --trainer.optimization.final-effective-lrate 0.0001 \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs $remove_egs \
    --cleanup.preserve-model-interval 50 \
    --feat-dir $train_data_dir \
    --tree-dir $tree_dir \
    --lat-dir $lat_dir \
    --dir $dir

fi

graph_dir=$dir/graph_${LM}
if [ $stage -le 17 ]; then
  # Note: it might appear that this data/lang_chain directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_${LM} $dir $graph_dir
fi

if [ $stage -le 18 ]; then
  rm $dir/.error 2>/dev/null || true
  for decode_set in dev eval; do
      (
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $nj --cmd "$decode_cmd" \
          --online-ivector-dir exp/$mic/nnet3${nnet3_affix}/ivectors_${decode_set}_hires \
          --scoring-opts "--min-lmwt 5" \
         $graph_dir data/$mic/${decode_set}_hires $dir/decode_${decode_set} || exit 1;
      ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi

exit 0
