#!/usr/bin/env bash

# 1i is like 1h, while it introduces 'apply-cmvn-online' that does
# cmn normalization both for i-extractor and TDNN input.
# run_tdnn_1i.sh in local/chain2 uses new kaldi recipe.

# local/chain2/compare_wer.sh exp/chain2_online_cmn/tdnn1i_sp
# System                tdnn1i_sp
#WER dev93 (tgpr)                6.83
#WER dev93 (tg)                  6.53
#WER dev93 (big-dict,tgpr)       4.71
#WER dev93 (big-dict,fg)         4.31
#WER eval92 (tgpr)               4.86
#WER eval92 (tg)                 4.43
#WER eval92 (big-dict,tgpr)      2.71
#WER eval92 (big-dict,fg)        2.27
# Final train prob        -0.0397
# Final valid prob        -0.0346
# Final train prob (xent)   -0.7091
# Final valid prob (xent)   -0.6436
# Num-params                 9476352

# steps/info/chain_dir_info.pl exp/chain_online_cmn/tdnn1i_sp
# exp/chain_online_cmn/tdnn1i_sp: num-iters=108 nj=2..8 num-params=8.4M dim=40+100->2880 combine=-0.044->-0.044 (over 1) xent:train/valid[71,107,final]=(-0.873,-0.660,-0.672/-0.906,-0.714,-0.734) logprob:train/valid[71,107,final]=(-0.067,-0.044,-0.044/-0.068,-0.054,-0.055)

# Set -e here so that we catch if any executable fails immediately
set -euo pipefail

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=0
nj=30
train_set=train_si284
test_sets="test_dev93 test_eval92"
gmm=tri4b        # this is the source gmm-dir that we'll use for alignments; it
                 # should have alignments for the specified training data.

num_threads_ubm=8

nj_extractor=10
# It runs a JOB with '-pe smp N', where N=$[threads*processes]
num_threads_extractor=4
num_processes_extractor=2

nnet3_affix=_online_cmn   # affix for exp dirs, e.g. it was _cleaned in tedlium.

# Options which are not passed through to run_ivector_common.sh
affix=1i   #affix for TDNN+LSTM directory e.g. "1a" or "1b", in case we change the configuration.
reporting_email=

# Setting 'online_cmvn' to true replaces 'apply-cmvn' by
# 'apply-cmvn-online' both for i-vector extraction and TDNN input.
# The i-vector extractor uses the config 'conf/online_cmvn.conf' for
# both the UBM and the i-extractor. The TDNN input is configured via
# '--feat.cmvn-opts' that is set to the same config, so we use the
# same cmvn for i-extractor and the TDNN input.
online_cmvn=true

# LSTM/chain options
train_stage=-10
xent_regularize=0.1
dropout_schedule='0,0@0.20,0.5@0.50,0'

# training chunk-options
chunk_width=140,100,160
bottom_subsampling_factor=1  # I'll set this to 3 later, 1 is for compatibility with a broken ru.
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

# training options
srand=0
remove_egs=true

#decode options
test_online_decoding=true  # if true, it will run the last decoding stage.

# End configuration section.
echo "$0 $*"  # Print the command line for logging


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

local/nnet3/run_ivector_common.sh \
  --stage $stage --nj $nj \
  --train-set $train_set --gmm $gmm \
  --online-cmvn-iextractor $online_cmvn \
  --num-threads-ubm $num_threads_ubm \
  --nj-extractor $nj_extractor \
  --num-processes-extractor $num_processes_extractor \
  --num-threads-extractor $num_threads_extractor \
  --nnet3-affix "$nnet3_affix"



gmm_dir=exp/${gmm}
ali_dir=exp/${gmm}_ali_${train_set}_sp
lat_dir=exp/chain2${nnet3_affix}/${gmm}_${train_set}_sp_lats
dir=exp/chain2${nnet3_affix}/tdnn${affix}_sp
train_data_dir=data/${train_set}_sp_hires
train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires
lores_train_data_dir=data/${train_set}_sp

# note: you don't necessarily have to change the treedir name
# each time you do a new experiment-- only if you change the
# configuration in a way that affects the tree.
tree_dir=exp/chain2${nnet3_affix}/tree_a_sp
# the 'lang' directory is created by this script.
# If you create such a directory with a non-standard topology
# you should probably name it differently.
lang=data/lang_chain

for f in $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp \
    $lores_train_data_dir/feats.scp $gmm_dir/final.mdl \
    $ali_dir/ali.1.gz $gmm_dir/final.mdl; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done


if [ $stage -le 12 ]; then
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

if [ $stage -le 13 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --nj 100 --cmd "$train_cmd" ${lores_train_data_dir} \
    data/lang $gmm_dir $lat_dir
  rm $lat_dir/fsts.*.gz # save space
fi

if [ $stage -le 14 ]; then
  # Build a tree using our new topology.  We know we have alignments for the
  # speed-perturbed data (local/nnet3/run_ivector_common.sh made them), so use
  # those.  The num-leaves is always somewhat less than the num-leaves from
  # the GMM baseline.
   if [ -f $tree_dir/final.mdl ]; then
     echo "$0: $tree_dir/final.mdl already exists, refusing to overwrite it."
     exit 1;
  fi
  steps/nnet3/chain/build_tree.sh \
    --frame-subsampling-factor ${frame_subsampling_factor} \
    --context-opts "--context-width=2 --central-position=1" \
    --cmd "$train_cmd" 3500 ${lores_train_data_dir} \
    $lang $ali_dir $tree_dir
fi


if [ $stage -le 15 ]; then
  # $dir/configs will contain xconfig and config files for the initial
  # models.  It's a scratch space used by this script but not by
  # scripts called from here.
  mkdir -p $dir/configs/
  # $dir/init will contain the initial models
  mkdir -p $dir/init/
  
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print(0.5/$xent_regularize)" | python)
  tdnn_opts="l2-regularize=0.01 dropout-proportion=0.0 dropout-per-dim-continuous=true"
  tdnnf_opts="l2-regularize=0.01 dropout-proportion=0.0 bypass-scale=0.66"
  linear_opts="l2-regularize=0.01 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.01"
  output_opts="l2-regularize=0.005"

  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  idct-layer name=idct input=input dim=40 cepstral-lifter=22 affine-transform-file=$dir/configs/idct.mat
  delta-layer name=delta input=idct
  no-op-component name=input2 input=Append(delta, Scale(1.0, ReplaceIndex(ivector, t, 0)))

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 $tdnn_opts dim=1024 input=input2
  tdnnf-layer name=tdnnf2 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=1
  tdnnf-layer name=tdnnf3 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=1
  tdnnf-layer name=tdnnf4 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=1
  tdnnf-layer name=tdnnf5 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=0
  tdnnf-layer name=tdnnf6 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf7 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf9 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf10 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf11 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf12 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf13 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  linear-component name=prefinal-l dim=192 $linear_opts


  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts big-dim=1024 small-dim=192
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts
  output-layer name=output-default input=prefinal-chain include-log-softmax=false dim=$num_targets $output_opts

  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts big-dim=1024 small-dim=192
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
  output-layer name=output-default-xent input=prefinal-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
  if [ ! -f $dir/init/default_trans.mdl ]; then # checking this because it may have been copied in a previous run of the same script
      copy-transition-model $tree_dir/final.mdl $dir/init/default_trans.mdl  || exit 1 &
  else
      echo "Keeping the old $dir/init/default_trans.mdl as it already exists."
  fi
fi

init_info=$dir/init/info.txt
if [ $stage -le 16 ]; then

  if [ ! -f $dir/configs/ref.raw ]; then
      echo "Expected $dir/configs/ref.raw to exist"
      exit
  fi

  nnet3-info $dir/configs/ref.raw  > $dir/configs/temp.info 
  model_left_context=$(grep -F 'left-context' $dir/configs/temp.info | awk '{print $2}')
  model_right_context=$(grep -F 'right-context' $dir/configs/temp.info | awk '{print $2}')
  cat >$init_info <<EOF
frame_subsampling_factor $frame_subsampling_factor
langs $langs
model_left_context $model_left_context
model_right_context $model_right_context
EOF
  rm $dir/configs/temp.info
fi

# Make phone LM and denominator and normalization FST
if [ $stage -le 17 ]; then
  echo "$0: Making Phone LM and denominator and normalization FST"
  mkdir -p $dir/den_fsts/log

  # We may later reorganize this.
  cp $tree_dir/tree $dir/default.tree

  echo "$0: creating phone language-model"
  $train_cmd $dir/den_fsts/log/make_phone_lm_default.log \
    chain-est-phone-lm --num-extra-lm-states=2000 \
       "ark:gunzip -c $gmm_dir/ali.*.gz | ali-to-phones $gmm_dir/final.mdl ark:- ark:- |" \
       $dir/den_fsts/default.phone_lm.fst

  echo "$0: creating denominator FST"
  $train_cmd $dir/den_fsts/log/make_den_fst.log \
     chain-make-den-fst $dir/default.tree $dir/init/default_trans.mdl $dir/den_fsts/default.phone_lm.fst \
     $dir/den_fsts/default.den.fst $dir/den_fsts/default.normalization.fst || exit 1;
fi

model_left_context=$(awk '/^model_left_context/ {print $2;}' $dir/init/info.txt)
model_right_context=$(awk '/^model_right_context/ {print $2;}' $dir/init/info.txt)
if [ -z $model_left_context ]; then
    echo "ERROR: Cannot find entry for model_left_context in $dir/init/info.txt"
fi
if [ -z $model_right_context ]; then
    echo "ERROR: Cannot find entry for model_right_context in $dir/init/info.txt"
fi
# Note: we add frame_subsampling_factor/2 so that we can support the frame
# shifting that's done during training, so if frame-subsampling-factor=3, we
# train on the same egs with the input shifted by -1,0,1 frames.  This is done
# via the --frame-shift option to nnet3-chain-copy-egs in the script.
egs_left_context=$((model_left_context+(frame_subsampling_factor/2)+egs_extra_left_context))
egs_right_context=$((model_right_context+(frame_subsampling_factor/2)+egs_extra_right_context))

for d in $dir/raw_egs $dir/processed_egs; do
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $d/storage ] ; then
    mkdir -p $d
    utils/create_split_dir.pl \
      /export/b0{3,4,5,6}/$USER/kaldi-data/egs/wsj-$(date +'%m_%d_%H_%M')/s5/$d/storage $d/storage
  fi
done

if [ $stage -le 18 ]; then
  echo "$0: about to dump raw egs."
  # Dump raw egs.
  steps/chain2/get_raw_egs.sh --cmd "$train_cmd" \
    --lang "default" \
    --cmvn-opts "--config=conf/online_cmvn.conf" \
    --left-context $egs_left_context \
    --right-context $egs_right_context \
    --frame-subsampling-factor $frame_subsampling_factor \
    --alignment-subsampling-factor $frame_subsampling_factor \
    --frames-per-chunk ${chunk_width} \
    --online-ivector-dir ${train_ivector_dir} \
    ${train_data_dir} ${dir} ${lat_dir} ${dir}/raw_egs
fi

if [ $stage -le 19 ]; then
  echo "$0: about to process egs"
  steps/chain2/process_egs.sh  --cmd "$train_cmd" \
      --num-repeats 1 \
    ${dir}/raw_egs ${dir}/processed_egs
fi

if [ $stage -le 20 ]; then
  echo "$0: about to randomize egs"
  steps/chain2/randomize_egs.sh --frames-per-job 5000000 \
    ${dir}/processed_egs ${dir}/egs
fi

if [ $stage -le 21 ]; then
    echo "$0: Preparing initial acoustic model"
    if [ -f $dir/configs/init.config ]; then
            $train_cmd ${dir}/log/add_first_layer.log \
                    nnet3-init --srand=${srand} ${dir}/configs/init.raw \
                    ${dir}/configs/final.config ${dir}/init/default.raw || exit 1
    else
            $train_cmd ${dir}/log/init_model.log \
               nnet3-init --srand=${srand} ${dir}/configs/final.config ${dir}/init/default.raw || exit 1
    fi

    $train_cmd $dir/log/init_mdl.log \
        nnet3-am-init ${dir}/init/default_trans.mdl $dir/init/default.raw $dir/init/default.mdl || exit 1
fi

if [ $stage -le 22 ]; then
  echo "$0: about to train model"
  steps/chain2/train.sh \
    --stage $train_stage --cmd "$decode_cmd" \
    --xent-regularize $xent_regularize --leaky-hmm-coefficient 0.1 \
    --max-param-change 2.0 \
    --dropout-schedule ${dropout_schedule} \
    --num-jobs-initial 2 --num-jobs-final 8 \
    --initial-effective-lrate 0.0005 \
    --final-effective-lrate 0.00005 \
    --num-epochs 10 \
    --minibatch-size 128,64 \
     $dir/egs $dir || exit 1;
fi

if [ $stage -le 23 ]; then
  # The reason we are using data/lang here, instead of $lang, is just to
  # emphasize that it's not actually important to give mkgraph.sh the
  # lang directory with the matched topology (since it gets the
  # topology file from the model).  So you could give it a different
  # lang directory, one that contained a wordlist and LM of your choice,
  # as long as phones.txt was compatible.

  utils/lang/check_phones_compatible.sh \
    data/lang_test_tgpr/phones.txt $lang/phones.txt
  utils/mkgraph.sh \
    --self-loop-scale 1.0 data/lang_test_tgpr \
    $tree_dir $tree_dir/graph_tgpr || exit 1;

  utils/lang/check_phones_compatible.sh \
    data/lang_test_bd_tgpr/phones.txt $lang/phones.txt
  utils/mkgraph.sh \
    --self-loop-scale 1.0 data/lang_test_bd_tgpr \
    $tree_dir $tree_dir/graph_bd_tgpr || exit 1;
fi

if [ $stage -le 24 ]; then
  frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    (
      data_affix=$(echo $data | sed s/test_//)
      nspk=$(wc -l <data/${data}_hires/spk2utt)
      for lmtype in tgpr bd_tgpr; do
        steps/nnet3/decode.sh \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --extra-left-context $egs_left_context --extra-right-context $egs_right_context \
          --extra-left-context-initial 0 \
          --extra-right-context-final 0 \
          --frames-per-chunk $frames_per_chunk \
          --nj $nspk --cmd "$decode_cmd"  --num-threads 4 \
          --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${data}_hires \
          $tree_dir/graph_${lmtype} data/${data}_hires ${dir}/decode_${lmtype}_${data_affix} || exit 1
      done
      steps/lmrescore.sh \
        --self-loop-scale 1.0 \
        --cmd "$decode_cmd" data/lang_test_{tgpr,tg} \
        data/${data}_hires ${dir}/decode_{tgpr,tg}_${data_affix} || exit 1
      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
        data/lang_test_bd_{tgpr,fgconst} \
       data/${data}_hires ${dir}/decode_${lmtype}_${data_affix}{,_fg} || exit 1
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

# Not testing the 'looped' decoding separately, because for
# TDNN systems it would give exactly the same results as the
# normal decoding.

if $test_online_decoding && [ $stage -le 25 ]; then
  cp $dir/default.tree $dir/tree
  # note: if the features change (e.g. you add pitch features), you will have to
  # change the options of the following command line.
  steps/online/nnet3/prepare_online_decoding.sh \
    --mfcc-config conf/mfcc_hires.conf \
    $lang exp/nnet3${nnet3_affix}/extractor ${dir} ${dir}_online

  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    (
      data_affix=$(echo $data | sed s/test_//)
      nspk=$(wc -l <data/${data}_hires/spk2utt)
      # note: we just give it "data/${data}" as it only uses the wav.scp, the
      # feature type does not matter.
      for lmtype in tgpr bd_tgpr; do
        steps/online/nnet3/decode.sh \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $nspk --cmd "$decode_cmd" \
          $tree_dir/graph_${lmtype} data/${data} ${dir}_online/decode_${lmtype}_${data_affix} || exit 1
      done
      steps/lmrescore.sh \
        --self-loop-scale 1.0 \
        --cmd "$decode_cmd" data/lang_test_{tgpr,tg} \
        data/${data}_hires ${dir}_online/decode_{tgpr,tg}_${data_affix} || exit 1
      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
        data/lang_test_bd_{tgpr,fgconst} \
       data/${data}_hires ${dir}_online/decode_${lmtype}_${data_affix}{,_fg} || exit 1
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi


exit 0;
