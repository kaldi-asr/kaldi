#!/bin/bash


# local/chain/compare_wer.sh exp/chain/e2e_tdnn_1a

# steps/info/chain_dir_info.pl exp/chain/e2e_tdnn_1a
# exp/chain/e2e_tdnn_1a: num-iters=102 nj=2..5 num-params=3.7M dim=40->84 combine=-0.117->-0.116 (over 3) logprob:train/valid[67,101,final]=(-0.080,-0.073,-0.073/-0.090,-0.089,-0.088)

set -e

# configs for 'chain'
stage=0
nj=30
gmm=mono
train_stage=-5 # starting from -5 to skip phone-lm estimation
get_egs_stage=-10
affix=1e
remove_egs=false
xent_regularize=0.1

# training options
srand=0
num_epochs=6
num_jobs_initial=2
num_jobs_final=5
chunk_width=140,100,160
common_egs_dir=
reporting_email=
dim=80
bn_dim=20
frames_per_iter=3000000
bs_scale=0.0
train_set=train
combined_train_set=train_combined
test_sets="dev eval"
aug_affix="reverb noise music babble"
aug_prefix="rev1 noise music babble"
wake_word="嗨小问"

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

local/nnet3/run_ivector_common.sh \
  --stage $stage --nj $nj \
  --train-set $train_set --gmm $gmm --combined-train-set $combined_train_set \
  --aug-affix "$aug_affix" || exit 1;

gmm_dir=exp/${gmm}
ali_dir=exp/${gmm}_ali_${train_set}_sp
lat_dir=exp/chain/${gmm}_${train_set}_sp_lats
combined_lat_dir=exp/chain/${gmm}_${combined_train_set}_lats
train_data_dir=data/${train_set}_sp_hires
combined_train_data_dir=data/${combined_train_set}_hires
train_ivector_dir=exp/nnet3/ivectors_${train_set}_sp_hires
combined_train_ivector_dir=exp/nnet3/ivectors_${combined_train_set}_hires
lores_train_data_dir=data/${train_set}_sp

lang=data/lang_chain
lang_decode=data/lang_chain_decode
tree_dir=exp/chain/tree  # it's actually just a trivial tree (no tree building)
dir=exp/chain/tdnn_${affix}

for f in $train_data_dir/feats.scp $combined_train_data_dir/feats.scp \
  $train_ivector_dir/ivector_online.scp $combined_train_ivector_dir/ivector_online.scp \
  $lores_train_data_dir/feats.scp $gmm_dir/final.mdl $ali_dir/ali.1.gz; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done

if [ $stage -le 10 ]; then
  echo "$0: creating lang directory $lang with chain-type topology"
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.
  if [ -d $lang ]; then
    if [ $lang/L.fst -nt data/lang/L.fst ]; then
      echo "$0: $lang already exists, not overwriting it; continuing"
    else
      echo "$0: $lang already exists and seems to be older than data/lang ..."
      echo " ... not sure what to do.  Exiting."
      exit 1;
    fi
  else
    rm -rf $lang
    cp -r data/lang $lang
    silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
    nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
    # Use our special topology... note that later on may have to tune this
    # topology.
    local/gen_topo.pl 4 1 $nonsilphonelist $silphonelist >$lang/topo
  fi
fi

if [ $stage -le 11 ]; then
  # Get the alignments as lattices (gives the chain training more freedom)
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --nj 75 --cmd "$train_cmd" ${lores_train_data_dir} \
    data/lang $gmm_dir $lat_dir
  rm $lat_dir/fsts.*.gz # save space
fi

if [ $stage -le 12 ]; then
  local/copy_lat_dir.sh --nj 75 --utt-prefixes "$aug_prefix" \
    $combined_train_data_dir $lat_dir $combined_lat_dir
fi

if [ $stage -le 13 ]; then
  # Build a tree using our new topology.  We know we have alignments for the
  # speed-perturbed data (local/nnet3/run_ivector_common.sh made them), so use
  # those.  The num-leaves is always somewhat less than the num-leaves from
  # the GMM baseline.
  if [ -f $tree_dir/final.mdl ]; then
    echo "$0: $tree_dir/final.mdl already exists, refusing to overwrite it."
    exit 1;
  fi
  local/chain/build_tree.sh \
    --frame-subsampling-factor 3 \
    --cmd "$train_cmd" ${lores_train_data_dir} \
    $lang $ali_dir $tree_dir

  echo "$0: Estimating a phone language model for the denominator graph..."
  id_sil=`cat data/lang/phones.txt | grep "SIL" | awk '{print $2}'`
  id_word=`cat data/lang/phones.txt | grep "hixiaowen" | awk '{print $2}'`
  id_freetext=`cat data/lang/phones.txt | grep "freetext" | awk '{print $2}'`
  cat <<EOF > $tree_dir/phone_lm.txt
0 1 $id_sil $id_sil
1 2 $id_word $id_word
0 2 $id_word $id_word
2 3 $id_sil $id_sil
1 4 $id_freetext $id_freetext
0 4 $id_freetext $id_freetext
4 5 $id_sil $id_sil
3 1.9
5 0.7
EOF
#0 1 $id_sil $id_sil
#1 0.0
#1 2 $id_word $id_word
#0 2 $id_word $id_word
#2 2.0
#2 3 $id_sil $id_sil 
#3 2.0

#0 1 $id_sil $id_sil 0.5
#1 2 $id_word $id_word 0.0
#2 3 $id_sil $id_sil 0.0
#0 3 $id_sil $id_sil 0.5
#3
  fstcompile $tree_dir/phone_lm.txt $tree_dir/phone_lm.fst
  fstdeterminizestar $tree_dir/phone_lm.fst $tree_dir/phone_lm.fst.tmp
  mv $tree_dir/phone_lm.fst.tmp $tree_dir/phone_lm.fst
fi

if [ $stage -le 14 ]; then
  echo "$0: creating neural net configs using the xconfig parser";
  num_targets=$(tree-info $tree_dir/tree | grep num-pdfs | awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)
  affine_opts="l2-regularize=0.01 dropout-proportion=0.0 dropout-per-dim=true dropout-per-dim-continuous=true"
  tdnnf_opts="l2-regularize=0.01 dropout-proportion=0.0 bypass-scale=0.66"
  linear_opts="l2-regularize=0.01 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.01"
  output_opts="l2-regularize=0.002"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  #input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-1,0,1) affine-transform-file=$dir/configs/lda.mat

  relu-batchnorm-dropout-layer name=tdnn1 $affine_opts dim=$dim
  tdnnf-layer name=tdnnf2 $tdnnf_opts dim=$dim bottleneck-dim=$bn_dim time-stride=1
  tdnnf-layer name=tdnnf3 $tdnnf_opts dim=$dim bottleneck-dim=$bn_dim time-stride=1
  tdnnf-layer name=tdnnf4 $tdnnf_opts dim=$dim bottleneck-dim=$bn_dim time-stride=1
  tdnnf-layer name=tdnnf4p $tdnnf_opts dim=$dim bottleneck-dim=$bn_dim time-stride=1
  tdnnf-layer name=tdnnf4pp $tdnnf_opts dim=$dim bottleneck-dim=$bn_dim time-stride=1
  tdnnf-layer name=tdnnf4ppp $tdnnf_opts dim=$dim bottleneck-dim=$bn_dim time-stride=1
  tdnnf-layer name=tdnnf4pppp $tdnnf_opts dim=$dim bottleneck-dim=$bn_dim time-stride=1
  tdnnf-layer name=tdnnf5 $tdnnf_opts dim=$dim bottleneck-dim=$bn_dim time-stride=0
  tdnnf-layer name=tdnnf6 $tdnnf_opts dim=$dim bottleneck-dim=$bn_dim time-stride=3
  tdnnf-layer name=tdnnf7 $tdnnf_opts dim=$dim bottleneck-dim=$bn_dim time-stride=3
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=$dim bottleneck-dim=$bn_dim time-stride=3
  tdnnf-layer name=tdnnf9 $tdnnf_opts dim=$dim bottleneck-dim=$bn_dim time-stride=3
  tdnnf-layer name=tdnnf10 $tdnnf_opts dim=$dim bottleneck-dim=$bn_dim time-stride=3
  tdnnf-layer name=tdnnf11 $tdnnf_opts dim=$dim bottleneck-dim=$bn_dim time-stride=3
  tdnnf-layer name=tdnnf12 $tdnnf_opts dim=$dim bottleneck-dim=$bn_dim time-stride=3
  tdnnf-layer name=tdnnf13 $tdnnf_opts dim=$dim bottleneck-dim=$bn_dim time-stride=3
  tdnnf-layer name=tdnnf14 $tdnnf_opts dim=$dim bottleneck-dim=$bn_dim time-stride=3
  tdnnf-layer name=tdnnf15 $tdnnf_opts dim=$dim bottleneck-dim=$bn_dim time-stride=3
  tdnnf-layer name=tdnnf15p $tdnnf_opts dim=$dim bottleneck-dim=$bn_dim time-stride=3
  linear-component name=prefinal-l dim=30 $linear_opts
  
  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts big-dim=$dim small-dim=30
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts big-dim=$dim small-dim=30
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs
fi

if [ $stage -le 15 ]; then
  # no need to store the egs in a shared storage because we always
  # remove them. Anyway, it takes only 5 minutes to generate them.

  cp $tree_dir/phone_lm.fst $dir/phone_lm.fst

  #--feat.online-ivector-dir=$combined_train_ivector_dir \
  steps/nnet3/chain/train.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.cmvn-opts="--norm-means=true --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient=0.1 \
    --chain.l2-regularize=0.0 \
    --chain.apply-deriv-weights=false \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=$num_epochs \
    --trainer.frames-per-iter $frames_per_iter \
    --trainer.optimization.num-jobs-initial $num_jobs_initial \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.initial-effective-lrate 0.00005 \
    --trainer.optimization.final-effective-lrate 0.000005 \
    --trainer.optimization.backstitch-training-scale $bs_scale \
    --trainer.num-chunk-per-minibatch=128,64 \
    --trainer.optimization.momentum=0.0 \
    --egs.chunk-width=$chunk_width \
    --egs.chunk-left-context=0 \
    --egs.chunk-right-context=0 \
    --egs.chunk-left-context-initial=0 \
    --egs.chunk-right-context-final=0 \
    --egs.dir="$common_egs_dir" \
    --egs.opts="--frames-overlap-per-eg 0" \
    --cleanup.remove-egs=$remove_egs \
    --use-gpu=true \
    --reporting.email="$reporting_email" \
    --feat-dir $combined_train_data_dir \
    --tree-dir $tree_dir \
    --lat-dir=$combined_lat_dir \
    --dir=$dir  || exit 1;
fi

if [ $stage -le 16 ]; then
  rm -rf $lang_decode
  utils/prepare_lang.sh --num-sil-states 1 --num-nonsil-states 4 --sil-prob 0.0 \
    --position-dependent-phones false \
  data/local/dict "<sil>" $lang_decode/temp $lang_decode

  sil_id=`cat $lang_decode/words.txt | grep "<sil>" | awk '{print $2}'`
  freetext_id=`cat $lang_decode/words.txt | grep "FREETEXT" | awk '{print $2}'`
  id=`cat $lang_decode/words.txt | grep "嗨小问" | awk '{print $2}'`
  mkdir -p $lang_decode/lm
  cat <<EOF > $lang_decode/lm/fst.txt
0 1 $sil_id $sil_id
1 4 $freetext_id $freetext_id 0.6
4 0 $sil_id $sil_id
1 2 $id $id 2.5
2 0 $sil_id $sil_id
0
EOF
#0 1 $sil_id $sil_id
#0 4 $freetext_id $freetext_id
#1 4 $freetext_id $freetext_id
#4 5 $sil_id $sil_id
#1 2 $id $id
#0 2 $id $id
#2 3 $sil_id $sil_id
#2 1.9
#3 1.9
#4 0.7
#5 0.7
  fstcompile $lang_decode/lm/fst.txt $lang_decode/G.fst
  set +e
  fstisstochastic $lang_decode/G.fst
  set -e
  utils/validate_lang.pl $lang_decode
  cp $lang/topo $lang_decode/topo
fi

if [ $stage -le 17 ]; then
  # The reason we are using data/lang here, instead of $lang, is just to
  # emphasize that it's not actually important to give mkgraph.sh the
  # lang directory with the matched topology (since it gets the
  # topology file from the model).  So you could give it a different
  # lang directory, one that contained a wordlist and LM of your choice,
  # as long as phones.txt was compatible.

  utils/lang/check_phones_compatible.sh \
    data/lang/phones.txt $lang_decode/phones.txt
  utils/mkgraph.sh \
    --self-loop-scale 1.0 $lang_decode \
    $dir $tree_dir/graph || exit 1;
fi

if [ $stage -le 18 ]; then
  frames_per_chunk=150
  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    (
      nspk=$(wc -l <data/${data}_hires/spk2utt)
      #--online-ivector-dir exp/nnet3/ivectors_${data}_hires \
      steps/nnet3/decode.sh \
        --acwt 1.0 --post-decode-acwt 10.0 \
        --extra-left-context-initial 0 \
        --extra-right-context-final 0 \
        --frames-per-chunk $frames_per_chunk \
        --nj $nspk --cmd "$decode_cmd"  --num-threads 4 \
        $tree_dir/graph data/${data}_hires ${dir}/decode_${data} || exit 1
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

if [ $stage -le 19 ]; then
  for data in $test_sets; do
    python3 local/plot_roc.py $dir/decode_${data}/scoring_kaldi/all_results
    nspk=$(wc -l <data/${data}_hires/spk2utt)
    local/process_lattice.sh --nj $nspk --wake-word $wake_word ${dir}/decode_${data} data/${data}_hires $lang || exit 1
  done
  echo "Done. Date: $(date). Results:"
  local/chain/compare_wer.sh $dir
fi


