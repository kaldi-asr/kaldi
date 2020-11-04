#!/usr/bin/env bash
set -e

# This script fine tunes a pretrained model on additional data
# which is reverberant, like LibriCSS. We only fine tune for
# 1 epoch.

# configs for 'chain'
stage=0
nj=40
decode_nj=50
train_set=train_960_cleaned
gmm=tri6b_cleaned
nnet3_affix=_cleaned

# Pretrained models for AM and i-vector extractor
src_model_dir=../s5_css/exp/chain$nnet3_affix/tdnn_1d2_sp
ivector_extractor=exp/nnet3$nnet3_affix/extractor
primary_lr_factor=0.1 # The learning-rate factor for transferred layers from source
                       # model. e.g. if 0, the paramters transferred from source model
                       # are fixed.
                       # The learning-rate factor for new added layers is 1.0.

# The rest are configs specific to this script.  Most of the parameters
# are just hardcoded at this level, in the commands below.
affix=1d2_ft
tree_affix=reverb
train_stage=-10
get_egs_stage=-10
decode_iter=

# TDNN options
frames_per_eg=150,110,100
remove_egs=true
common_egs_dir=
xent_regularize=0.1
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

lang=data/lang_chain
ali_dir=exp/${gmm}_ali_${train_set}_reverb
tree_dir=exp/chain${nnet3_affix}/tree${tree_affix:+_$tree_affix}
lat_dir=exp/chain${nnet3_affix}/chain_${train_set}_reverb_lats
dir=exp/chain${nnet3_affix}/tdnn${affix:+_$affix}
train_data_dir=data/${train_set}_reverb_hires
train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_reverb_hires

if [ $stage -le 1 ]; then
  # Adding simulated RIRs to the original data directory
  echo "$0: Preparing data/${train_set}_reverb directory"

  if [ ! -d "RIRS_NOISES" ]; then
    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
  fi

  if [ ! -f data/$train_set/reco2dur ]; then
    utils/data/get_reco2dur.sh --nj $nj --cmd "$train_cmd" data/$train_set || exit 1;
  fi

  # Make a version with reverberated speech
  rvb_opts=()
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

  # Make a reverberated version of train-960.
  # Note that we don't add any additive noise here.
  steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --prefix "reverb" \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications 1 \
    --source-sampling-rate 16000 \
    data/$train_set data/${train_set}_reverb
fi

if [ $stage -le 2 ]; then 
  # Feature extraction for reverberated data.
  mfccdir=mfcc_hires
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/mfcc/librispeech-$date/s5c/$mfccdir/storage $mfccdir/storage
  fi

  # First we extract 13-dim MFCCs which will be used to obtain training lattices
  echo "$0: Creating MFCCs for dir data/${train_set}_reverb"
  utils/copy_data_dir.sh data/${train_set}_reverb data/${train_set}_reverb_hires
  utils/data/perturb_data_dir_volume.sh data/${train_set}_reverb_hires

  steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/${train_set}_reverb_hires exp/make_hires/${train_set}_reverb $mfccdir;
  steps/compute_cmvn_stats.sh data/${train_set}_reverb_hires exp/make_hires/${train_set}_reverb $mfccdir;

  # Now we extract hires MFCCs which will be used for acoustic model training
  echo "$0: Creating hi resolution MFCCs for dir data/${train_set}_reverb"
  utils/copy_data_dir.sh data/${train_set}_reverb data/${train_set}_reverb_hires
  utils/data/perturb_data_dir_volume.sh data/${train_set}_reverb_hires

  steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/${train_set}_reverb_hires exp/make_hires/${train_set}_reverb $mfccdir;
  steps/compute_cmvn_stats.sh data/${train_set}_reverb_hires exp/make_hires/${train_set}_reverb $mfccdir;

  # Remove the small number of utterances that couldn't be extracted for some
  # reason (e.g. too short; no such file).
  utils/fix_data_dir.sh data/${train_set}_reverb_hires;
fi

if [ $stage -le 3 ]; then
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 60 \
    data/${train_set}_reverb_hires $ivector_extractor \
    $train_ivector_dir || exit 1;
fi

for f in $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done

if [ $stage -le 4 ]; then
  if [ -f $ali_dir/ali.1.gz ]; then
    echo "$0: alignments in $ali_dir appear to already exist.  Please either remove them "
    echo " ... or use a later --stage option."
    exit 1
  fi
  echo "$0: aligning with the low-resolution data"
  steps/align_fmllr.sh --nj 100 --cmd "$train_cmd" \
    data/${train_set}_reverb data/lang exp/$gmm $ali_dir || exit 1
fi

if [ $stage -le 5 ]; then
  # Build a tree using our new topology. We know we have alignments for the
  # speed-perturbed data (local/nnet3/run_ivector_common.sh made them), so use
  # those.
  if [ -f $tree_dir/final.mdl ]; then
    echo "$0: $tree_dir/final.mdl already exists, refusing to overwrite it."
    exit 1;
  fi
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --context-opts "--context-width=2 --central-position=1" \
      --cmd "$train_cmd" 6000 data/${train_set}_reverb $lang ${ali_dir} $tree_dir
fi

# Now we generate training lattices
if [ $stage -le 6 ]; then
  nj=$(cat ${ali_dir}/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" data/${train_set}_reverb \
    data/lang exp/$gmm ${lat_dir}
  rm ${lat_dir}/fsts.*.gz # save space
fi

# We remove output layer of trained model since the new one has different number
# of leaves.
if [ $stage -le 7 ]; then
  echo "$0: Create neural net configs using the xconfig parser for";
  echo " generating new layers, that are specific to LibriCSS. These layers ";
  echo " are added to the transferred part of the Librispeech network.";
  num_targets=$(tree-info --print-args=false $tree_dir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)
  tdnnf_opts="l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.75"
  linear_opts="l2-regularize=0.008 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.008"
  output_opts="l2-regularize=0.002"
  mkdir -p $dir
  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  tdnnf-layer name=tdnnf18 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3 input=tdnnf17.batchnorm
  ## adding the layers for chain branch
  linear-component name=prefinal-l dim=256 $linear_opts

  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts big-dim=1536 small-dim=256
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts big-dim=1536 small-dim=256
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --existing-model $src_model_dir/final.mdl \
    --xconfig-file  $dir/configs/network.xconfig  \
    --config-dir $dir/configs/

  # Set the learning-rate-factor to be primary_lr_factor for transferred layers "
  # and adding new layers to them.
  $train_cmd $dir/log/generate_input_mdl.log \
    nnet3-copy --edits="set-learning-rate-factor name=* learning-rate-factor=$primary_lr_factor" $src_model_dir/final.mdl - \| \
      nnet3-init --srand=1 - $dir/configs/final.config $dir/input.raw  || exit 1;
fi

if [ $stage -le 8 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b{09,10,11,12}/$USER/kaldi-data/egs/librispeech-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --trainer.input-model $dir/input.raw \
    --feat.cmvn-opts "--norm-means=true --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.0 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0 --constrained false" \
    --egs.chunk-width $frames_per_eg \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --trainer.num-chunk-per-minibatch 64 \
    --trainer.frames-per-iter 2500000 \
    --trainer.num-epochs 1 \
    --trainer.optimization.num-jobs-initial 6 \
    --trainer.optimization.num-jobs-final 12 \
    --trainer.optimization.initial-effective-lrate 0.0001 \
    --trainer.optimization.final-effective-lrate 0.00001 \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs $remove_egs \
    --use-gpu=wait \
    --feat-dir $train_data_dir \
    --tree-dir $tree_dir \
    --lat-dir $lat_dir \
    --dir $dir  || exit 1;

fi

exit 0;
