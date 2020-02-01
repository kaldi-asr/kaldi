#!/usr/bin/env bash
# This script extracts mfcc features using mfcc_config and trains ubm model and
# ivector extractor and extracts ivector for train and test.
. ./cmd.sh


stage=1
nnet_affix=_online
ivector_dim=50
mfcc_config=conf/mfcc_hires.conf
use_ivector=true # If false, it skips training ivector extractor and
                 # ivector extraction stages.
online_cmvn_iextractor=false

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

extractor=exp/nnet2${nnet_affix}/extractor

if $use_gpu; then
  if ! cuda-compiled; then
    cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.  Otherwise, call this script with --use-gpu false
EOF
  fi
  parallel_opts="--gpu 1"
  num_threads=1
  minibatch_size=512
else
  # Use 4 nnet jobs just like run_4d_gpu.sh so the results should be
  # almost the same, but this may be a little bit slow.
  num_threads=16
  minibatch_size=128
  parallel_opts="--num-threads $num_threads"
  dir=exp/nnet2${nnet_affix}/nnet
fi

train_set=train
test_set=test
if [ $stage -le 0 ]; then
  echo "$0: creating high-resolution MFCC features."
  mfccdir=data/${train_set}_hires/data

  for datadir in $train_set test; do
    utils/copy_data_dir.sh data/$datadir data/${datadir}_hires

    steps/make_mfcc.sh --nj 30 --mfcc-config $mfcc_config \
      --cmd "$train_cmd" data/${datadir}_hires || exit 1;
    steps/compute_cmvn_stats.sh data/${datadir}_hires
    utils/fix_data_dir.sh data/${datadir}_hires
  done
  train_set=${train_set}_hires
  test_set=${test_set}_hires
fi

if [ ! -f $extractor/final.ie ] && [ $ivector_dim -gt 0 ]; then
  if [ $stage -le 1 ]; then
    mkdir -p exp/nnet2${nnet_affix}
    steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 40 \
      --num-threads 6 --num-frames 200000 \
      data/${train_set} 256 exp/tri3b exp/nnet2${nnet_affix}/diag_ubm
  fi

  if [ $stage -le 2 ]; then
    # use a smaller iVector dim (50) than the default (100) because RM has a very
    # small amount of data.
    steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
      --num-threads 3 --num-processes 2 --ivector-dim $ivector_dim \
      --online-cmvn-iextractor $online_cmvn_iextractor \
     data/${train_set} exp/nnet2${nnet_affix}/diag_ubm $extractor || exit 1;
  fi
fi

if [ $stage -le 3 ] && [ $ivector_dim -gt 0 ]; then
  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 data/${train_set} data/${train_set}_max2

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 40 \
    data/${train_set}_max2 $extractor exp/nnet2${nnet_affix}/ivectors || exit 1;

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 10 \
    data/${test_set} $extractor exp/nnet2${nnet_affix}/ivectors_test || exit 1;
fi
