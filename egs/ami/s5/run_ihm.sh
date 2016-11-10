#!/bin/bash -u

. ./cmd.sh
. ./path.sh

# IHM - Independent Headset Microphone,
# Please do not change 'mic'! (Identifies both the datasets and experiments: ihm, sdm, mdm)
mic=ihm

stage=0
. utils/parse_options.sh

# Set bash to 'debug' mode, it prints the commands (option '-x') and exits on :
# -e 'error', -u 'undefined variable', -o pipefail 'error in pipeline',
set -euxo pipefail

# Path where AMI gets downloaded (or where locally available):
AMI_DIR=$PWD/wav_db # Default,
case $(hostname -d) in
  fit.vutbr.cz) AMI_DIR=/mnt/matylda5/iveselyk/KALDI_AMI_WAV ;; # BUT,
  clsp.jhu.edu) AMI_DIR=/export/corpora4/ami/amicorpus ;; # JHU,
  cstr.ed.ac.uk) AMI_DIR= ;; # Edinburgh,
esac

[ ! -r data/local/lm/final_lm ] && echo "Please, run 'run_prepare_shared.sh' first!" && exit 1
final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-7

# Download AMI corpus, You need arount 130GB of free space to get whole data ihm+mdm,
# Avoiding re-download, using 'wget --continue ...',
if [ $stage -le 0 ]; then
  [ -e data/local/downloads/wget_${mic}.sh ] && \
    echo "$data/local/downloads/wget_${mic}.sh already exists, better quit than re-download... (use --stage N)" && \
    exit 1
  local/ami_download.sh $mic $AMI_DIR
fi

# Prepare ihm data directories,
if [ $stage -le 1 ]; then
  local/ami_ihm_data_prep.sh $AMI_DIR
  local/ami_ihm_scoring_data_prep.sh $AMI_DIR dev
  local/ami_ihm_scoring_data_prep.sh $AMI_DIR eval
fi

exit 0

# Here starts the normal recipe, which is mostly shared across mic scenarios,
# - for ihm we adapt to speaker by fMLLR,
# - for sdm and mdm we do not adapt for speaker, but for environment only (cmn),

# Feature extraction,
if [ $stage -le 2 ]; then
  for dset in train dev eval; do
    steps/make_mfcc.sh --nj 15 --cmd "$train_cmd" data/$mic/$dset data/$mic/$dset/log data/$mic/$dset/data
    steps/compute_cmvn_stats.sh data/$mic/$dset data/$mic/$dset/log data/$mic/$dset/data
  done
  for dset in train eval dev; do utils/fix_data_dir.sh data/$mic/$dset; done
fi

if [ $stage -le 3 ]; then
  # Taking a subset, now unused, can be handy for quick experiments,
  # Full set 77h, reduced set 10.8h,
  utils/subset_data_dir.sh data/$mic/train 15000 data/$mic/train_15k
fi

# Train systems,
nj=30 # number of parallel jobs,

if [ $stage -le 4 ]; then
  # Mono,
  steps/train_mono.sh --nj $nj --cmd "$train_cmd" --cmvn-opts "--norm-means=true --norm-vars=false" \
    data/$mic/train data/lang exp/$mic/mono
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/$mic/train data/lang exp/$mic/mono exp/$mic/mono_ali

  # Deltas,
  steps/train_deltas.sh --cmd "$train_cmd" --cmvn-opts "--norm-means=true --norm-vars=false" \
    5000 80000 data/$mic/train data/lang exp/$mic/mono_ali exp/$mic/tri1
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/$mic/train data/lang exp/$mic/tri1 exp/$mic/tri1_ali
fi

if [ $stage -le 5 ]; then
  # Deltas again, (full train-set),
  steps/train_deltas.sh --cmd "$train_cmd" --cmvn-opts "--norm-means=true --norm-vars=false" \
    5000 80000 data/$mic/train data/lang exp/$mic/tri1_ali exp/$mic/tri2a
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/$mic/train data/lang exp/$mic/tri2a exp/$mic/tri2_ali
  # Decode,
  graph_dir=exp/$mic/tri2a/graph_${LM}
  $decode_cmd --mem 4G $graph_dir/mkgraph.log \
    utils/mkgraph.sh data/lang_${LM} exp/$mic/tri2a $graph_dir
  steps/decode.sh --nj $nj --cmd "$decode_cmd" --config conf/decode.conf \
    $graph_dir data/$mic/dev exp/$mic/tri2a/decode_dev_${LM}
  steps/decode.sh --nj $nj --cmd "$decode_cmd" --config conf/decode.conf \
    $graph_dir data/$mic/eval exp/$mic/tri2a/decode_eval_${LM}
fi

if [ $stage -le 6 ]; then
  # Train tri3a, which is LDA+MLLT,
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" \
    5000 80000 data/$mic/train data/lang exp/$mic/tri2_ali exp/$mic/tri3a
  # Align with SAT,
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    data/$mic/train data/lang exp/$mic/tri3a exp/$mic/tri3a_ali
  # Decode,
  graph_dir=exp/$mic/tri3a/graph_${LM}
  $decode_cmd --mem 4G $graph_dir/mkgraph.log \
    utils/mkgraph.sh data/lang_${LM} exp/$mic/tri3a $graph_dir
  steps/decode.sh --nj $nj --cmd "$decode_cmd" --config conf/decode.conf \
    $graph_dir data/$mic/dev exp/$mic/tri3a/decode_dev_${LM}
  steps/decode.sh --nj $nj --cmd "$decode_cmd" --config conf/decode.conf \
    $graph_dir data/$mic/eval exp/$mic/tri3a/decode_eval_${LM}
fi

if [ $stage -le 7 ]; then
  # Train tri4a, which is LDA+MLLT+SAT,
  steps/train_sat.sh  --cmd "$train_cmd" \
    5000 80000 data/$mic/train data/lang exp/$mic/tri3a_ali exp/$mic/tri4a
  # Decode,
  graph_dir=exp/$mic/tri4a/graph_${LM}
  $decode_cmd --mem 4G $graph_dir/mkgraph.log \
    utils/mkgraph.sh data/lang_${LM} exp/$mic/tri4a $graph_dir
  steps/decode_fmllr.sh --nj $nj --cmd "$decode_cmd"  --config conf/decode.conf \
    $graph_dir data/$mic/dev exp/$mic/tri4a/decode_dev_${LM}
  steps/decode_fmllr.sh --nj $nj --cmd "$decode_cmd" --config conf/decode.conf \
    $graph_dir data/$mic/eval exp/$mic/tri4a/decode_eval_${LM}
fi

nj_mmi=80
if [ $stage -le 8 ]; then
  # Align,
  steps/align_fmllr.sh --nj $nj_mmi --cmd "$train_cmd" \
    data/$mic/train data/lang exp/$mic/tri4a exp/$mic/tri4a_ali
fi

# At this point you can already run the DNN script with fMLLR features:
# local/nnet/run_dnn.sh
# exit 0

if [ $stage -le 9 ]; then
  # MMI training starting from the LDA+MLLT+SAT systems,
  steps/make_denlats.sh --nj $nj_mmi --cmd "$decode_cmd" --config conf/decode.conf \
    --transform-dir exp/$mic/tri4a_ali \
    data/$mic/train data/lang exp/$mic/tri4a exp/$mic/tri4a_denlats
fi

# 4 iterations of MMI seems to work well overall. The number of iterations is
# used as an explicit argument even though train_mmi.sh will use 4 iterations by
# default.
if [ $stage -le 10 ]; then
  num_mmi_iters=4
  steps/train_mmi.sh --cmd "$train_cmd" --boost 0.1 --num-iters $num_mmi_iters \
    data/$mic/train data/lang exp/$mic/tri4a_ali exp/$mic/tri4a_denlats \
    exp/$mic/tri4a_mmi_b0.1
fi
if [ $stage -le 11 ]; then
  # Decode,
  graph_dir=exp/$mic/tri4a/graph_${LM}
  for i in 4 3 2 1; do
    decode_dir=exp/$mic/tri4a_mmi_b0.1/decode_dev_${i}.mdl_${LM}
    steps/decode.sh --nj $nj --cmd "$decode_cmd" --config conf/decode.conf \
      --transform-dir exp/$mic/tri4a/decode_dev_${LM} --iter $i \
      $graph_dir data/$mic/dev $decode_dir
    decode_dir=exp/$mic/tri4a_mmi_b0.1/decode_eval_${i}.mdl_${LM}
    steps/decode.sh --nj $nj --cmd "$decode_cmd"  --config conf/decode.conf \
      --transform-dir exp/$mic/tri4a/decode_eval_${LM} --iter $i \
      $graph_dir data/$mic/eval $decode_dir
  done
fi

# DNN training. This script is based on egs/swbd/s5b/local/run_dnn.sh
# Some of them would be out of date.
if [ $stage -le 12 ]; then
  local/nnet/run_dnn.sh $mic
fi

# nnet3 systems
if [ $stage -le 13 ]; then

  # tdnn model + xent training
  local/nnet3/run_tdnn.sh --mic $mic

  # lstm model + xent training
  local/nnet3/run_lstm.sh --mic $mic \
    --stage 10 --use-sat-alignments true

  # blstm model + xent training
  local/nnet3/run_blstm.sh --mic $mic \
    --stage 10 --chunk-right-context 20

  # tdnn model + chain training
  local/chain/run_tdnn_ami_5.sh  --mic $mic --affix msl1.5_45wer

fi

echo "Done."
exit 0;

# Older nnet2 scripts. They are still kept here
# as we have not yet committed sMBR training scripts for AMI in nnet3
#if [ $stage -le 13 ]; then
#  local/online/run_nnet2_ms_perturbed.sh \
#    --mic $mic \
#    --hidden-dim 950 \
#    --splice-indexes "layer0/-2:-1:0:1:2 layer1/-1:2 layer2/-3:3 layer3/-7:2 layer4/-3:3" \
#    --use-sat-alignments true
#
#  local/online/run_nnet2_ms_sp_disc.sh  \
#    --mic $mic  \
#    --gmm-dir exp/$mic/tri4a \
#    --srcdir exp/$mic/nnet2_online/nnet_ms_sp
#fi
