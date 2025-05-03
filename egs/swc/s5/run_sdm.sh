#!/bin/bash -u

. ./cmd.sh
. ./path.sh

echo "Starting time:"
date

# SDM - Single Distant Microphone,
# Please do not change 'mic'! (Identifies both the datasets and experiments: ihm, sdm, mdm)
mic=sdm

stage=0
. utils/parse_options.sh

# Set bash to 'debug' mode, it prints the commands (option '-x') and exits on : 
# -e 'error', -u 'undefined variable', -o pipefail 'error in pipeline',
set -euxo pipefail

# Path where SWC gets downloaded (or where locally available):
SWCDIR=$PWD/wav_db # Default, 
case $(hostname -d) in
  minigrid.dcs.shef.ac.uk) SWCDIR=/share/spandh.ami1/usr/yulan/sw/kaldi/dev/kaldi-trunk/egs/swc/swc ;; # UoS
esac

# Prepare ihm data directories, by default we choose the easiest dataset
MODE='SA1'	# Other options: 'SA2', 'AD1', 'AD2'
if [ $stage -le 1 ]; then
  local/swc_sdm_data_prep.sh  $SWCDIR  $MODE
fi



# LM downloading
[ ! -r data/local/lm/final_lm ] && echo "Please, run 'run_prepare_shared.sh' first!" && exit 1
final_lm=`cat data/local/lm/final_lm`
# LM=$final_lm
LM=${final_lm}.pr1-7		# Use pruned LM to save memory


# Here starts the normal recipe, which is mostly shared across mic scenarios,
# - for ihm we adapt to speaker by fMLLR,
# - for sdm and mdm we do not adapt for speaker, but for environment only (cmn),

# Feature extraction,
if [ $stage -le 2 ]; then
  for dset in train dev eval; do
    fd=data/$mic/$MODE/$dset
    steps/make_mfcc.sh --nj 15 --cmd "$train_cmd"  $fd  $fd/log  $fd/data
    steps/compute_cmvn_stats.sh  $fd  $fd/log  $fd/data
  done
  for dset in train eval dev; do utils/fix_data_dir.sh $fd; done
fi

if [ $stage -le 3 ]; then
  # Taking a subset, now unused, can be handy for quick experiments,
  # All data: 24.6h; full train set in SA1 mode: 13.5h; reduced set ~9h,
  utils/subset_data_dir.sh data/$mic/$MODE/train  15000  data/$mic/$MODE/train_15k
fi


# Train systems,
nj=20 # number of parallel jobs,
nj_dev=$(cat data/$mic/$MODE/dev/spk2utt | wc -l)
nj_eval=$(cat data/$mic/$MODE/eval/spk2utt | wc -l)

if [ $stage -le 4 ]; then
  # Mono,
  fd=data/$mic/$MODE
  steps/train_mono.sh --nj $nj --cmd "$train_cmd" --cmvn-opts "--norm-means=true --norm-vars=false" \
    $fd/train  data/lang  exp/$mic/$MODE/mono
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    $fd/train  data/lang  exp/$mic/$MODE/mono exp/$mic/$MODE/mono_ali

  # Deltas,
  steps/train_deltas.sh --cmd "$train_cmd" --cmvn-opts "--norm-means=true --norm-vars=false" \
    5000 80000  $fd/train  data/lang  exp/$mic/$MODE/mono_ali  exp/$mic/$MODE/tri1
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    $fd/train  data/lang  exp/$mic/$MODE/tri1 exp/$mic/$MODE/tri1_ali
fi

if [ $stage -le 5 ]; then
  # Deltas again, (full train-set),
  fd=data/$mic/$MODE
  steps/train_deltas.sh --cmd "$train_cmd" --cmvn-opts "--norm-means=true --norm-vars=false" \
    5000 80000  $fd/train  data/lang  exp/$mic/$MODE/tri1_ali  exp/$mic/$MODE/tri2a
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    $fd/train  data/lang  exp/$mic/$MODE/tri2a  exp/$mic/$MODE/tri2_ali
  # Decode,
  graph_dir=exp/$mic/$MODE/tri2a/graph_${LM}
  $highmem_cmd $graph_dir/mkgraph.log \
    utils/mkgraph.sh data/lang_${LM} exp/$mic/$MODE/tri2a $graph_dir

# [DEBUG] will run scoring later separately
  steps/decode.sh --nj $nj_dev --cmd "$decode_large_cmd" --config conf/decode.conf \
    --skip-scoring true \
    $graph_dir  $fd/dev  exp/$mic/$MODE/tri2a/decode_dev_${LM}
  steps/decode.sh --nj $nj_eval --cmd "$decode_large_cmd" --config conf/decode.conf \
    --skip-scoring true \
    $graph_dir  $fd/eval  exp/$mic/$MODE/tri2a/decode_eval_${LM}
fi


# THE TARGET LDA+MLLT+SAT+BMMI PART GOES HERE:

if [ $stage -le 6 ]; then
  # Train tri3a, which is LDA+MLLT,
  fd=data/$mic/$MODE

  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" \
    5000 80000  $fd/train  data/lang  exp/$mic/$MODE/tri2_ali  exp/$mic/$MODE/tri3a
  # Align with SAT,
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    $fd/train  data/lang  exp/$mic/$MODE/tri3a  exp/$mic/$MODE/tri3a_ali
  # Decode,
  graph_dir=exp/$mic/$MODE/tri3a/graph_${LM}
  $highmem_cmd $graph_dir/mkgraph.log \
    utils/mkgraph.sh  data/lang_${LM}  exp/$mic/$MODE/tri3a  $graph_dir
  steps/decode.sh --nj $nj_dev --cmd "$decode_large_cmd" --config conf/decode.conf \
    --skip-scoring true \
    $graph_dir  $fd/dev  exp/$mic/$MODE/tri3a/decode_dev_${LM}
  steps/decode.sh --nj $nj_eval --cmd "$decode_large_cmd" --config conf/decode.conf \
    --skip-scoring true \ 
    $graph_dir  $fd/eval  exp/$mic/$MODE/tri3a/decode_eval_${LM}
fi


if [ $stage -le 7 ]; then
  # Train tri4a, which is LDA+MLLT+SAT,
  fd=data/$mic/$MODE

  steps/train_sat.sh  --cmd "$train_cmd" \
    5000 80000  $fd/train  data/lang  exp/$mic/$MODE/tri3a_ali  exp/$mic/$MODE/tri4a
  # Decode,  
  graph_dir=exp/$mic/$MODE/tri4a/graph_${LM}
  $highmem_cmd $graph_dir/mkgraph.log \
    utils/mkgraph.sh  data/lang_${LM}  exp/$mic/$MODE/tri4a  $graph_dir
  steps/decode_fmllr.sh --nj $nj_dev --cmd "$decode_large_cmd"  --config conf/decode.conf \
    --skip-scoring true \
    $graph_dir  $fd/dev  exp/$mic/$MODE/tri4a/decode_dev_${LM}
  steps/decode_fmllr.sh --nj $nj_eval --cmd "$decode_large_cmd" --config conf/decode.conf \
    --skip-scoring true \ 
    $graph_dir  $fd/eval  exp/$mic/$MODE/tri4a/decode_eval_${LM}
fi


nj_mmi=22
if [ $stage -le 8 ]; then
  # Align,
  fd=data/$mic/$MODE
  steps/align_fmllr.sh --nj $nj_mmi --cmd "$train_cmd" \
    $fd/train  data/lang  exp/$mic/$MODE/tri4a  exp/$mic/$MODE/tri4a_ali
fi


# At this point you can already run the DNN script with fMLLR features:
# local/nnet/run_dnn.sh
# exit 0


# Fix folder path so that the external scripts just works
for x in 'train'  'dev'  'eval'
do
  if [ -d $PWD/data/$mic/$MODE/$x ] ; then
    if [ ! -d $PWD/data/$mic/$x ] ; then
      ln -s $PWD/data/$mic/$MODE/$x $PWD/data/$mic/$x
    else
      echo "[WARNING] file exist already: $PWD/data/$mic/$x"
    fi
  fi
done

for subfd in 'tri3a'  'tri3a_ali'  'tri4a'  'tri4a_ali'
do
  if [ ! -d $PWD/exp/$mic/$subfd ] ; then
    ln -s ./$MODE/$subfd   exp/$mic/
  else
    echo "[WARNING] file exist already: exp/$mic/$subfd "
  fi
done




if [ $stage -le 9 ]; then
  # MMI training starting from the LDA+MLLT+SAT systems,
  fd=data/$mic/$MODE/
  steps/make_denlats.sh --nj $nj_mmi --cmd "$decode_large_cmd" --config conf/decode.conf \
    --transform-dir exp/$mic/tri4a_ali \
    $fd/train  data/lang  exp/$mic/$MODE/tri4a  exp/$mic/$MODE/tri4a_denlats
fi


# 4 iterations of MMI seems to work well overall. The number of iterations is
# used as an explicit argument even though train_mmi.sh will use 4 iterations by
# default.
if [ $stage -le 10 ]; then
  num_mmi_iters=4
  fd=data/$mic/$MODE/
  steps/train_mmi.sh --cmd "$train_cmd" --boost 0.1 --num-iters $num_mmi_iters \
    $fd/train  data/lang  exp/$mic/$MODE/tri4a_ali  exp/$mic/$MODE/tri4a_denlats \
    exp/$mic/$MODE/tri4a_mmi_b0.1
fi
if [ $stage -le 11 ]; then
  # Decode,
  fd=data/$mic/$MODE/
  graph_dir=exp/$mic/$MODE/tri4a/graph_${LM}
  for i in 4 3 2 1; do
    decode_dir=exp/$mic/$MODE/tri4a_mmi_b0.1/decode_dev_${i}.mdl_${LM}
    steps/decode.sh --nj $nj_dev --cmd "$decode_large_cmd" --config conf/decode.conf \
      --skip-scoring true \
      --transform-dir exp/$mic/$MODE/tri4a/decode_dev_${LM} --iter $i \
      $graph_dir  $fd/dev $decode_dir
    decode_dir=exp/$mic/$MODE/tri4a_mmi_b0.1/decode_eval_${i}.mdl_${LM}
    steps/decode.sh --nj $nj_eval --cmd "$decode_large_cmd"  --config conf/decode.conf \
      --skip-scoring true \
      --transform-dir exp/$mic/$MODE/tri4a/decode_eval_${LM} --iter $i \
      $graph_dir  $fd/eval $decode_dir
  done
fi


# DNN training. This script is based on egs/swbd/s5b/local/run_dnn.sh
# Some of them would be out of date.
if [ $stage -le 12 ]; then
#  local/nnet/run_dnn.sh $mic
  local/nnet/run_dnn_lda_mllt.sh $mic
fi

# TDNN training.
if [ $stage -le 13 ]; then
  local/online/run_nnet2_ms_perturbed.sh \
    --mic $mic \
    --hidden-dim 950 \
    --splice-indexes "layer0/-2:-1:0:1:2 layer1/-1:2 layer2/-3:3 layer3/-7:2 layer4/-3:3" \
    --use-sat-alignments true

  local/online/run_nnet2_ms_sp_disc.sh  \
    --mic $mic  \
    --gmm-dir exp/$mic/tri4a \
    --srcdir exp/$mic/nnet2_online/nnet_ms_sp
fi

echo "Done."

exit;

# Decode,
graph_dir=exp/$mic/tri4a/graph_${LM}
$highmem_cmd $graph_dir/mkgraph.log \
  utils/mkgraph.sh data/lang_${LM} exp/$mic/tri4a $graph_dir
steps/decode_fmllr.sh --nj $nj_dev --cmd "$decode_large_cmd" --config conf/decode.conf \
  --skip-scoring true \
  $graph_dir data/$mic/dev exp/$mic/tri4a/decode_dev_${LM}
steps/decode_fmllr.sh --nj $nj_eval --cmd "$decode_large_cmd" --config conf/decode.conf \
  --skip-scoring true \
  $graph_dir data/$mic/eval exp/$mic/tri4a/decode_eval_${LM}








