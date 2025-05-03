#!/bin/bash

#    This is the standard "tdnn" system, built in nnet3.
# Please see RESULTS_* for examples of command lines invoking this script.


# local/nnet3/run_tdnn.sh --mic sdm1 --use-ihm-ali true

# local/nnet3/run_tdnn.sh --mic ihm --stage 11
# local/nnet3/run_tdnn.sh --mic ihm --train-set train --gmm tri3 --nnet3-affix "" &
#
# local/nnet3/run_tdnn.sh --mic sdm1 --stage 11 --affix _cleaned2 --gmm tri4a_cleaned2 --train-set train_cleaned2 &

# local/nnet3/run_tdnn.sh --use-ihm-ali true --mic sdm1 --train-set train --gmm tri3 --nnet3-affix "" &

# local/nnet3/run_tdnn.sh --use-ihm-ali true --mic mdm8 &

#  local/nnet3/run_tdnn.sh --use-ihm-ali true --mic mdm8 --train-set train --gmm tri3 --nnet3-affix "" &

# this is an example of how you'd train a non-IHM system with the IHM
# alignments.  the --gmm option in this case refers to the IHM gmm that's used
# to get the alignments.
# local/nnet3/run_tdnn.sh --mic sdm1 --use-ihm-ali true --affix _cleaned2 --gmm tri4a --train-set train_cleaned2 &



set -e -o pipefail -u

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=0
mic=ihm
nj=30
min_seg_len=1.55
use_ihm_ali=false
train_set=train_cleaned
gmm=tri3_cleaned  # this is the source gmm-dir for the data-type of interest; it
                  # should have alignments for the specified training data.
ihm_gmm=tri3      # Only relevant if $use_ihm_ali is true, the name of the gmm-dir in
                  # the ihm directory that is to be used for getting alignments.
num_threads_ubm=32
nnet3_affix=_cleaned  # cleanup affix for exp dirs, e.g. _cleaned
tdnn_affix=  #affix for TDNN directory e.g. "a" or "b", in case we change the configuration.

# Options which are not passed through to run_ivector_common.sh
train_stage=-10
splice_indexes="-2,-1,0,1,2 -1,2 -3,3 -7,2 -3,3 0 0"
remove_egs=true
relu_dim=850
num_epochs=3

. cmd.sh
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
                                  --min-seg-len $min_seg_len \
                                  --train-set $train_set \
                                  --gmm $gmm \
                                  --num-threads-ubm $num_threads_ubm \
                                  --nnet3-affix "$nnet3_affix"

# Note: the first stage of the following script is stage 8.
local/nnet3/prepare_lores_feats.sh --stage $stage \
                                   --mic $mic \
                                   --nj $nj \
                                   --min-seg-len $min_seg_len \
                                   --use-ihm-ali $use_ihm_ali \
                                   --train-set $train_set

if $use_ihm_ali; then
  gmm_dir=exp/ihm/${ihm_gmm}
  ali_dir=exp/${mic}/${ihm_gmm}_ali_${train_set}_sp_comb_ihmdata
  lores_train_data_dir=data/$mic/${train_set}_ihmdata_sp_comb
  maybe_ihm="IHM "
  dir=exp/$mic/nnet3${nnet3_affix}/tdnn${tdnn_affix}_sp_ihmali
else
  gmm_dir=exp/${mic}/${gmm}
  ali_dir=exp/${mic}/${gmm}_ali_${train_set}_sp_comb
  lores_train_data_dir=data/$mic/${train_set}_sp_comb
  maybe_ihm=
  dir=exp/$mic/nnet3${nnet3_affix}/tdnn${tdnn_affix}_sp
fi


train_data_dir=data/$mic/${train_set}_sp_hires_comb
train_ivector_dir=exp/$mic/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires_comb
final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-7
graph_dir=$gmm_dir/graph_${LM}



for f in $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp \
     $graph_dir/HCLG.fst; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done



if [ $stage -le 11 ]; then
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
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/ami-$(date +'%m_%d_%H_%M')/s5b/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/tdnn/train.sh --stage $train_stage \
    --num-epochs $num_epochs --num-jobs-initial 2 --num-jobs-final 12 \
    --splice-indexes "$splice_indexes" \
    --feat-type raw \
    --online-ivector-dir ${train_ivector_dir} \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --initial-effective-lrate 0.0015 --final-effective-lrate 0.00015 \
    --cmd "$decode_cmd" \
    --relu-dim "$relu_dim" \
    --remove-egs "$remove_egs" \
    $train_data_dir data/lang $ali_dir $dir
fi

if [ $stage -le 12 ]; then
  rm $dir/.error || true 2>/dev/null
  for decode_set in dev eval; do
      (
      decode_dir=${dir}/decode_${decode_set}
      steps/nnet3/decode.sh --nj $nj --cmd "$decode_cmd" \
          --online-ivector-dir exp/$mic/nnet3${nnet3_affix}/ivectors_${decode_set}_hires \
         $graph_dir data/$mic/${decode_set}_hires $decode_dir
      ) &
  done
  wait;
  if [ -f $dir/.error ]; then
    echo "$0: error detected during decoding"
    exit 1
  fi
fi


exit 0;
