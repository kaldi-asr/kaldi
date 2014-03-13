#!/bin/bash

# Copyright 2014  Vimal Manohar
# Apache 2.0

# Run DNN training on untranscribed data
# This uses approx 70 hours of untranscribed data

set -e           #Exit on non-zero return code from any command
set -o pipefail  #Exit if any of the commands in the pipeline will
                 #return non-zero return code

. conf/common_vars.sh || exit 1;
. ./lang.conf || exit 1;
. cmd.sh
. path.sh

# Can provide different neural net structure than for supervised data
. conf/common.semisupervised.limitedLP || exit 1

#debugging stuff
echo $0 $@

train_stage=-100
decode_dir=
ali_dir=
nj=

. parse_options.sh || exit 1

if [ $# -ne 1 ]; then
  echo "Usage: $0 [options] <untranscribed-data-dir>" 
  echo 
  echo "--nj  <num_jobs>                  # Number of parallel jobs for decoding untranscribed data"
  echo "--decode-dir <decode_directory>   # Decode directory with posteriors and best path done"
  echo "--ali-dir <alignment_directory>   # Alignment directory"
  echo "--weight-threshold <0.7>          # Frame confidence threshold for frame selection"
  echo "--do-supervised-tuning (default: true) # Train only the last layer at the end."
  echo "e.g.: "
  echo "$0 --decode-dir exp/dnn_sgmm_combine/decode_train_unt.seg --ali-dir exp/tri6_nnet_ali data/train_unt.seg"
  exit 1
fi

untranscribed_datadir=$1

[ -z $nj ] && nj=$unsup_nj

###############################################################################
#
# Supervised data alignment
#
###############################################################################

if [ -z $alidir ]; then
  # If alignment directory is not done, use exp/tri6_nnet_ali as alignment 
  # directory
  ali_dir=exp/tri6_nnet_ali
fi

if [ ! -f $ali_dir/.done ]; then
  echo "$0: Aligning supervised training data in exp/tri6_nnet_ali"
  [ ! -f exp/tri6_nnet/final.mdl ] && echo "exp/tri6_nnet/final.mdl not found!\nRun run-6-nnet.sh first!" && exit 1
  if [ ! -f exp/tri6_nnet_ali/.done ]; then
    steps/nnet2/align.sh  --cmd "$decode_cmd" \
      --use-gpu no --transform-dir exp/tri5_ali --nj $train_nj \
      data/train data/lang exp/tri6_nnet exp/tri6_nnet_ali || exit 1
    touch exp/tri6_nnet_ali/.done
  fi
else
  echo "$0: Using supervised data alignments from $ali_dir"
fi

###############################################################################
#
# Unsupervised data decoding
#
###############################################################################

datadir=$untranscribed_datadir
dirid=`basename $datadir`

decode=exp/tri5/decode_${dirid}
if [ ! -f ${decode}/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Spawning decoding with SAT models  on" `date`
  echo ---------------------------------------------------------------------
  utils/mkgraph.sh \
    data/lang exp/tri5 exp/tri5/graph |tee exp/tri5/mkgraph.log

  mkdir -p $decode
  #By default, we do not care about the lattices for this step -- we just want the transforms
  #Therefore, we will reduce the beam sizes, to reduce the decoding times
  steps/decode_fmllr_extra.sh --skip-scoring true --beam 10 --lattice-beam 4\
    --nj $nj --cmd "$decode_cmd" "${decode_extra_opts[@]}"\
    exp/tri5/graph ${datadir} ${decode} |tee ${decode}/decode.log
  touch ${decode}/.done
fi

if [ -z $decode_dir ]; then
  decode=exp/tri6_nnet/decode_${dirid}
  [ ! -f exp/tri6_nnet/final.mdl ] && echo "exp/tri6_nnet/final.mdl not found!\nRun run-6-nnet.sh first!" && exit 1
  if [ ! -f $decode/.done ]; then
    echo "$0: Decoding unsupervised data from $untranscribed_datadir using exp/tri6_nnet models"
    mkdir -p $decode
    steps/nnet2/decode.sh --cmd "$decode_cmd" --nj $nj \
      --beam $dnn_beam --lat-beam $dnn_lat_beam \
      --skip-scoring true "${decode_extra_opts[@]}" \
      --transform-dir exp/tri5/decode_${dirid} \
      exp/tri5/graph ${datadir} $decode | tee $decode/decode.log

    touch $decode/.done
  fi
  
  echo "$0: Getting frame posteriors for unsupervised data"
  # Get per-frame weights (posteriors) by best path
  if [ ! -f $decode/.best_path.done ]; then

    $decode_cmd JOB=1:$nj $decode/log/best_path.JOB.log \
      lattice-best-path --acoustic-scale=0.1 \
      "ark,s,cs:gunzip -c $decode/lat.JOB.gz |" \
      ark:/dev/null "ark:| gzip -c > $decode/best_path_ali.JOB.gz" || exit 1
    touch $decode/.best_path.done
  fi

  model=`dirname $decode`/final.mdl
  $decode_cmd JOB=1:$nj $decode/log/get_post.JOB.log \
    lattice-to-post --acoustic-scale=0.1 \
    "ark,s,cs:gunzip -c $decode/lat.JOB.gz|" ark:- \| \
    post-to-pdf-post $model ark,s,cs:- ark:- \| \
    get-post-on-ali ark,s,cs:- "ark,s,cs:gunzip -c $decode/best_path_ali.JOB.gz | ali-to-pdf $model ark,s,cs:- ark:- |" "ark:| gzip -c > $decode/weights.JOB.gz" || exit 1

else
  echo "$0: Using unsupervised data from $decode_dir"
  decode=$decode_dir

  for f in $decode/weights.1.gz $decode/best_path_ali.1.gz; do 
    [ ! -f $f ] && echo "$0: Expecting $f to exist. Probably need to run local/combine_posteriors.sh first." && exit 1
  done
fi

###############################################################################
#
# Semi-supervised DNN training
#
###############################################################################

mkdir -p exp/tri6_nnet_semi_supervised

if [ ! -f exp/tri6_nnet_semi_supervised/.egs.done ]; then
  local/nnet2/get_egs_semi_supervised.sh $spk_vecs_opt \
    "${egs_gpu_opts[@]}" --io-opts "$egs_io_opts" \
    --transform-dir-sup exp/tri5_ali \
    --transform-dir-unsup exp/tri5/decode_${dirid} \
    --weight-threshold $weight_threshold \
    data/train $untranscribed_datadir data/lang \
    $ali_dir $decode exp/tri6_nnet_semi_supervised || exit 1;

  touch exp/tri6_nnet_semi_supervised/.egs.done
fi

if [ ! -f exp/tri6_nnet_semi_supervised/.done ]; then
  steps/nnet2/train_pnorm.sh \
    --stage $train_stage --mix-up $dnn_mixup \
    --initial-learning-rate $dnn_init_learning_rate \
    --final-learning-rate $dnn_final_learning_rate \
    --num-hidden-layers $dnn_num_hidden_layers \
    --pnorm-input-dim $dnn_input_dim \
    --pnorm-output-dim $dnn_output_dim \
    --num-epochs $num_epochs \
    --num-epochs-extra $num_epochs_extra \
    --num-iters-final $num_iters_final \
    --cmd "$train_cmd" "${dnn_gpu_parallel_opts[@]}" \
    --transform-dir exp/tri5_ali \
    --egs-dir exp/tri6_nnet_semi_supervised/egs \
    data/train data/lang $ali_dir exp/tri6_nnet_semi_supervised || exit 1

  touch exp/tri6_nnet_semi_supervised/.done
fi

if $do_supervised_tuning; then
  # Necessary only when semi-supervised DNN is trained using the unsupervised 
  # data that was decoded using only the tri6_nnet system.
  if [ ! -f exp/tri6_nnet_supervised_tuning/.done ]; then
    learning_rates="0"
    for i in `seq 1 $[dnn_num_hidden_layers-1]`; do
      learning_rates="$learning_rates:0"
    done
    learning_rates="$learning_rates:0.0008"

    steps/nnet2/update_nnet.sh \
      --learning-rates $learning_rates \
      --cmd "$train_cmd" \
      "${dnn_update_gpu_parallel_opts[@]}" \
      --num-epochs 2 --num-iters-final 5 \
      --transform-dir exp/tri5_ali \
      data/train data/lang $ali_dir \
      exp/tri6_nnet_semi_supervised exp/tri6_nnet_supervised_tuning || exit 1

    touch exp/tri6_nnet_supervised_tuning/.done
  fi
fi
