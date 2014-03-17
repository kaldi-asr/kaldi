#!/bin/bash

# Copyright 2014  Vimal Manohar
#           2014  Johns Hopkins University(Yenda Trmal)
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


#debugging stuff
echo $0 $@

train_stage=-100
ali_dir=
do_supervised_tuning=true      # Set to true only when using posteriors 
                                # from a single DNN system

. parse_options.sh || exit 1

if [ $# -ne 2 ]; then
  echo "Usage: $0 [options] <untranscribed-data-dir>" 
  echo 
  echo "--ali-dir <alignment_directory>   # Alignment directory"
  echo "--weight-threshold <0.7>          # Frame confidence threshold for frame selection"
  echo "--do-supervised-tuning (default: true) # Train only the last layer at the end."
  echo "e.g.: "
  echo "$0 --decode-dir exp/dnn_sgmm_combine/decode_train_unt.seg --ali-dir exp/tri6_nnet_ali data/train_unt.seg"
  exit 1
fi

set -u

unsup_datadir=$1
unsup_postdir=$2
unsup_dirid=`basename $unsup_datadir`

###############################################################################
#
# Supervised data alignment
#
###############################################################################

if [ -z $ali_dir ] ; then
  # If alignment directory is not done, use exp/tri6_nnet_ali as alignment 
  # directory
  ali_dir=exp/tri6_nnet_ali
fi

if [ ! -f $ali_dir/.done ]; then
  echo "$0: Aligning supervised training data in exp/tri6_nnet_ali"
  [ ! -f exp/tri6_nnet/final.mdl ] && echo "exp/tri6_nnet/final.mdl not found!\nRun run-6-nnet.sh first!" && exit 1
  steps/nnet2/align.sh  --cmd "$train_cmd" \
    --use-gpu no --transform-dir exp/tri5_ali --nj $train_nj \
    data/train data/lang exp/tri6_nnet $ali_dir || exit 1
  touch $ali_dir/.done
fi
exit 0
echo "$0: Using supervised data alignments from $ali_dir"

###############################################################################
#
# Unsupervised data decoding
#
###############################################################################
echo "$0: Using unsupervised data from $unsup_postdir"

for f in $unsup_postdir/weights.1.gz $unsup_postdir/best_path_ali.1.gz; do 
  [ ! -f $f ] && echo "$0: Expecting $f to exist. Probably need to run local/combine_posteriors.sh first." && exit 1
done

###############################################################################
#
# Semi-supervised DNN training
#
###############################################################################

mkdir -p exp/tri6_nnet_semi_supervised

if [ ! -f exp/tri6_nnet_semi_supervised/.egs.done ] ; then
  local/nnet2/get_egs_semi_supervised.sh --cmd "$train_cmd" \
    "${dnn_update_egs_opts[@]}" \
    --transform-dir-sup exp/tri5_ali \
    --transform-dir-unsup exp/tri5/decode_${unsup_dirid} \
    data/train $unsup_datadir data/lang \
    $ali_dir $unsup_postdir exp/tri6_nnet_semi_supervised || exit 1;

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
    --cmd "$train_cmd" "${dnn_gpu_parallel_opts[@]}" \
    --transform-dir exp/tri5_ali \
    --egs-dir exp/tri6_nnet_semi_supervised/egs \
    data/train data/lang $ali_dir exp/tri6_nnet_semi_supervised || exit 1

  touch exp/tri6_nnet_semi_supervised/.done
fi

if $do_supervised_tuning; then
  # Necessary only when semi-supervised DNN is trained using the unsupervised 
  # data that was decoded using only the tri6_nnet system.
  if [ ! -f exp/tri6_nnet_supervised_tuning2/.done ]; then


    num_layers=$( nnet-am-info exp/tri6_nnet_semi_supervised/final.mdl | \
                  grep num-updatable-components | awk '{print $2}' )

    learning_rates=`perl -E 'say "0:" x '$((num_layers-1))`
    learning_rates+="$dnn_final_learning_rate"

    steps/nnet2/update_nnet.sh \
      --learning-rates "$learning_rates" \
      --cmd "$train_cmd" \
      --transform-dir exp/tri5_ali \
      "${dnn_gpu_parallel_opts[@]}" \
      "${dnn_update_parallel_opts[@]}" \
      data/train data/lang $ali_dir \
      exp/tri6_nnet_semi_supervised exp/tri6_nnet_supervised_tuning || exit 1

    touch exp/tri6_nnet_supervised_tuning/.done
  fi
 fi
