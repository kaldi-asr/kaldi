#!/bin/bash

# Copyright 2014  Pegah Ghahremani
#           2014  Johns Hopkins (Yenda Trmal)

# Apache 2.0

# This is really an alternative path to the BNF-SGMM, 
# where we train a DNN instead of an SGMM.


. conf/common_vars.sh
. ./lang.conf
[ -f local.conf ] && . ./local.conf

set -e
set -o pipefail
set -u

semisupervised=true
dnn_train_stage=-100
unsup_string=

. ./utils/parse_options.sh

if [ $babel_type == "full" ] && $semisupervised; then
  echo "Error: Using unsupervised training for fullLP is meaningless, use semisupervised=false "
  exit 1
fi

if [ -z "$unsup_string" ]; then
  if $semisupervised ; then
    unsup_string="_semisup"
  else
    unsup_string=""  #" ": supervised training, _semi_supervised: unsupervised BNF training
  fi
fi
exp_dir=exp_bnf${unsup_string}
data_bnf_dir=data_bnf${unsup_string}

if [ ! -e $exp_dir/tri6/.done ]; then
  echo "$0: $exp_dir/tri6/.done does not exist"
  echo "$0: this script needs to be run _AFTER_ the script run-2b-bnf.sh"
  echo "$0: with the appropriate parameters -- mostly the same to the parameters"
  echo "$0: of this script"
  exit 1
fi

# We create an alignment with a lot of jobs, because the LDA accumulation
# when training the pnorm network will be slow, due to the large dimension.
if [ ! $exp_dir/tri6_ali_50/.done -nt $exp_dir/tri6/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Aligning fMLLR system with 50 jobs"
  echo ---------------------------------------------------------------------
  steps/align_fmllr.sh \
    --boost-silence $boost_sil --nj $train_nj --cmd "$train_cmd" \
    $data_bnf_dir/train data/lang $exp_dir/tri6 $exp_dir/tri6_ali_50
  touch $exp_dir/tri6_ali_50/.done
fi


if [ ! $exp_dir/tri7_nnet/.done -nt $exp_dir/tri6_ali_50/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting hybrid system building (over bottleneck features)"
  echo ---------------------------------------------------------------------
  steps/nnet2/train_pnorm.sh \
    --stage $dnn_train_stage --mix-up $dnn_mixup \
    --initial-learning-rate $dnn_init_learning_rate \
    --final-learning-rate $dnn_final_learning_rate \
    --num-hidden-layers $dnn_num_hidden_layers \
    --pnorm-input-dim $dnn_input_dim \
    --pnorm-output-dim $dnn_output_dim \
    --egs-opts "--feat-type raw" --lda-opts "--feat-type raw --lda-dim $dnn_output_dim" --splice-width 5 \
     "${dnn_gpu_parallel_opts[@]}" --cmd "$train_cmd" \
    $data_bnf_dir/train data/lang $exp_dir/tri6_ali_50 $exp_dir/tri7_nnet || exit 1 

  touch $exp_dir/tri7_nnet/.done 
fi


echo ---------------------------------------------------------------------
echo "Finished successfully on" `date`
echo "To decode a data-set, use run-4b-anydecode-bnf.sh"
echo ---------------------------------------------------------------------

exit 0
