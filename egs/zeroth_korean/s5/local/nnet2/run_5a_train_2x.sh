#!/bin/bash

# This is p-norm neural net training, with the "fast" script, on top of adapted
# 40-dimensional features.

# Modified by Lucas Jo 2017 (Altas Guide)


train_stage=-10
use_gpu=true
stage=0

. cmd.sh
. ./path.sh
. utils/parse_options.sh


if $use_gpu; then
  if ! cuda-compiled; then
    cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
  fi
  parallel_opts="--gpu 1"
  num_threads=1
  minibatch_size=512
  dir=exp/nnet5a_train_2x_gpu
else
  # with just 4 jobs this might be a little slow.
  num_threads=16
  parallel_opts="--num-threads $num_threads" 
  minibatch_size=128
  dir=exp/nnet5a_train_2x
fi

. ./cmd.sh
. utils/parse_options.sh

if [ $stage -le 1 ]; then
    if [ ! -f $dir/final.mdl ]; then
      #if [[  $(hostname -f) ==  *.clsp.jhu.edu ]]; then 
      #   # spread the egs over various machines.  will help reduce overload of any
      #   # one machine.
      #   utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/egs/librispeech/s5/$dir/egs/storage $dir/egs/storage
      #fi
    
      steps/nnet2/train_pnorm_fast.sh --stage $train_stage \
       --samples-per-iter 400000 \
       --parallel-opts "$parallel_opts" \
       --num-threads "$num_threads" \
       --minibatch-size "$minibatch_size" \
       --num-jobs-nnet 4  --mix-up 8000 \
       --initial-learning-rate 0.01 --final-learning-rate 0.001 \
       --num-hidden-layers 4 \
       --pnorm-input-dim 2000 --pnorm-output-dim 400 \
       --cmd "$decode_cmd" \
        data/train_2x data/lang exp/tri5b $dir || exit 1
    fi
fi
    
if [ $stage -le 2 ]; then
    for test in test_200 test_noisy_snr20_200; do
      steps/nnet2/decode.sh --nj 20 --cmd "$decode_cmd" \
        --transform-dir exp/tri5b/decode_tgsmall_$test \
        exp/tri5b/graph_tgsmall data/$test $dir/decode_tgsmall_$test || exit 1;
      #steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
      #  data/$test $dir/decode_{tgsmall,tgmed}_$test  || exit 1;
      #steps/lmrescore_const_arpa.sh \
      #  --cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
      #  data/$test $dir/decode_{tgsmall,tglarge}_$test || exit 1;
      steps/lmrescore_const_arpa.sh \
        --cmd "$decode_cmd" data/lang_test_{tgsmall,fglarge} \
        data/$test $dir/decode_{tgsmall,fglarge}_$test || exit 1;
    done
fi

#if [ $stage -le 3 ]; then
#	echo "#### $0: stage 3 #####"
#	# If this setup used PLP features, we'd have to give the option --feature-type plp
#	# to the script below.
#	steps/online/nnet2/prepare_online_decoding.sh data/lang "$dir" ${dir}_online || exit 1;
#fi
#
#if [ $stage -le 4 ]; then
#	echo "#### $0: stage 4 #####"
#	# this version of the decoding treats each utterance separately
#	# without carrying forward speaker information.
#	for test in test_200 test_noisy_snr20_200; do
#		steps/online/nnet2/decode.sh --cmd "$decode_cmd" --nj 20 \
#			--per-utt true exp/tri4b/graph_tgsmall data/$test ${dir}_online/decode_${test}_tgsmall_utt || exit 1;
#		#steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_{tgsmall,tgmed} \
#		#	data/$test ${dir}_online/decode_${test}_{tgsmall,tgmed}_utt  || exit 1;
#		#steps/lmrescore_const_arpa.sh \
#		#	--cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
#		#	data/$test ${dir}_online/decode_${test}_{tgsmall,tglarge}_utt || exit 1;
#		steps/lmrescore_const_arpa.sh \
#			--cmd "$decode_cmd" data/lang_test_{tgsmall,fglarge} \
#			data/$test ${dir}_online/decode_${test}_{tgsmall,fglarge}_utt || exit 1;
#	done
#fi
#
#exit 0;

