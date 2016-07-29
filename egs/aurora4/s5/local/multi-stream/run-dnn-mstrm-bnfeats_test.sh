#!/bin/bash

. ./cmd.sh
. ./path.sh 

stage=1
scratch=/export/a06/$USER/tmp/

njdec=30

lang_test=data/lang_test_tgpr_5k

exp="fbank-traps_mstrm_9strms-2BarkPerStrm_CMN_bnfeats_splice5_traps_dct_basis6_iters-per-epoch5" # same as run-dnn-mstrm-bnfeats_train

test_bn=data-fbank-bn-${exp}/test_eval92

. utils/parse_options.sh

test_bn_fmllr=data-fbank-bn-fmllr-${exp}/$(basename $test_bn)


##################################################
#------------------------------------------------------------------------------------
dir=exp/dnn8b_bn-gmm_${exp}
graph=$dir/graph_$(basename $lang_test)
if [ $stage -le 1 ]; then
  if [ ! -d $graph ]; then
    utils/mkgraph.sh ${lang_test} $dir $graph || exit 1
  fi

  steps/decode.sh --nj $njdec --cmd "$decode_cmd" \
    --num-threads 3 --parallel-opts "-pe smp 3" \
    --acwt 0.1 --beam 15.0 --lattice-beam 8.0 \
    $graph $test_bn $dir/decode_$(basename $test_bn) || exit 1
fi

##################################################
#------------------------------------------------------------------------------------
dir=exp/dnn8c_fmllr-gmm_${exp}
graph=$dir/graph_$(basename $lang_test)
if [ $stage -le 2 ]; then
  if [ ! -d $graph ]; then
    utils/mkgraph.sh ${lang_test} $dir $graph || exit 1
  fi

  # Decode,
  steps/decode_fmllr.sh --nj $njdec --cmd "$decode_cmd" \
    --num-threads 3 --parallel-opts "-pe smp 3" \
    --acwt 0.1 --beam 15.0 --lattice-beam 8.0 \
    $graph $test_bn $dir/decode_$(basename $test_bn) || exit 1
fi

##################################################
#------------------------------------------------------------------------------------
# Store the bottleneck-FMLLR features
gmm=exp/dnn8c_fmllr-gmm_${exp} # fmllr-feats, dnn-targets,
if [ $stage -le 3 ]; then
  # Test set
  steps/nnet/make_fmllr_feats.sh --nj $njdec --cmd "$train_cmd" \
     --transform-dir $gmm/decode_$(basename $test_bn) \
     $test_bn_fmllr $test_bn $gmm $test_bn_fmllr/log $test_bn_fmllr/data || exit 1;
fi

##################################################
#------------------------------------------------------------------------------------
# DNN optimized cross-entropy.
dir=exp/dnn8e_pretrain-dbn_dnn_${exp}
graph=$gmm/graph_$(basename $lang_test)
if [ $stage -le 4 ]; then
  # Decode conversational.dev
  steps/nnet/decode.sh --nj $njdec --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.10 \
    --num-threads 3 --parallel-opts "-pe smp 2" --max-mem 150000000 \
    $graph $test_bn_fmllr $dir/decode_$(basename $test_bn_fmllr) || exit 1
fi

##################################################
#------------------------------------------------------------------------------------
# DNN sMBR training
dir=exp/dnn8f_pretrain-dbn_dnn_smbr_${exp}
graph=$gmm/graph_$(basename $lang_test)
if [ $stage -le 5 ]; then
  # Decode conversational.dev
  steps/nnet/decode.sh --nj $njdec --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.10 \
    --scoring-opts "$scoring" --num-threads 2 --parallel-opts "-pe smp 2" --max-mem 150000000 \
    $graph $test_bn_fmllr $dir/decode_$(basename $test_bn_fmllr) || exit 1
fi

exit 0;


