#!/bin/bash

. cmd.sh

( 
 steps/train_nnet_cpu.sh \
   --mix-up 8000 \
   --initial-learning-rate 0.01 --final-learning-rate 0.001 \
   --num-jobs-nnet 16 --num-hidden-layers 4 \
   --num-parameters 8000000 \
   --cmd "$decode_cmd" \
    data/train_100k_nodup data/lang exp/tri5a exp/nnet6a

  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
    --config conf/decode.config --transform-dir exp/tri5a/decode_train_dev \
   exp/tri5a/graph data/train_dev exp/nnet6a/decode_train_dev &

  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
    --config conf/decode.config --transform-dir exp/tri5a/decode_eval2000 \
   exp/tri5a/graph data/eval2000 exp/nnet6a/decode_eval2000 &
)

# Here are the results (copied from RESULTS file)
#exp/nnet6a/decode_train_dev/wer_10:%WER 24.87 [ 12053 / 48460, 1590 ins, 3017 del, 7446 sub ]
#exp/nnet6a/decode_eval2000/score_10/eval2000.ctm.filt.sys:     | Sum/Avg    | 4459  42989 | 77.1   16.0    6.9    2.7   25.6   62.6 |


# Here are some older results when the system had 2k not 4k leaves and ran from a worse SAT
# system.
#exp/nnet5c/decode_train_dev/wer_8:%WER 26.06 [ 12627 / 48460, 1891 ins, 2891 del, 7845 sub ]
#exp/nnet5c/decode_eval2000/score_9/eval2000.ctm.filt.sys:     | Sum/Avg    | 4459  42989 | 76.0   16.8    7.2    2.7   26.7   64.0 |


# the following setup had essentially the same WER as nnet6a, showing the 
# #parameters is about optimal in nnet6a.
# ( 
#  steps/train_nnet_cpu.sh \
#    --mix-up 8000 \
#    --initial-learning-rate 0.01 --final-learning-rate 0.001 \
#    --num-jobs-nnet 16 --num-hidden-layers 4 \
#    --num-parameters 10000000 \
#    --cmd "$decode_cmd" \
#     data/train_100k_nodup data/lang exp/tri5a exp/nnet6b

#   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
#     --config conf/decode.config --transform-dir exp/tri5a/decode_train_dev \
#    exp/tri5a/graph data/train_dev exp/nnet6b/decode_train_dev &

#   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
#     --config conf/decode.config --transform-dir exp/tri5a/decode_eval2000 \
#    exp/tri5a/graph data/eval2000 exp/nnet6b/decode_eval2000 &
# )

# 0.1% worse on train_dev, 0.1% better on eval2000.
#exp/nnet6b/decode_train_dev/wer_9:%WER 24.94 [ 12087 / 48460, 1769 ins, 2788 del, 7530 sub ]
#exp/nnet6b/decode_eval2000/score_10/eval2000.ctm.filt.sys:     | Sum/Avg    | 4459  42989 | 77.2   15.9    6.9    2.7   25.5   62.4 |