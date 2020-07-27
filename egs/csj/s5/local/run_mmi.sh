#!/usr/bin/env bash

. cmd.sh

set -e

dev_set=
# Thish runs on all the data data/train_nodup; the triphone baseline, tri4 is
# also trained on that set
if [ -e data/train_dev ]; then
  dev_set=train_dev
fi
### MMI training

# MMI training starting from the LDA+MLLT+SAT systems with the entire train_nodup (233hr)
steps/make_denlats.sh --nj 50 --cmd "$decode_cmd" --config conf/decode.config \
  --transform-dir exp/tri4_ali_nodup \
  data/train_nodup data/lang exp/tri4 exp/tri4_denlats_nodup

# 4 iterations of MMI seems to work well overall. The number of iterations is
# used as an explicit argument even though train_mmi.sh will use 4 iterations by
# default.
num_mmi_iters=4
steps/train_mmi.sh --cmd "$decode_cmd" --boost 0.1 --num-iters $num_mmi_iters \
  data/train_nodup data/lang exp/tri4_{ali,denlats}_nodup exp/tri4_mmi_b0.1

for eval_num in eval1 eval2 eval3 $dev_set ; do
    for iter in 1 2 3 4; do
	graph_dir=exp/tri4/graph_csj_tg
	decode_dir=exp/tri4_mmi_b0.1/decode_${eval_num}_${iter}.mdl_csj

	steps/decode.sh --nj 10 --cmd "$decode_cmd" --config conf/decode.config \
	    --iter $iter --transform-dir exp/tri4/decode_${eval_num}_csj \
	    $graph_dir data/$eval_num $decode_dir
    done
done
wait

# Now do fMMI+MMI training
steps/train_diag_ubm.sh --silence-weight 0.5 --nj 50 --cmd "$train_cmd" \
  700 data/train_nodup data/lang exp/tri4_ali_nodup exp/tri4_dubm

steps/train_mmi_fmmi.sh --learning-rate 0.005 --boost 0.1 --cmd "$train_cmd" \
  data/train_nodup data/lang exp/tri4_ali_nodup exp/tri4_dubm \
  exp/tri4_denlats_nodup exp/tri4_fmmi_b0.1

for eval_num in eval1 eval2 eval3 $dev_set ; do
    for iter in 4 5 6 7 8; do
	graph_dir=exp/tri4/graph_csj_tg
	decode_dir=exp/tri4_fmmi_b0.1/decode_${eval_num}_it${iter}_csj

	steps/decode_fmmi.sh --nj 10 --cmd "$decode_cmd" --iter $iter \
	    --transform-dir exp/tri4/decode_${eval_num}_csj \
	    --config conf/decode.config $graph_dir data/${eval_num} $decode_dir
    done
done
wait

