#!/bin/bash

. ./cmd.sh
. ./path.sh
set -e

if [ -e data/train_dev ] ;then
    dev_set=train_dev
fi

#:<<"#SKIP"

# This runs on all the data data/train_nodup; the triphone baseline, tri4 is
# also trained on that set.

. utils/parse_options.sh || exit 1;

if [ ! -f exp/ubm5/final.ubm ]; then
  steps/train_ubm.sh --cmd "$train_cmd" 1400 data/train_nodup data/lang \
    exp/tri4_ali_nodup exp/ubm5 || exit 1;
fi 

# steps/train_sgmm2.sh --cmd "$train_cmd" \
steps/train_sgmm2_group.sh --cmd "$train_cmd" \
  18000 60000 data/train_nodup data/lang exp/tri4_ali_nodup \
  exp/ubm5/final.ubm exp/sgmm2_5 || exit 1;



  graph_dir=exp/sgmm2_5/graph_csj_tg
  $train_cmd $graph_dir/mkgraph.log \
    utils/mkgraph.sh data/lang_csj_tg exp/sgmm2_5 $graph_dir
for eval_num in eval1 eval2 eval3 $dev_set ; do
  steps/decode_sgmm2.sh --nj 10 \
    --cmd "$decode_cmd" --config conf/decode.config \
    --transform-dir exp/tri4/decode_${eval_num}_csj $graph_dir \
    data/$eval_num exp/sgmm2_5/decode_${eval_num}_csj
done
wait

# Now discriminatively train the SGMM system on data/train_nodup data.
steps/align_sgmm2.sh --nj 10 --cmd "$train_cmd" \
  --transform-dir exp/tri4_ali_nodup \
  --use-graphs true --use-gselect true \
  data/train_nodup data/lang exp/sgmm2_5 exp/sgmm2_5_ali_nodup

# Took the beam down to 10 to get acceptable decoding speed.
steps/make_denlats_sgmm2.sh --nj 10 --sub-split 30 --num-threads 6 \
  --beam 9.0 --lattice-beam 6 --cmd "$decode_cmd" \
  --transform-dir exp/tri4_ali_nodup \
  data/train_nodup data/lang exp/sgmm2_5_ali_nodup exp/sgmm2_5_denlats_nodup

steps/train_mmi_sgmm2.sh --cmd "$decode_cmd" \
  --transform-dir exp/tri4_ali_nodup --boost 0.1 \
  data/train_nodup data/lang exp/sgmm2_5_ali_nodup \
  exp/sgmm2_5_denlats_nodup exp/sgmm2_5_mmi_b0.1

#SKIP

for eval_num in eval1 eval2 eval3 $dev_set ; do
    for iter in 1 2 3 4; do
	steps/decode_sgmm2_rescore.sh --cmd "$decode_cmd" --iter $iter \
	    --transform-dir exp/tri4/decode_${eval_num}_csj \
	    data/lang_csj_tg data/$eval_num \
	    exp/sgmm2_5/decode_${eval_num}_csj \
	    exp/sgmm2_5_mmi_b0.1/decode_${eval_num}_csj_it$iter
    done
done
wait

# The following is the results of sgmm2.
# We use only academic lecture speech data (986) for AM training.
:<<EOF
=== evaluation set 1 ===
%WER 14.45 [ 3996 / 27651, 552 ins, 1021 del, 2423 sub ] exp/sgmm2_5/decode_eval1_csj/wer_12_0.0
%WER 13.65 [ 3774 / 27651, 486 ins, 1029 del, 2259 sub ] exp/sgmm2_5_mmi_b0.1/decode_eval1_csj_it1/wer_12_0.5
%WER 13.40 [ 3706 / 27651, 501 ins, 953 del, 2252 sub ] exp/sgmm2_5_mmi_b0.1/decode_eval1_csj_it2/wer_12_0.5
%WER 13.26 [ 3667 / 27651, 560 ins, 852 del, 2255 sub ] exp/sgmm2_5_mmi_b0.1/decode_eval1_csj_it3/wer_11_0.5
%WER 13.25 [ 3663 / 27651, 490 ins, 948 del, 2225 sub ] exp/sgmm2_5_mmi_b0.1/decode_eval1_csj_it4/wer_11_1.0
=== evaluation set 2 ===
%WER 12.28 [ 3490 / 28424, 486 ins, 945 del, 2059 sub ] exp/sgmm2_5/decode_eval2_csj/wer_12_0.0
%WER 11.60 [ 3296 / 28424, 555 ins, 764 del, 1977 sub ] exp/sgmm2_5_mmi_b0.1/decode_eval2_csj_it1/wer_11_0.0
%WER 11.38 [ 3234 / 28424, 484 ins, 789 del, 1961 sub ] exp/sgmm2_5_mmi_b0.1/decode_eval2_csj_it2/wer_10_0.5
%WER 11.32 [ 3219 / 28424, 420 ins, 858 del, 1941 sub ] exp/sgmm2_5_mmi_b0.1/decode_eval2_csj_it3/wer_10_1.0
%WER 11.27 [ 3204 / 28424, 395 ins, 881 del, 1928 sub ] exp/sgmm2_5_mmi_b0.1/decode_eval2_csj_it4/wer_11_1.0
=== evaluation set 3 ===
%WER 15.32 [ 2801 / 18283, 403 ins, 756 del, 1642 sub ] exp/sgmm2_5/decode_eval3_csj/wer_14_0.0
%WER 14.37 [ 2628 / 18283, 393 ins, 631 del, 1604 sub ] exp/sgmm2_5_mmi_b0.1/decode_eval3_csj_it1/wer_11_0.5
%WER 14.45 [ 2642 / 18283, 438 ins, 601 del, 1603 sub ] exp/sgmm2_5_mmi_b0.1/decode_eval3_csj_it2/wer_11_0.5
%WER 14.55 [ 2661 / 18283, 467 ins, 588 del, 1606 sub ] exp/sgmm2_5_mmi_b0.1/decode_eval3_csj_it3/wer_11_0.5
%WER 14.60 [ 2669 / 18283, 498 ins, 561 del, 1610 sub ] exp/sgmm2_5_mmi_b0.1/decode_eval3_csj_it4/wer_11_0.5
EOF
