#!/bin/bash

lang_suffix=

echo "$0 $@"  # Print the command line for logging
. utils/parse_options.sh || exit 1;

. cmd.sh
featdir=mfcc_vtln
num_leaves=2500
num_gauss=15000

# train linear vtln
steps/train_lvtln.sh --cmd "$train_cmd"  $num_leaves $num_gauss \
  data/train_si84 data/lang${lang_suffix} exp/tri2b exp/tri2c || exit 1

mkdir -p data/train_si84_vtln
cp -r data/train_si84/* data/train_si84_vtln || exit 1
cp exp/tri2c/final.warp data/train_si84_vtln/spk2warp || exit 1

utils/mkgraph.sh data/lang${lang_suffix}_test_bg_5k \
  exp/tri2c exp/tri2c/graph${lang_suffix}_bg_5k || exit 1;
utils/mkgraph.sh data/lang${lang_suffix}_test_tgpr \
  exp/tri2c exp/tri2c/graph${lang_suffix}_tgpr || exit 1;

for t in eval93 dev93 eval92; do 
  nj=10
  [ $t == eval92 ] && nj=8
  steps/decode_lvtln.sh --nj $nj --cmd "$decode_cmd" \
    exp/tri2c/graph${lang_suffix}_bg_5k data/test_$t \
    exp/tri2c/decode${lang_suffix}_${t}_bg_5k || exit 1
  mkdir -p data/test_${t}_vtln
  cp -r data/test_$t/* data/test_${t}_vtln || exit 1
  cp exp/tri2c/decode${lang_suffix}_${t}_bg_5k/final.warp \
    data/test_${t}_vtln/spk2warp || exit 1
done


for x in test_eval92 test_eval93 test_dev93 train_si84; do
  steps/make_mfcc.sh --nj 20 --cmd "$train_cmd" data/${x}_vtln exp/make_mfcc/${x}_vtln ${featdir} || exit 1
  steps/compute_cmvn_stats.sh data/${x}_vtln exp/make_mfcc/${x}_vtln ${featdir} || exit 1
  utils/fix_data_dir.sh data/${x}_vtln || exit 1 # remove segments with problems
done

steps/align_si.sh --nj 10 --cmd "$train_cmd" \
  data/train_si84_vtln data/lang${lang_suffix} \
  exp/tri2c exp/tri2c_ali_si84 || exit 1

steps/train_lda_mllt.sh --cmd "$train_cmd" \
  --splice-opts "--left-context=3 --right-context=3" \
  2500 15000 data/train_si84_vtln data/lang${lang_suffix} \
  exp/tri2c_ali_si84 exp/tri2d  || exit 1

(
utils/mkgraph.sh data/lang${lang_suffix}_test_tgpr \
  exp/tri2d exp/tri2d/graph${lang_suffix}_tgpr || exit 1;
steps/decode.sh --nj 10 --cmd "$decode_cmd" \
  exp/tri2d/graph${lang_suffix}_tgpr data/test_dev93_vtln \
  exp/tri2d/decode${lang_suffix}_tgpr_dev93 || exit 1;
steps/decode.sh --nj 8 --cmd "$decode_cmd" \
  exp/tri2d/graph${lang_suffix}_tgpr data/test_eval92_vtln \
  exp/tri2d/decode${lang_suffix}_tgpr_eval92 || exit 1;
) &

steps/align_si.sh --nj 10 --cmd "$train_cmd" \
  data/train_si84_vtln data/lang${lang_suffix} \
  exp/tri2d exp/tri2d_ali_si84 || exit 1

# From 2d system, train 3c which is LDA + MLLT + SAT.
steps/train_sat.sh --cmd "$train_cmd" \
  2500 15000 data/train_si84_vtln data/lang${lang_suffix} \
  exp/tri2d_ali_si84 exp/tri3c || exit 1;

(
utils/mkgraph.sh data/lang${lang_suffix}_test_tgpr \
  exp/tri3c exp/tri3c/graph${lang_suffix}_tgpr || exit 1;
steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" \
  exp/tri3c/graph${lang_suffix}_tgpr data/test_dev93_vtln \
  exp/tri3c/decode${lang_suffix}_tgpr_dev93 || exit 1;
steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" \
  exp/tri3c/graph${lang_suffix}_tgpr data/test_eval93_vtln \
  exp/tri3c/decode${lang_suffix}_tgpr_eval93 || exit 1;
steps/decode_fmllr.sh --nj 8 --cmd "$decode_cmd" \
  exp/tri3c/graph${lang_suffix}_tgpr data/test_eval92_vtln \
  exp/tri3c/decode${lang_suffix}_tgpr_eval92 || exit 1;
) &


# Below shows the results we got with this script.
# Actually we only have improvement on dev93 and the others get worse.
# With VTLN:
# for x in exp/tri3c/decode_tgpr_{dev,eval}{92,93}; do grep WER $x/wer_* | utils/best_wer.sh ; done
# %WER 13.86 [ 1141 / 8234, 235 ins, 123 del, 783 sub ] exp/tri3c/decode_tgpr_dev93/wer_17
# %WER 9.23 [ 521 / 5643, 131 ins, 31 del, 359 sub ] exp/tri3c/decode_tgpr_eval92/wer_16
# %WER 12.47 [ 430 / 3448, 67 ins, 43 del, 320 sub ] exp/tri3c/decode_tgpr_eval93/wer_14

# Baseline:
#(note, I had to run the following extra decoding to get this)
#steps/decode_fmllr.sh --nj 10 --cmd "$decode_cmd" exp/tri3b/graph_tgpr data/test_eval93_vtln exp/tri3b/decode_tgpr_eval93
#
# a04:s5: for x in exp/tri3b/decode_tgpr_{dev,eval}{92,93}; do grep WER $x/wer_* | utils/best_wer.sh ; done
# %WER 14.37 [ 1183 / 8234, 228 ins, 122 del, 833 sub ] exp/tri3b/decode_tgpr_dev93/wer_19
# %WER 8.98 [ 507 / 5643, 129 ins, 28 del, 350 sub ] exp/tri3b/decode_tgpr_eval92/wer_14
# %WER 12.21 [ 421 / 3448, 68 ins, 39 del, 314 sub ] exp/tri3b/decode_tgpr_eval93/wer_14

