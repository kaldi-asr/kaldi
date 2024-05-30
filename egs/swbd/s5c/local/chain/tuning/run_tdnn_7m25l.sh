#!/usr/bin/env bash

# 7m25l is as 7m25j but with no dropout on the prefinal layer.  Hoping to resolve
# bad objf in middle of training.
# Caution: in 7m25l2 there is a run which by mistake, did have dropout on the
# prefinal layer, and which should for the most part be just a rerun of 7m25j.

# This seems *maybe* slightly better than j and l2 (note: l2 is like j).

# local/chain/compare_wer_general.sh --rt03 tdnn7m23t_sp tdnn7m25j_sp tdnn7m25l2_sp tdnn7m25l_sp
# System                tdnn7m23t_sp tdnn7m25j_sp tdnn7m25l2_sp tdnn7m25l_sp
# WER on train_dev(tg)      12.18     11.95     11.98     11.90
# WER on train_dev(fg)      11.12     11.08     11.00     10.92
# WER on eval2000(tg)        14.9      14.6      14.7      14.7
# WER on eval2000(fg)        13.5      13.3      13.3      13.3
# WER on rt03(tg)            18.4      18.1      18.1      18.0
# WER on rt03(fg)            16.2      15.8      15.8      15.7
# Final train prob         -0.077    -0.078    -0.078    -0.076
# Final valid prob         -0.093    -0.091    -0.091    -0.091
# Final train prob (xent)        -0.994    -0.987    -0.987    -0.973
# Final valid prob (xent)       -1.0194   -1.0161   -1.0142   -1.0041
# Num-parameters               20111396  22735140  22735140  22735140

#
# But I may have changed the training code to accept more models in averaging,
# so that could be responsible for some of the change.
#

# local/chain/compare_wer_general.sh --rt03 tdnn7m23t_sp tdnn7m25i_sp tdnn7m25l_sp
# System                tdnn7m23t_sp tdnn7m25i_sp tdnn7m25l_sp
# WER on train_dev(tg)      12.18     12.13     11.98
# WER on train_dev(fg)      11.12     11.22     11.00
# WER on eval2000(tg)        14.9      15.0      14.7
# WER on eval2000(fg)        13.5      13.7      13.3
# WER on rt03(tg)            18.4      18.2      18.1
# WER on rt03(fg)            16.2      15.7      15.8
# Final train prob         -0.077    -0.078    -0.078
# Final valid prob         -0.093    -0.092    -0.091
# Final train prob (xent)        -0.994    -0.996    -0.987
# Final valid prob (xent)       -1.0194   -1.0214   -1.0142
# Num-parameters               20111396  22735140  22735140

# 7m25j is as 7m25i but with the dropout schedule peaking at 0.5 not 0.3,
#   and with 8 instead of 6 epochs (like g->h).
#   This run failed due to instability.

# 7m25i is as 7m25g but with dropout-per-dim-continuous=true.
#
# 7m25g is as 7m25f but with dim=1536 for the subsampled layers (more like 7m25d than 7m25e).

# 7m25f is as 7m25e but with a dropout schedule borrowed from the LSTM experiments.
#
# 7m25e is as 7m25d but reverting dims back from 1536 to 1280.

# 7m25d is as 7m25c but reverting to sharing the linear layer before the
# prefinal layer (more like 7m23t{,2}).  Also changing one splicing input
# to be from a layer that wasn't otherwise used as splicing input.

# 7m25c is as 7m25b but for the layers after we start using 3's not 1's,
#  increasing dim from 1280 to 1536.
# 7m25b is as 7m25a but with slightly different skip connections,
#  so all layers are the sources of skip connections.  (Also see 7m23u, although
#  that experiment didn't give clear results).
# 7m25a is as 7m23t but with some renamings of layers to make it more
# understandable, and changing how the last layer is done (there's now a little
# bit less sharing).

# 7m23t is as 7m23r but with 1280 instead of 1536 as the dim.
# Differernce vs. 23r is unclear (maybe slightly worse), but it
# seems slightly better than 23h, and it's nice that it has fewer parameters.


# local/chain/compare_wer_general.sh --rt03 tdnn7m23h_sp tdnn7m23r_sp tdnn7m23t_sp
# System                tdnn7m23h_sp tdnn7m23r_sp tdnn7m23t_sp
# WER on train_dev(tg)      12.28     11.95     12.18
# WER on train_dev(fg)      11.21     10.97     11.12
# WER on eval2000(tg)        15.0      15.0      14.9
# WER on eval2000(fg)        13.5      13.6      13.5
# WER on rt03(tg)            18.5      18.4      18.4
# WER on rt03(fg)            16.1      15.9      16.2
# Final train prob         -0.083    -0.076    -0.077
# Final valid prob         -0.097    -0.091    -0.093
# Final train prob (xent)        -1.036    -0.978    -0.994
# Final valid prob (xent)       -1.0629   -1.0026   -1.0194
# Num-parameters               23513380  23513380  20111396

# 7m23r is as 7m23h but with 6 epochs instead of 4.  See also 7m23p, which
# had 3 epochs.

# 7m23h is as 7m23b2 but with a small bugfix, removing a stray 'bottleneck-dim=192'.
# Seems slightly better.  The comparison below  includes our old TDNN+LSTM result
# with dropout, to show that we're doing better than that now.

# local/chain/compare_wer_general.sh --rt03 tdnn_lstm_1m_ld5_sp tdnn7m23b2_sp tdnn7m23h_sp
# System                tdnn_lstm_1m_ld5_sp tdnn7m23b2_sp tdnn7m23h_sp
# WER on train_dev(tg)      12.33     12.38     12.28
# WER on train_dev(fg)      11.42     11.44     11.21
# WER on eval2000(tg)        15.2      15.1      15.0
# WER on eval2000(fg)        13.8      13.6      13.5
# WER on rt03(tg)            18.6      18.4      18.5
# WER on rt03(fg)            16.3      16.1      16.1
# Final train prob         -0.082    -0.084    -0.083
# Final valid prob         -0.099    -0.098    -0.097
# Final train prob (xent)        -0.959    -1.049    -1.036
# Final valid prob (xent)       -1.0305   -1.0661   -1.0629
# Num-parameters               39558436  23120164  23513380
#
# 7m23b2 is as 7m23b but fixing an issue at the last layers.
# 7m23b is as 7m23 but making the splicing more 'symmetric'... doing the
#  splicing in 2 stages.  Interestingly, objf is not better than 23, but
# WER is slightly better.

# local/chain/compare_wer_general.sh --rt03 tdnn7m19m_sp tdnn7m23_sp tdnn7m23b2_sp
# System                tdnn7m19m_sp tdnn7m23_sp tdnn7m23b2_sp
# WER on train_dev(tg)      12.55     12.23     12.38
# WER on train_dev(fg)      11.52     11.29     11.44
# WER on eval2000(tg)        15.2      15.2      15.1
# WER on eval2000(fg)        13.6      13.7      13.6
# WER on rt03(tg)            18.6      18.7      18.4
# WER on rt03(fg)            16.2      16.3      16.1
# Final train prob         -0.089    -0.083    -0.084
# Final valid prob         -0.101    -0.097    -0.098
# Final train prob (xent)        -1.080    -1.025    -1.049
# Final valid prob (xent)       -1.0990   -1.0548   -1.0661
# Num-parameters               21055012  23120164  23120164


# 7m23 is as 7m19m but removing the bottlenecks from the batchnorm components and
#  reducing the dim of the linear components... it's basically an attempt to
#  reverse the factorization to have the splicing at a different point.
#

# 7m19m is as 7m19l but with more skip connections
#   Hm-- seems better than 19h.
#
# local/chain/compare_wer_general.sh --rt03 tdnn7m19h_sp tdnn7m19l_sp tdnn7m19m_sp
# System                tdnn7m19h_sp tdnn7m19l_sp tdnn7m19m_sp
# WER on train_dev(tg)      12.61     12.72     12.55
# WER on train_dev(fg)      11.72     11.62     11.52
# WER on eval2000(tg)        15.4      15.4      15.2
# WER on eval2000(fg)        13.7      13.8      13.6
# WER on rt03(tg)            18.9      18.9      18.6
# WER on rt03(fg)            16.3      16.4      16.2
# Final train prob         -0.091    -0.091    -0.089
# Final valid prob         -0.102    -0.103    -0.101
# Final train prob (xent)        -1.098    -1.095    -1.080
# Final valid prob (xent)       -1.1031   -1.1191   -1.0990
# Num-parameters               21055012  20268580  21055012
#
# 7m19l is as 7m19h but projecting down to an intermediate dim (512) before
# doing the Append... doing this by inserting a linear-component between
# pairs of relu-batchnorm-layers.
#  A little worse.
# local/chain/compare_wer_general.sh --rt03 tdnn7m19h_sp tdnn7m19l_sp
# System                tdnn7m19h_sp tdnn7m19l_sp
# WER on train_dev(tg)      12.65     12.72
# WER on train_dev(fg)      11.57     11.62
# WER on eval2000(tg)        15.3      15.4
# WER on eval2000(fg)        13.7      13.8
# WER on rt03(tg)            18.8      18.9
# WER on rt03(fg)            16.4      16.4
# Final train prob         -0.091    -0.091
# Final valid prob         -0.102    -0.103
# Final train prob (xent)        -1.091    -1.095
# Final valid prob (xent)       -1.1064   -1.1191
# Num-parameters               21055012  20268580


# 7m19h is as 7m19e but with an extra bypass connection.  A bit better.

# local/chain/compare_wer_general.sh --rt03 tdnn7m19e_sp tdnn7m19h_sp
# System                tdnn7m19e_sp tdnn7m19h_sp
# WER on train_dev(tg)      12.75     12.65
# WER on train_dev(fg)      11.77     11.57
# WER on eval2000(tg)        15.5      15.3
# WER on eval2000(fg)        14.0      13.7
# WER on rt03(tg)            18.9      18.8
# WER on rt03(fg)            16.4      16.4
# Final train prob         -0.092    -0.091
# Final valid prob         -0.102    -0.102
# Final train prob (xent)        -1.094    -1.091
# Final valid prob (xent)       -1.1095   -1.1064
# Num-parameters               20760100  21055012

# 7m19e is as 7m19c,d but with dims increased to 1536.  Better!

# local/chain/compare_wer_general.sh --rt03 tdnn7m10_sp tdnn7m19c_sp tdnn7m19d_sp tdnn7m19e_sp
# System                tdnn7m10_sp tdnn7m19c_sp tdnn7m19d_sp tdnn7m19e_sp
# WER on train_dev(tg)      13.77     12.86     13.01     12.75
# WER on train_dev(fg)      12.65     11.82     12.02     11.77
# WER on eval2000(tg)        16.1      15.4      15.7      15.5
# WER on eval2000(fg)        14.3      13.8      14.0      14.0
# WER on rt03(tg)            19.9      19.1      19.2      18.9
# WER on rt03(fg)            17.4      16.6      16.7      16.4
# Final train prob         -0.111    -0.094    -0.096    -0.092
# Final valid prob         -0.120    -0.103    -0.105    -0.102
# Final train prob (xent)        -1.314    -1.117    -1.144    -1.094
# Final valid prob (xent)       -1.3247   -1.1223   -1.1478   -1.1095
# Num-parameters               13361700  17824036  14887972  20760100

# local/chain/compare_wer_general.sh --rt03 tdnn7m16_sp tdnn7m19_sp tdnn7m19b_sp tdnn7m19c_sp tdnn7m19d_sp
# System                tdnn7m16_sp tdnn7m19_sp tdnn7m19b_sp tdnn7m19c_sp tdnn7m19d_sp
# WER on train_dev(tg)      13.37     13.09     12.93     12.86     13.01
# WER on train_dev(fg)      12.47     12.12     11.87     11.82     12.02
# WER on eval2000(tg)        15.8      15.8      15.6      15.4      15.7
# WER on eval2000(fg)        14.3      14.3      14.0      13.8      14.0
# WER on rt03(tg)            15.1      14.8      14.9      14.8      14.9
# WER on rt03(fg)            12.7      12.4      12.5      12.5      12.6
# Final train prob         -0.099    -0.096    -0.096    -0.094    -0.096
# Final valid prob         -0.110    -0.106    -0.106    -0.103    -0.105
# Final train prob (xent)        -1.302    -1.198    -1.188    -1.117    -1.144
# Final valid prob (xent)       -1.3184   -1.2070   -1.1980   -1.1223   -1.1478
# Num-parameters               14216996  15528996  16512036  17824036  14887972

# 7m19c is as 7m19b but with one more layer (and moving the bypass connections up).
#  Seems about 0.1% better.

# local/chain/compare_wer_general.sh --rt03 tdnn7m19_sp tdnn7m19b_sp tdnn7m19c_sp
# System                tdnn7m19_sp tdnn7m19b_sp tdnn7m19c_sp
# WER on train_dev(tg)      13.09     12.93     12.86
# WER on train_dev(fg)      12.12     11.87     11.82
# WER on eval2000(tg)        15.8      15.6      15.4
# WER on eval2000(fg)        14.3      14.0      13.8
# WER on rt03(tg)            14.8      14.9      14.8
# WER on rt03(fg)            12.4      12.5      12.5
# Final train prob         -0.096    -0.096    -0.094
# Final valid prob         -0.106    -0.106    -0.103
# Final train prob (xent)        -1.198    -1.188    -1.117
# Final valid prob (xent)       -1.2070   -1.1980   -1.1223
# Num-parameters               15528996  16512036  17824036

# local/chain/compare_wer_general.sh --rt03 tdnn7m19_sp tdnn7m19b_sp
# System                tdnn7m19_sp tdnn7m19b_sp
# WER on train_dev(tg)      13.09     12.93
# WER on train_dev(fg)      12.12     11.87
# WER on eval2000(tg)        15.8      15.6
# WER on eval2000(fg)        14.3      14.0
# WER on rt03(tg)            14.8      14.9
# WER on rt03(fg)            12.4      12.5
# Final train prob         -0.096    -0.096
# Final valid prob         -0.106    -0.106
# Final train prob (xent)        -1.198    -1.188
# Final valid prob (xent)       -1.2070   -1.1980
# Num-parameters               15528996  16512036

# 7m19 is as 7m16 but adding an extra -3,0,3 layer.
# CAUTION: messing with queue opts.
# 7m16 is as 7m15 but removing the chain l2-regularize.  Does seem better.

# local/chain/compare_wer_general.sh --rt03 tdnn7m12_sp tdnn7m15_sp tdnn7m16_sp
# System                tdnn7m12_sp tdnn7m15_sp tdnn7m16_sp
# WER on train_dev(tg)      13.58     13.50     13.37
# WER on train_dev(fg)      12.43     12.44     12.47
# WER on eval2000(tg)        16.0      16.0      15.8
# WER on eval2000(fg)        14.3      14.3      14.3
# WER on rt03(tg)            15.2      15.4      15.1
# WER on rt03(fg)            13.0      13.0      12.7
# Final train prob         -0.109    -0.111    -0.099
# Final valid prob         -0.117    -0.119    -0.110
# Final train prob (xent)        -1.278    -1.291    -1.302
# Final valid prob (xent)       -1.2880   -1.3036   -1.3184
# Num-parameters               16089380  14216996  14216996

# 7m15 is as 7m12 but reducing the bottleneck dim at the output from
#   384 to 256 (like 11->14).
# 7m12 is as 7m11 but increasing all the TDNN dims from 1024 to 1280.
# Seems a little better but could be due to the increase in parameters.

# local/chain/compare_wer_general.sh --rt03 tdnn7m8_sp tdnn7m9_sp tdnn7m10_sp tdnn7m11_sp tdnn7m12_sp
# System                tdnn7m8_sp tdnn7m9_sp tdnn7m10_sp tdnn7m11_sp tdnn7m12_sp
# WER on train_dev(tg)      13.60     13.88     13.77     13.83     13.58
# WER on train_dev(fg)      12.62     12.64     12.65     12.65     12.43
# WER on eval2000(tg)        16.8      16.1      16.1      16.1      16.0
# WER on eval2000(fg)        15.4      14.4      14.3      14.5      14.3
# WER on rt03(tg)            16.2      15.5      15.6      15.3      15.2
# WER on rt03(fg)            13.7      13.1      13.2      13.0      13.0
# Final train prob         -0.105    -0.111    -0.111    -0.109    -0.109
# Final valid prob         -0.115    -0.119    -0.120    -0.118    -0.117
# Final train prob (xent)        -1.282    -1.309    -1.314    -1.292    -1.278
# Final valid prob (xent)       -1.3194   -1.3246   -1.3247   -1.3077   -1.2880
# Num-parameters               11580452  13818148  13361700  13809188  16089380

# 7m11 is as 7m10 but increasing the TDNN dims and reducing the bottlenecks.
# 7m10 is as 7m9 but reducing the bottleneck-dims for the non-splicing TDNN layers.
# 7m9 is as 7m8 but adding bottleneck-dims, and increasing the TDNN dims.

# local/chain/compare_wer_general.sh --rt03 tdnn7m8_sp tdnn7m9_sp
# System                tdnn7m8_sp tdnn7m9_sp
# WER on train_dev(tg)      13.60     13.88
# WER on train_dev(fg)      12.62     12.64
# WER on eval2000(tg)        16.8      16.1
# WER on eval2000(fg)        15.4      14.4
# WER on rt03(tg)            16.2      15.5
# WER on rt03(fg)            13.7      13.1
# Final train prob         -0.105    -0.111
# Final valid prob         -0.115    -0.119
# Final train prob (xent)        -1.282    -1.309
# Final valid prob (xent)       -1.3194   -1.3246
# Num-parameters               11580452  13818148

# 7m8 is as 7m5b but double the l2-regularization for the TDNN layers, which
# is the same as 7m2->7m3, which was helpful there.
#  Does seem helpful.

# local/chain/compare_wer_general.sh --rt03 tdnn_7m_sp tdnn_7m2_sp tdnn7m5b_sp tdnn7m8_sp
# System                tdnn_7m_sp tdnn_7m2_sp tdnn7m5b_sp tdnn7m8_sp
# WER on train_dev(tg)      13.70     13.74     13.81     13.60
# WER on train_dev(fg)      12.67     12.76     12.74     12.62
# WER on eval2000(tg)        16.6      17.1      17.0      16.8
# WER on eval2000(fg)        15.1      15.4      15.4      15.4
# WER on rt03(tg)            16.1      16.2      16.0      16.2
# WER on rt03(fg)            13.7      13.8      13.6      13.7
# Final train prob         -0.085    -0.106    -0.104    -0.105
# Final valid prob         -0.103    -0.118    -0.116    -0.115
# Final train prob (xent)        -1.230    -1.296    -1.285    -1.282
# Final valid prob (xent)       -1.2704   -1.3318   -1.3283   -1.3194
# Num-parameters               16292693  10924836  11580452  11580452


# 7m5b is as 7m5 but rducing the prefinal layer dims to previous values.
# WER changes (+ is worse): +1 +1 +2 +3 -2 -2...  so maybe worse on average,
#  but not clear at all... for consistency with other setups I may retain
#  this change.

# local/chain/compare_wer_general.sh --rt03 tdnn_7m_sp tdnn_7m2_sp tdnn7m5_sp tdnn7m5b_sp
# System                tdnn_7m_sp tdnn_7m2_sp tdnn7m5_sp tdnn7m5b_sp
# WER on train_dev(tg)      13.70     13.74     13.71     13.81
# WER on train_dev(fg)      12.67     12.76     12.64     12.74
# WER on eval2000(tg)        16.6      17.1      16.8      17.0
# WER on eval2000(fg)        15.1      15.4      15.1      15.4
# WER on rt03(tg)            16.1      16.2      16.2      16.0
# WER on rt03(fg)            13.7      13.8      13.8      13.6
# Final train prob         -0.085    -0.106    -0.103    -0.104
# Final valid prob         -0.103    -0.118    -0.114    -0.116
# Final train prob (xent)        -1.230    -1.296    -1.274    -1.285
# Final valid prob (xent)       -1.2704   -1.3318   -1.3016   -1.3283
# Num-parameters               16292693  10924836  12170788  11580452


# 7m5 is as 7m2 but increasing the dimension of the last TDNN layer
#  and the prefinal layers from 512 to 768.
# 7m2 is as 7m but with a bunch of tuning changes (model is smaller).
# 7m is as 7k but adding two non-splicing layers towards the beginning of the
#   network.
# The impovement is pretty small but I've seen similar improvements on other
# setups with this architecture so I tend to believe it.


# local/chain/compare_wer_general.sh tdnn_7k_sp tdnn_7m_sp
# System                tdnn_7k_sp tdnn_7m_sp
# WER on train_dev(tg)      13.83     13.65
# WER on train_dev(fg)      12.74     12.54
# WER on eval2000(tg)        16.9      16.8
# WER on eval2000(fg)        15.2      15.1
# Final train prob         -0.085    -0.084
# Final valid prob         -0.107    -0.103
# Final train prob (xent)        -1.267    -1.215
# Final valid prob (xent)       -1.3107   -1.2735

# steps/info/chain_dir_info.pl exp/chain/tdnn_7m_sp
# exp/chain/tdnn_7m_sp: num-iters=262 nj=3..16 num-params=16.3M dim=40+100->6034 combine=-0.103->-0.103 xent:train/valid[173,261,final]=(-1.28,-1.21,-1.21/-1.32,-1.27,-1.27) logprob:train/valid[173,261,final]=(-0.093,-0.084,-0.084/-0.109,-0.104,-0.103)

set -e

# configs for 'chain'
stage=0
train_stage=-10
get_egs_stage=-10
speed_perturb=true
affix=7m25l
suffix=
$speed_perturb && suffix=_sp
if [ -e data/rt03 ]; then maybe_rt03=rt03; else maybe_rt03= ; fi

dir=exp/chain/tdnn${affix}${suffix}
decode_iter=
decode_nj=50

# training options
frames_per_eg=150,110,100
remove_egs=false
common_egs_dir=
xent_regularize=0.1
dropout_schedule='0,0@0.20,0.5@0.50,0'

test_online_decoding=false  # if true, it will run the last decoding stage.

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 8" if you have already
# run those things.

train_set=train_nodup$suffix
ali_dir=exp/tri4_ali_nodup$suffix
treedir=exp/chain/tri5_7d_tree$suffix
lang=data/lang_chain_2y


# if we are using the speed-perturbed data we need to generate
# alignments for it.
local/nnet3/run_ivector_common.sh --stage $stage \
  --speed-perturb $speed_perturb \
  --generate-alignments $speed_perturb || exit 1;


if [ $stage -le 9 ]; then
  # Get the alignments as lattices (gives the LF-MMI training more freedom).
  # use the same num-jobs as the alignments
  nj=$(cat exp/tri4_ali_nodup$suffix/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" data/$train_set \
    data/lang exp/tri4 exp/tri4_lats_nodup$suffix
  rm exp/tri4_lats_nodup$suffix/fsts.*.gz # save space
fi


if [ $stage -le 10 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  rm -rf $lang
  cp -r data/lang $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi

if [ $stage -le 11 ]; then
  # Build a tree using our new topology. This is the critically different
  # step compared with other recipes.
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --context-opts "--context-width=2 --central-position=1" \
      --cmd "$train_cmd" 7000 data/$train_set $lang $ali_dir $treedir
fi

if [ $stage -le 12 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)
  opts="l2-regularize=0.002 dropout-proportion=0.0 dropout-per-dim=true dropout-per-dim-continuous=true"
  linear_opts="orthonormal-constraint=1.0"
  output_opts="l2-regularize=0.0005"

  mkdir -p $dir/configs

  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-dropout-layer name=tdnn1 $opts dim=1280
  linear-component name=tdnn2l dim=256 $linear_opts input=Append(-1,0)
  relu-batchnorm-dropout-layer name=tdnn2 $opts input=Append(0,1) dim=1280
  linear-component name=tdnn3l dim=256 $linear_opts
  relu-batchnorm-dropout-layer name=tdnn3 $opts dim=1280
  linear-component name=tdnn4l dim=256 $linear_opts input=Append(-1,0)
  relu-batchnorm-dropout-layer name=tdnn4 $opts input=Append(0,1) dim=1280
  linear-component name=tdnn5l dim=256 $linear_opts
  relu-batchnorm-dropout-layer name=tdnn5 $opts dim=1536 input=Append(0, tdnn3l)
  linear-component name=tdnn6l dim=256 $linear_opts input=Append(-3,0)
  relu-batchnorm-dropout-layer name=tdnn6 $opts input=Append(0,3) dim=1536
  linear-component name=tdnn7l dim=256 $linear_opts input=Append(-3,0)
  relu-batchnorm-dropout-layer name=tdnn7 $opts input=Append(0,3,tdnn6l,tdnn4l,tdnn2l) dim=1536
  linear-component name=tdnn8l dim=256 $linear_opts input=Append(-3,0)
  relu-batchnorm-dropout-layer name=tdnn8 $opts input=Append(0,3) dim=1536
  linear-component name=tdnn9l dim=256 $linear_opts input=Append(-3,0)
  relu-batchnorm-dropout-layer name=tdnn9 $opts input=Append(0,3,tdnn8l,tdnn6l,tdnn5l) dim=1536
  linear-component name=tdnn10l dim=256 $linear_opts input=Append(-3,0)
  relu-batchnorm-dropout-layer name=tdnn10 $opts input=Append(0,3) dim=1536
  linear-component name=tdnn11l dim=256 $linear_opts input=Append(-3,0)
  relu-batchnorm-dropout-layer name=tdnn11 $opts input=Append(0,3,tdnn10l,tdnn9l,tdnn7l) dim=1536
  linear-component name=prefinal-l dim=256 $linear_opts

  relu-batchnorm-layer name=prefinal-chain input=prefinal-l $opts dim=1536
  output-layer name=output include-log-softmax=false dim=$num_targets bottleneck-dim=256 $output_opts

  relu-batchnorm-layer name=prefinal-xent input=prefinal-l $opts dim=1536
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor bottleneck-dim=256 $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 13 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/swbd-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "queue.pl --config /home/dpovey/queue_conly.conf" \
    --feat.online-ivector-dir exp/nnet3/ivectors_${train_set} \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.0 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.dropout-schedule $dropout_schedule \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width $frames_per_eg \
    --trainer.num-chunk-per-minibatch 128 \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs 8 \
    --trainer.optimization.num-jobs-initial 3 \
    --trainer.optimization.num-jobs-final 16 \
    --trainer.optimization.initial-effective-lrate 0.001 \
    --trainer.optimization.final-effective-lrate 0.0001 \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs $remove_egs \
    --feat-dir data/${train_set}_hires \
    --tree-dir $treedir \
    --lat-dir exp/tri4_lats_nodup$suffix \
    --dir $dir  || exit 1;

fi

if [ $stage -le 14 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_sw1_tg $dir $dir/graph_sw1_tg
fi


graph_dir=$dir/graph_sw1_tg
iter_opts=
if [ ! -z $decode_iter ]; then
  iter_opts=" --iter $decode_iter "
fi
if [ $stage -le 15 ]; then
  rm $dir/.error 2>/dev/null || true
  for decode_set in train_dev eval2000 $maybe_rt03; do
      (
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $decode_nj --cmd "$decode_cmd" $iter_opts \
          --online-ivector-dir exp/nnet3/ivectors_${decode_set} \
          $graph_dir data/${decode_set}_hires \
          $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_sw1_tg || exit 1;
      if $has_fisher; then
          steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
            data/lang_sw1_{tg,fsh_fg} data/${decode_set}_hires \
            $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_sw1_{tg,fsh_fg} || exit 1;
      fi
      ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi

if $test_online_decoding && [ $stage -le 16 ]; then
  # note: if the features change (e.g. you add pitch features), you will have to
  # change the options of the following command line.
  steps/online/nnet3/prepare_online_decoding.sh \
       --mfcc-config conf/mfcc_hires.conf \
       $lang exp/nnet3/extractor $dir ${dir}_online

  rm $dir/.error 2>/dev/null || true
  for decode_set in train_dev eval2000 $maybe_rt03; do
    (
      # note: we just give it "$decode_set" as it only uses the wav.scp, the
      # feature type does not matter.

      steps/online/nnet3/decode.sh --nj $decode_nj --cmd "$decode_cmd" \
          --acwt 1.0 --post-decode-acwt 10.0 \
         $graph_dir data/${decode_set}_hires \
         ${dir}_online/decode_${decode_set}${decode_iter:+_$decode_iter}_sw1_tg || exit 1;
      if $has_fisher; then
          steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
            data/lang_sw1_{tg,fsh_fg} data/${decode_set}_hires \
            ${dir}_online/decode_${decode_set}${decode_iter:+_$decode_iter}_sw1_{tg,fsh_fg} || exit 1;
      fi
    ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi


exit 0;
