#!/bin/bash -u

# Copyright 2012  Arnab Ghoshal

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

# This script shows the steps needed to build a recognizer for certain languages
# of the GlobalPhone corpus. 
# !!! NOTE: The current recipe assumes that you have pre-built LMs. 
echo "This shell script may run as-is on your system, but it is recommended 
that you run the commands one by one by copying and pasting into the shell."
#exit 1;

[ -f cmd.sh ] && source ./cmd.sh \
  || echo "cmd.sh not found. Jobs may not execute properly."

# CHECKING FOR AND INSTALLING REQUIRED TOOLS:
#  This recipe requires shorten (3.6.1) and sox (14.3.2).
#  If they are not found, the local/gp_install.sh script will install them.
local/gp_check_tools.sh $PWD path.sh

. path.sh || { echo "Cannot source path.sh"; exit 1; }

# Set the locations of the GlobalPhone corpus and language models
GP_CORPUS=/mnt/matylda2/data/GLOBALPHONE
# GP_LM=/mnt/matylda6/ijanda/GLOBALPHONE_LM
GP_LM=$PWD/language_models

# Set the languages that will actually be processed
# export GP_LANGUAGES="CZ FR GE PL PO RU SP VN"
export GP_LANGUAGES="CZ FR GE PL PO RU SP"

# The following data preparation step actually converts the audio files from 
# shorten to WAV to take out the empty files and those with compression errors. 
local/gp_data_prep.sh --config-dir=$PWD/conf --corpus-dir=$GP_CORPUS \
  --languages="$GP_LANGUAGES"

for L in $GP_LANGUAGES; do
  utils/prepare_lang.sh --position-dependent-phones true \
    data/$L/local/dict "<UNK>" data/$L/local/lang_tmp data/$L/lang \
    >& data/$L/prepare_lang.log || exit 1;
done

# Convert the different available language models to FSTs, and create separate 
# decoding configurations for each.
( 
#  local/gp_format_lm.sh --filter-vocab-sri false $GP_LM $GP_LANGUAGES;
  local/gp_format_lm.sh --filter-vocab-sri true $GP_LM $GP_LANGUAGES;
) >& data/format_lm.log &

# Now make MFCC features.
for L in $GP_LANGUAGES; do
  mfccdir=mfcc/$L
  for x in train dev eval; do
    ( steps/make_mfcc.sh --nj 6 --cmd "$train_cmd" data/$L/$x \
        exp/$L/make_mfcc/$x $mfccdir;
      steps/compute_cmvn_stats.sh data/$L/$x exp/$L/make_mfcc/$x $mfccdir; ) &
  done
done
wait;

for L in $GP_LANGUAGES; do
  mkdir -p exp/$L/mono;
  steps/train_mono.sh --nj 10 --cmd "$train_cmd" \
    data/$L/train data/$L/lang exp/$L/mono >& exp/$L/mono/train.log &
done
wait;

# for L in $GP_LANGUAGES; do
#   for lm_suffix in tgpr_sri; do
#     graph_dir=exp/$L/mono/graph_${lm_suffix}
#     mkdir -p $graph_dir
#     utils/mkgraph.sh --mono data/$L/lang_test_${lm_suffix} exp/$L/mono \
#       $graph_dir >& $graph_dir/mkgraph.log

#     steps/decode.sh --nj 5 --cmd "$decode_cmd" $graph_dir data/$L/dev \
#       exp/$L/mono/decode_dev_${lm_suffix} &
#   done
# done


for L in $GP_LANGUAGES; do
  (
    mkdir -p exp/$L/mono_ali
    steps/align_si.sh --nj 10 --cmd "$train_cmd" \
      data/$L/train data/$L/lang exp/$L/mono exp/$L/mono_ali \
      >& exp/$L/mono_ali/align.log 

    num_states=$(grep "^$L" conf/tri.conf | cut -f2)
    num_gauss=$(grep "^$L" conf/tri.conf | cut -f3)
    mkdir -p exp/$L/tri1
    steps/train_deltas.sh --cmd "$train_cmd" --cluster-thresh 100 \
      $num_states $num_gauss data/$L/train data/$L/lang exp/$L/mono_ali \
      exp/$L/tri1 >& exp/$L/tri1/train.log
    ) &
done
wait;


# for L in $GP_LANGUAGES; do
#   for lm_suffix in tgpr_sri; do
#     graph_dir=exp/$L/tri1/graph_${lm_suffix}
#     mkdir -p $graph_dir
#     utils/mkgraph.sh data/$L/lang_test_${lm_suffix} exp/$L/tri1 $graph_dir \
#       >& $graph_dir/mkgraph.log

#     steps/decode.sh --nj 5 --cmd "$decode_cmd" $graph_dir data/$L/dev \
#       exp/$L/tri1/decode_dev_${lm_suffix} &
#   done
# done

for L in $GP_LANGUAGES; do
  (
    mkdir -p exp/$L/tri1_ali
    steps/align_si.sh --nj 10 --cmd "$train_cmd" \
      data/$L/train data/$L/lang exp/$L/tri1 exp/$L/tri1_ali \
      >& exp/$L/tri1_ali/align.log

    num_states=$(grep "^$L" conf/tri.conf | cut -f2)
    num_gauss=$(grep "^$L" conf/tri.conf | cut -f3)
    mkdir -p exp/$L/tri2a
    steps/train_deltas.sh --cmd "$train_cmd" --cluster-thresh 100 \
      $num_states $num_gauss data/$L/train data/$L/lang exp/$L/tri1_ali \
      exp/$L/tri2a >& exp/$L/tri2a/train.log
  ) &
done
wait;

# for L in $GP_LANGUAGES; do
#   for lm_suffix in tgpr_sri; do
#     graph_dir=exp/$L/tri2a/graph_${lm_suffix}
#     mkdir -p $graph_dir
#     utils/mkgraph.sh data/$L/lang_test_${lm_suffix} exp/$L/tri2a $graph_dir \
#       >& $graph_dir/mkgraph.log

#     steps/decode.sh --nj 5 --cmd "$decode_cmd" $graph_dir data/$L/dev \
#       exp/$L/tri2a/decode_dev_${lm_suffix} &
#   done
# done

# for L in $GP_LANGUAGES; do
#   mode=4
# # Doing this only for the LMs whose vocabs were limited using SRILM, since the
# # other approach didn't yield LMs for all languages.  
#   steps/lmrescore.sh --mode $mode --cmd "$decode_cmd" \
#     data/$L/lang_test_tgpr_sri data/$L/lang_test_tg_sri data/$L/dev \
#     exp/$L/tri2a/decode_dev_tgpr_sri exp/$L/tri2a/decode_dev_tg_sri$mode
# done

for L in $GP_LANGUAGES; do
  (
    steps/train_ubm.sh --cmd "$train_cmd" \
      400 data/$L/train data/$L/lang exp/$L/tri1_ali exp/$L/ubm2b || exit 1;

    num_states=$(grep "^$L" conf/tri.conf | cut -f2)
    num_gauss=$(grep "^$L" conf/tri.conf | cut -f3)
    mkdir -p exp/$L/sgmm2b
    steps/train_sgmm.sh --cmd "$train_cmd" --cluster-thresh 100 \
      $num_states $num_gauss data/$L/train data/$L/lang exp/$L/tri1_ali \
      exp/$L/ubm2b/final.ubm exp/$L/sgmm2b >& exp/$L/sgmm2b/train.log
  ) &
done
wait

for L in $GP_LANGUAGES; do
  (
    steps/align_fmllr.sh --nj 10 --cmd "$train_cmd" \
      data/$L/train data/$L/lang exp/$L/tri1 exp/$L/tri1_ali_fmllr || exit 1;

    steps/train_ubm.sh --cmd "$train_cmd" \
      400 data/$L/train data/$L/lang exp/$L/tri1_ali_fmllr exp/$L/ubm2c \
      || exit 1;

    num_states=$(grep "^$L" conf/tri.conf | cut -f2)
    num_gauss=$(grep "^$L" conf/tri.conf | cut -f3)
    mkdir -p exp/$L/sgmm2c
    steps/train_sgmm.sh --cmd "$train_cmd" --cluster-thresh 100 \
      $num_states $num_gauss data/$L/train data/$L/lang exp/$L/tri1_ali_fmllr \
      exp/$L/ubm2c/final.ubm exp/$L/sgmm2c >& exp/$L/sgmm2c/train.log
  ) &
done
wait
