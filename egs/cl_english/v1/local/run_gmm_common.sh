#!/usr/bin/env bash

# Copyright 2021  Behavox (author: Hossein Hadian)
# Apache 2.0

# This script runs all common stages for training the GMM models
# needed for chain training

train=train_ldc
bsil=1.2
nj=20
exp=exp
test_sets="test_ldc test_sp_oc"
num_mono_utts=1000
lm=
lang_opts=

tri3_lda=false  # If true, the final system (tri3) will be LDA
                  # and no SAT will be trained.
num_deltas_utts=30000
num_lda_utts=50000
num_tri3_utts=

deltas_leaves="1500 10000"
lda_leaves="2000 15000"
sat_leaves="2500 25000"

. ./cmd.sh
. ./path.sh

stage=0

echo "$0 $@"  # Print the command line for logging
. utils/parse_options.sh

set -euo pipefail

if [ $stage -le 0 ]; then
    for dset in $train $test_sets; do
      echo "$0: exctracting features for data/$dset..."
      if [ -f data/$dset/feats.scp ]; then
        printf "\nNote: data/$dset/feats.scp exists. skipping...\n\n"
      else
        steps/make_mfcc.sh --nj $nj data/$dset
        steps/compute_cmvn_stats.sh data/$dset
        utils/fix_data_dir.sh data/$dset
      fi
    done
fi

num_utts=$(wc -l < data/$train/utt2spk)
if [ $stage -le 1 ]; then
  utils/subset_data_dir.sh --shortest data/$train $num_mono_utts data/${train}_mono
  if [ $num_utts -gt $num_deltas_utts ]; then
    utils/subset_data_dir.sh data/$train $num_deltas_utts data/${train}_deltas
  fi
  if [ $num_utts -gt $num_lda_utts ]; then
    utils/subset_data_dir.sh data/$train $num_lda_utts data/${train}_lda
  fi
  if [[ ! -z $num_tri3_utts ]] && [[ $num_utts -gt $num_tri3_utts ]]; then
    utils/subset_data_dir.sh data/$train $num_tri3_utts data/${train}_tri3
  fi
fi

deltas_train=$train
lda_train=$train
tri3_train=$train
if [ $num_utts -gt $num_deltas_utts ]; then
  deltas_train=${train}_deltas
fi
if [ $num_utts -gt $num_lda_utts ]; then
  lda_train=${train}_lda
fi
if [[ ! -z $num_tri3_utts ]] && [[ $num_utts -gt $num_tri3_utts ]]; then
  tri3_train=${train}_tri3
fi

if [ $stage -le 3 ]; then
  steps/train_mono.sh --boost-silence $bsil --nj $nj --cmd "$train_cmd" \
    data/${train}_mono data/lang_nosp $exp/mono

  steps/align_si.sh --boost-silence $bsil --nj $nj --cmd "$train_cmd" \
    data/${deltas_train} data/lang_nosp $exp/mono $exp/mono_ali_${deltas_train}
fi

if [ $stage -le 4 ]; then
  steps/train_deltas.sh --boost-silence $bsil --cmd "$train_cmd" \
    $deltas_leaves data/$deltas_train data/lang_nosp $exp/mono_ali_${deltas_train} $exp/tri1

  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/$lda_train data/lang_nosp $exp/tri1 $exp/tri1_ali_${lda_train}
fi

if [ $stage -le 5 ]; then
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" $lda_leaves \
    data/$lda_train data/lang_nosp $exp/tri1_ali_${lda_train} $exp/tri2b

  # Align utts using the tri2b model
  steps/align_si.sh  --nj $nj --cmd "$train_cmd" --use-graphs true \
    data/${tri3_train} data/lang_nosp $exp/tri2b $exp/tri2b_ali_${tri3_train}
fi

# Train tri3b, which is LDA+MLLT+SAT
if [ $stage -le 6 ]; then
  if $tri3_lda; then
    steps/train_lda_mllt.sh --cmd "$train_cmd" \
      --splice-opts "--left-context=3 --right-context=3" $sat_leaves \
      data/$tri3_train data/lang_nosp $exp/tri2b_ali_${tri3_train} $exp/tri3b
  else
    steps/train_sat.sh --cmd "$train_cmd" $sat_leaves \
                       data/${tri3_train} data/lang_nosp $exp/tri2b_ali_${tri3_train} $exp/tri3b
  fi
fi

# Now we compute the pronunciation and silence probabilities from training data,
# and re-create the lang directory.
if [ $stage -le 7 ]; then
  steps/get_prons.sh --cmd "$train_cmd" \
    data/${train} data/lang_nosp $exp/tri3b

  if [ ! -f $exp/tri3b/pron_counts_nowb.txt ]; then
    cp $exp/tri3b/pron_counts{,_nowb}.txt
  fi

  utils/dict_dir_add_pronprobs.sh --max-normalize true \
    data/local/dict_nosp \
    $exp/tri3b/pron_counts_nowb.txt $exp/tri3b/sil_counts_nowb.txt \
    $exp/tri3b/pron_bigram_counts_nowb.txt $exp/dict

  utils/prepare_lang.sh $lang_opts $exp/dict "<unk>" $exp/lang/{tmp,}
  if [ ! -z $lm ]; then
    utils/format_lm.sh $exp/lang $lm $exp/dict/lexicon.txt $exp/lang_tg
  fi
fi

if [[ $stage -le 8 ]] && [[ ! -z $test_sets ]]; then
  graph_dir=$exp/tri3b/graph_tg
  $train_cmd $graph_dir/mkgraph.log utils/mkgraph.sh $exp/lang_tg $exp/tri3b $graph_dir
  for eval_ds in $test_sets; do
      nspk=$(wc -l <data/$eval_ds/spk2utt)
      if [ $nspk -gt $nj ]; then
        nspk=$nj
      fi
      steps/decode_fmllr.sh --nj $nspk --num-threads 8 --cmd "$decode_cmd" \
                          $graph_dir data/$eval_ds $exp/tri3b/decode_${eval_ds}
  done
fi
exit
