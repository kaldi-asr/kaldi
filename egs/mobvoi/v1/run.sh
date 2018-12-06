#!/bin/bash

stage=0


. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -euo pipefail

if [ $stage -le 0 ]; then
  local/mobvoi_data_download.sh
  echo "$0: Extracted all datasets into data/download/"
fi

if [ $stage -le 1 ]; then
  echo "$0: Splitting datasets..."
  local/split_datasets.sh
  echo "$0: text and utt2spk have been generated in data/{train|dev|eval}."
fi
    
if [ $stage -le 2 ]; then
  echo "$0: Preparing wav.scp..."
  local/prepare_wav.py data
  echo "wav.scp has been generated in data/{train|dev|eval}."
fi

if [ $stage -le 3 ]; then
  echo "$0: Extracting MFCC..."
  for folder in train dev eval; do
    dir=data/$folder
    utils/fix_data_dir.sh $dir
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 16 $dir
    steps/compute_cmvn_stats.sh $dir
    utils/fix_data_dir.sh $dir
    utils/validate_data_dir.sh $dir
  done
fi

if [ $stage -le 4 ]; then
  echo "$0: Post processing transcripts..."
  for folder in train dev eval; do
    dir=data/$folder
    cat $dir/text | awk '{if ($2=="嗨小问" || $2=="嗨小问嗨小问") {print $1,"嗨小问";} else {print $1,""}}' > $dir/text.tmp || exit 1
    cat $dir/text.tmp > $dir/text || exit 1
    rm -f $dir/text.tmp 2>/dev/null || true
  done
fi

if [ $stage -le 5 ]; then
  echo "$0: Preparing dictionary and lang..."
  local/prepare_dict.sh
  utils/prepare_lang.sh --num-sil-states 1 --num-nonsil-states 4 --sil-prob 0.5 \
    --position-dependent-phones false \
    data/local/dict "<sil>" data/lang/temp data/lang
fi

if [ $stage -le 6 ]; then
  id=`cat data/lang/words.txt | grep "嗨小问" | awk '{print $2}'`
  mkdir -p data/lang/lm
  cat <<EOF > data/lang/lm/fst.txt
0 1 $id $id
0 4.0
1 0.0
EOF
  fstcompile data/lang/lm/fst.txt data/lang/G.fst
  set +e
  fstisstochastic data/lang/G.fst
  set -e
  utils/validate_lang.pl data/lang
  exit 0
fi

# monophone training
if [ $stage -le 7 ]; then
  steps/train_mono.sh --nj 50 --cmd "$train_cmd" \
    data/train data/lang exp/mono
  (
    utils/mkgraph.sh data/lang \
      exp/mono exp/mono/graph
    for test in dev eval; do
      steps/decode.sh --nj 50 --cmd "$decode_cmd" exp/mono/graph \
        data/$test exp/mono/decode_$test
    done
  )&

  steps/align_si.sh --nj 50 --cmd "$train_cmd" \
    data/train data/lang exp/mono exp/mono_ali
fi

# train an LDA+MLLT system.
if [ $stage -le 8 ]; then
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" 2000 30000 \
    data/train data/lang exp/mono_ali exp/mono2

  # decode using the LDA+MLLT model
  (
    utils/mkgraph.sh data/lang exp/mono2 exp/mono2/graph
    for test in dev eval; do
      steps/decode.sh --nj 50 --cmd "$decode_cmd" exp/mono2/graph \
        data/$test exp/mono2/decode_$test
    done
  )&

  steps/align_si.sh  --nj 50 --cmd "$train_cmd" --use-graphs true \
    data/train data/lang exp/mono2 exp/mono2_ali
  exit 0
fi

trainset=train
if [ -f data/${trainset}_spe2e_hires/feats.scp ]; then
  echo "$0: It seems that features for the perturbed training data already exist."
  echo "If you want to extract them anyway, remove them first and run this"
  echo "stage again. Skipping this stage..."
else
  if [ $stage -le 9 ]; then
    echo "$0: perturbing the training data to allowed lengths..."
    utils/data/get_utt2dur.sh data/${trainset}  # necessary for the next command

    # 12 in the following command means the allowed lengths are spaced
    # by 12% change in length.
    utils/data/perturb_speed_to_allowed_lengths.py 12 data/${trainset} \
                                                   data/${trainset}_spe2e_hires
    cat data/${trainset}_spe2e_hires/utt2dur | \
      awk '{print $1 " " substr($1,5)}' >data/${trainset}_spe2e_hires/utt2uniq
    utils/fix_data_dir.sh data/${trainset}_spe2e_hires
  fi

  if [ $stage -le 10 ]; then
    echo "$0: extracting MFCC features for the training data..."
    steps/make_mfcc.sh --nj 50 --mfcc-config conf/mfcc_hires.conf \
                       --cmd "$train_cmd" data/${trainset}_spe2e_hires || exit 1;
    steps/compute_cmvn_stats.sh data/${trainset}_spe2e_hires || exit 1;
    utils/fix_data_dir.sh data/${trainset}_spe2e_hires
    utils/validate_data_dir.sh data/${trainset}_spe2e_hires
  fi
fi

if [ $stage -le 11 ]; then
  echo "$0: extracting MFCC features for the dev/eval data..."
  for datadir in dev eval; do
    utils/copy_data_dir.sh data/$datadir data/${datadir}_hires
  done
  for datadir in dev eval; do
    steps/make_mfcc.sh --nj 30 --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/${datadir}_hires || exit 1;
    steps/compute_cmvn_stats.sh data/${datadir}_hires || exit 1;
    utils/fix_data_dir.sh data/${datadir}_hires
  done
  exit 0
fi

if [ $stage -le 12 ]; then
  echo "$0: Calling the end-to-end chain recipe..."
  local/chain/run_e2e_tdnn_1a.sh
fi


