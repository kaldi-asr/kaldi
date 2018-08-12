#!/bin/bash

set -e
stage=0
nj=80

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh
mkdir -p data/{train,train_unsup,test}/data
mkdir -p data/local/backup
if [ $stage -le 0 ]; then
  local/prepare_data.sh --language tamil
fi

if [ $stage -le 1 ]; then
  image/get_image2num_frames.py --feat-dim 40 data/train
  image/get_allowed_lengths.py --frame-subsampling-factor 4 10 data/train

  for dataset in test train train_unsup; do
    echo "$0: Extracting features and calling compute_cmvn_stats for dataset:  $dataset. "
    echo "Date: $(date)."
    #local/extract_features.sh --nj $nj --cmd $cmd --feat-dim 40 data/$dataset
    local/make_features.py data/$dataset/images.scp --feat-dim 40 \
      --allowed_len_file_path data/$dataset/allowed_lengths.txt --no-augment | \
      copy-feats --compress=true --compression-method=7 \
        ark:- ark,scp:data/$dataset/data/images.ark,data/$dataset/feats.scp
    steps/compute_cmvn_stats.sh data/$dataset || exit 1;
  done
  utils/fix_data_dir.sh data/train
fi

if [ $stage -le 2 ]; then
  echo "$0: Preparing bpe data $(date)"
  cp -r data/train data/local/backup/
  cp -r data/test data/local/backup/
  cut -d' ' -f2- data/train/text | python3 local/get_phones.py > data/local/text/cleaned/phones.txt
  cut -d' ' -f2- data/train/text > data/local/text/cleaned/train.txt
  cat data/local/text/ta.txt | python3 local/process_corpus.py > data/local/text/cleaned/corpus.txt
  cat data/local/text/val.txt | python3 local/process_corpus.py > data/local/text/cleaned/val.txt
  cat data/local/text/cleaned/phones.txt data/local/text/cleaned/train.txt | python3 local/prepend_words.py | python3 utils/lang/bpe/learn_bpe.py -s 700 > data/train/bpe.out
  for datasplit in test train; do
      cut -d' ' -f1 data/$datasplit/text > data/$datasplit/ids
      cut -d' ' -f2- data/$datasplit/text | python3 local/prepend_words.py | python3 utils/lang/bpe/apply_bpe.py -c data/train/bpe.out | sed 's/@@//g' > data/$datasplit/bpe_text
      mv data/$datasplit/text data/$datasplit/text.old
      paste -d' ' data/$datasplit/ids data/$datasplit/bpe_text > data/$datasplit/text
  done
  echo "$0: Preparing corpus"
  cat data/local/text/cleaned/corpus.txt | python3 local/prepend_words.py | python3 utils/lang/bpe/apply_bpe.py -c data/train/bpe.out | sed 's/@@//g' > data/local/text/cleaned/bpe_corpus.txt
  cat data/local/text/cleaned/val.txt | python3 local/prepend_words.py | python3 utils/lang/bpe/apply_bpe.py -c data/train/bpe.out | sed 's/@@//g' > data/local/text/cleaned/bpe_val.txt
  echo "$0: Preparing dictionary and lang..."
  local/prepare_dict.sh --dir data/local/dict
  utils/prepare_lang.sh --num-sil-states 4 --num-nonsil-states 8 --sil-prob 0.0 --position-dependent-phones false \
      data/local/dict "<sil>" data/lang/temp data/lang
  utils/lang/bpe/add_final_optional_silence.sh --final-sil-prob 0.5 data/lang
fi

if [ $stage -le 3 ]; then
  echo "$0: Estimating a language model for decoding...$(date)"
  local/train_lm.sh --dir data/local/local_lm --order 3
  utils/format_lm.sh data/lang data/local/local_lm/data/arpa/3gram_unpruned.arpa.gz \
      data/local/dict/lexicon.txt data/lang_test
fi

if [ $stage -le 4 ]; then
  echo "$0: Calling the flat-start chain recipe...$(date)"
  local/chain/run_flatstart_cnn1a.sh --nj $nj
fi

if [ $stage -le 5 ]; then
  echo "$0: Aligning the training data using the e2e chain model... $(date)"
  steps/nnet3/align.sh --nj $nj --cmd "$cmd" \
      --scale-opts '--transition-scale=1.0 --acoustic-scale=1.0 --self-loop-scale=1.0' \
      data/train data/lang exp/chain/e2e_cnn_1a exp/chain/e2e_ali_train
fi

affix=_1b
decode_dir=decode_test
if [ $stage -le 6 ]; then
  echo "$0: Building a tree and training a regular chain model using the e2e alignments...$(date)"
  local/chain/run_cnn_e2eali_1b.sh --nj $nj --affix $affix --decode_dir $decode_dir
fi
