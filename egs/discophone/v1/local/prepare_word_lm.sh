#!/usr/bin/env bash
# Copyright 2020  Johns Hopkins University (Author: Piotr Żelasko)
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

set -eou pipefail

train_set=
order=3

. path.sh
. cmd.sh
. utils/parse_options.sh

function langname() {
  echo "$(basename "$1")"
}

# Prepare training texts for each language
for data_dir in $train_set; do
  lang_name=$(langname $data_dir)
  # Create a directory with LM training language-suffix-words texts
  # Train set
  mkdir -p data/word_lm/train/$lang_name
  cat data/$data_dir/text.bkp_suffix | cut -f2- -d' ' >data/word_lm/train/$lang_name/word_suffix_text
  # Dev set
  dev_data_dir=${data_dir//train/dev}
  mkdir -p data/word_lm/dev/$lang_name
  cat data/$dev_data_dir/text.bkp_suffix | cut -f2- -d' ' >data/word_lm/dev/$lang_name/word_suffix_text
done

# Concatenate all languages train texts and train the LM.
# Uses the words with language suffixes, like "hi_English amy_English"
mkdir -p data/word_lm/train_all
find data/word_lm/train -name word_suffix_text -print0 | xargs -0 cat >data/word_lm/train_all/text
ngram-count \
  -text data/word_lm/train_all/text \
  -order $order \
  -unk \
  -map-unk "<unk>" \
  -interpolate \
  -kndiscount \
  -lm data/word_lm/train_all/srilm.o${order}g.kn.gz

# Evaluate the multilingual LM perplexity on the dev sets
for data_dir in $train_set; do
  lang_name=$(langname $data_dir)
  dev_lm_dir=data/word_lm/dev/$lang_name
  ngram \
    -lm data/word_lm/train_all/srilm.o${order}g.kn.gz \
    -unk \
    -ppl $dev_lm_dir/word_suffix_text \
    2>&1 | tee data/word_lm/train_all/dev_ppl${order}_${lang_name}
done
