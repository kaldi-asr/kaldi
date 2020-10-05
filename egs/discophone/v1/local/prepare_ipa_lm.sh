#!/usr/bin/env bash
# Copyright 2020  Johns Hopkins University (Author: Piotr Żelasko)
# Apache 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

set -eou pipefail

train_set=
phone_token_opt='--phones'
order=2

. path.sh
. cmd.sh
. utils/parse_options.sh

function langname() {
  echo "$(basename "$1")"
}

# Prepare phone lexicons and training texts for each language
mkdir -p data/ipa_lm
for data_dir in $train_set; do
  lang_name=$(langname $data_dir)
  # Create a lexicon directory with LM training IPA texts
  python3 local/prepare_ipa_lm_text.py \
    $phone_token_opt \
    data/$data_dir/lexicon_ipa_all.txt \
    data/$data_dir/text.bkp \
    data/ipa_lm/train/$lang_name
  # Create a lexicon directory with LM dev IPA texts - we will ignore the lexicons here and just use the texts
  dev_data_dir=${data_dir//train/dev}
  python3 local/prepare_ipa_lm_text.py \
    $phone_token_opt \
    data/$data_dir/lexicon_ipa_all.txt \
    data/$dev_data_dir/text.bkp \
    data/ipa_lm/dev/$lang_name
done

# Concatenate all languages IPA train texts and train the LM
mkdir -p data/ipa_lm/train_all
find data/ipa_lm/train -name phones_text -print0 | xargs -0 cat >data/ipa_lm/train_all/text
ngram-count \
  -text data/ipa_lm/train_all/text \
  -order $order \
  -unk \
  -map-unk "<unk>" \
  -interpolate \
  -lm data/ipa_lm/train_all/srilm.o${order}g.kn.gz
# We don't use KN discounting because of an issue with the backoff estimation...
#  -kndiscount

# Evaluate the multilingual LM perplexity on the dev sets
for data_dir in $train_set; do
  lang_name=$(langname $data_dir)
  dev_lm_dir=data/ipa_lm/dev/$lang_name
  ngram \
    -lm data/ipa_lm/train_all/srilm.o${order}g.kn.gz \
    -unk \
    -ppl $dev_lm_dir/phones_text \
    2>&1 | tee data/ipa_lm/train_all/dev_ppl${order}_${lang_name}
done
