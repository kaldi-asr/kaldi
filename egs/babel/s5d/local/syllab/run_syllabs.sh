#!/bin/bash
# Copyright (c) 2015, Johns Hopkins University ( Yenda Trmal <jtrmal@gmail.com> )
# License: Apache 2.0

# Begin configuration section.
# End configuration section
set -e -o pipefail
set -o nounset                              # Treat unset variables as an error

. ./cmd.sh
. ./path.sh

. ./conf/common_vars.sh
. ./lang.conf

local/syllab/generate_syllable_lang.sh \
    data/train data/local/ data/lang data/lang.syll data/local/dict.syll

local/syllab/ali_to_syllabs.sh \
    data/train data/lang.syll exp/tri5_ali exp/tri5_ali_syll


utils/copy_data_dir.sh data/train data/train.syll
cp exp/tri5_ali_syll/text data/train.syll/text

#Create syllab LM
local/train_lms_srilm.sh \
    --words-file data/lang.syll/words.txt --train-text data/train.syll/text \
    --oov-symbol "`cat data/lang.syll/oov.txt`"  data data/srilm.syll

local/arpa2G.sh  data/srilm.syll/lm.gz  data/lang.syll/ data/lang.syll/

#Create dev10h.syll.pem dir
steps/align_fmllr.sh \
    --boost-silence $boost_sil --nj $train_nj --cmd "$train_cmd" \
    data/dev10h.pem data/lang exp/tri5 exp/tri5_ali/align_dev10h.pem

local/syllab/ali_to_syllabs.sh \
  --cmd "$decode_cmd" \
  data/dev10h.pem data/lang.syll exp/tri5_ali/align_dev10h.pem exp/tri5_ali_syll/align_dev10h.pem

utils/copy_data_dir.sh data/dev10h.pem data/dev10h.syll.pem
cp exp/tri5_ali_syll/align_dev10h.pem/text data/dev10h.syll.pem/text
touch data/dev10h.syll.pem/.plp.done
touch data/dev10h.syll.pem/.done



