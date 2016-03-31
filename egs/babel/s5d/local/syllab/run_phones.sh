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

local/syllab/generate_phone_lang.sh \
    data/train data/local/ data/lang data/lang.phn data/local/dict.phn

local/syllab/ali_to_syllabs.sh \
    data/train data/lang.phn exp/tri5_ali exp/tri5_ali_phn


utils/copy_data_dir.sh data/train data/train.phn
cp exp/tri5_ali_phn/text data/train.phn/text

#Create syllab LM
local/train_lms_srilm.sh \
    --words-file data/lang.phn/words.txt --train-text data/train.phn/text \
    --oov-symbol "`cat data/lang.phn/oov.txt`"  data data/srilm.phn

local/arpa2G.sh  data/srilm.phn/lm.gz  data/lang.phn/ data/lang.phn/

#Create dev10h.phn.pem dir
steps/align_fmllr.sh \
    --boost-silence $boost_sil --nj $train_nj --cmd "$train_cmd" \
    data/dev10h.pem data/lang exp/tri5 exp/tri5_ali/align_dev10h.pem

local/syllab/ali_to_syllabs.sh \
  --cmd "$decode_cmd" \
  data/dev10h.pem data/lang.phn exp/tri5_ali/align_dev10h.pem exp/tri5_ali_phn/align_dev10h.pem

utils/copy_data_dir.sh data/dev10h.pem data/dev10h.phn.pem
cp exp/tri5_ali_phn/align_dev10h.pem/text data/dev10h.phn.pem/text
touch data/dev10h.phn.pem/.plp.done
touch data/dev10h.phn.pem/.done



