#!/usr/bin/env bash
# Copyright (c) 2015, Johns Hopkins University ( Yenda Trmal <jtrmal@gmail.com> )
# License: Apache 2.0

# Begin configuration section.
stage=0
# End configuration section
. ./utils/parse_options.sh
set -e -o pipefail
set -o nounset                              # Treat unset variables as an error

. ./cmd.sh
. ./path.sh

. ./conf/common_vars.sh
. ./lang.conf

if [ $# -ne 1 ] ; then
  echo "Invalid number of parameters"
  exit 1
fi

idir=$1

if [ ! -d "$idir" ] ; then
  echo "The directory $idir does not exist"
  exit 1
fi

idata=${idir##*/}


if [ "$idata" == ${idata%%.*} ]; then
  odata=${idata%%.*}.phn
else
  odata=${idata%%.*}.phn.${idata#*.}
fi

if [ $stage -le -1 ] ; then
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
fi

if [ $stage -le 0 ] && [ -f "$idir/text" ] ; then
  #Create dev10h.phn.pem dir
  steps/align_fmllr.sh \
      --boost-silence $boost_sil --nj $train_nj --cmd "$train_cmd" \
      $idir data/lang exp/tri5 exp/tri5_ali/align_$idata

  local/syllab/ali_to_syllabs.sh \
    --cmd "$decode_cmd" \
    $idir data/lang.phn exp/tri5_ali/align_$idata exp/tri5_ali_phn/align_$idata
fi

if [ $stage -le 1 ] ; then
  utils/copy_data_dir.sh data/$idata data/$odata
  [ -f exp/tri5_ali_phn/align_$idata/text ] && \
    cp exp/tri5_ali_phn/align_$idata/text data/$odata/text
  touch data/$odata/.plp.done
  touch data/$odata/.done
fi


