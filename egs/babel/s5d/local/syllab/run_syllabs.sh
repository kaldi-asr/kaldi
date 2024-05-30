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
  odata=${idata%%.*}.syll
else
  odata=${idata%%.*}.syll.${idata#*.}
fi

if [ $stage -le -1 ] ; then
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
fi

if [ $stage -le 0 ] && [ -f "$idir/text" ]; then
  #Create dev10h.syll.pem dir
  steps/align_fmllr.sh \
      --boost-silence $boost_sil --nj $train_nj --cmd "$train_cmd" \
      $idir data/lang exp/tri5 exp/tri5_ali/align_$idata

  local/syllab/ali_to_syllabs.sh \
    --cmd "$decode_cmd" \
    $idir data/lang.syll exp/tri5_ali/align_$idata exp/tri5_ali_syll/align_$idata
fi

if [ $stage -le 1 ] ; then
  utils/copy_data_dir.sh data/$idata data/$odata
  [ -f exp/tri5_ali_syll/align_$idata/text ] && \
    cp exp/tri5_ali_syll/align_$idata/text data/$odata/text
  touch data/$odata/.plp.done
  touch data/$odata/.done
fi


