#!/usr/bin/env bash
# Copyright (c) 2015, Johns Hopkins University ( Yenda Trmal <jtrmal@gmail.com> )
# License: Apache 2.0

# Begin configuration section.
cmd=run.pl
unk="<unk>"
# End configuration section
. ./utils/parse_options.sh

set -e -o pipefail
set -o nounset                              # Treat unset variables as an error

datadir=$1
langdir=$2
idict=$3
amdir=$4
odict=$5
olocallang=$6
olang=$7


mkdir -p $odict
mkdir -p $olang
mkdir -p $olocallang
steps/get_prons.sh --cmd "$cmd" $datadir $langdir $amdir
utils/dict_dir_add_pronprobs.sh --max-normalize true $idict  \
  $amdir/pron_counts_nowb.txt $amdir/sil_counts_nowb.txt \
  $amdir/pron_bigram_counts_nowb.txt $odict

utils/prepare_lang.sh  --phone-symbol-table $langdir/phones.txt \
  $odict "$unk" $olocallang $olang

