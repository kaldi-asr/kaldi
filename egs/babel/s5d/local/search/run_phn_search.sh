#!/bin/bash
# Copyright (c) 2016, Johns Hopkins University ( Yenda Trmal <jtrmal@gmail.com> )
# License: Apache 2.0

# Begin configuration section.
stage=2
# End configuration section
. ./conf/common_vars.sh
. ./utils/parse_options.sh
set -e -o pipefail
set -o nounset                              # Treat unset variables as an error

. ./lang.conf

#Example script how to run keyword search using the Kaldi-native pipeline

lang=data/lang.phn
data=data/dev10h.phn.pem

if [ $stage -le 0 ]; then
  local/generate_confusion_matrix.sh --nj 64 --cmd "$decode_cmd" \
    exp/sgmm5_denlats/dengraph/ exp/sgmm5 exp/sgmm5_ali exp/sgmm5_denlats exp/conf_matrix
fi

if [ $stage -le 1 ] ; then
  local/train_g2p.sh --cmd "$decode_cmd" data/local/lexicon.txt  exp/g2p
fi

kwsets=${!dev10h_kwlists[@]}
echo "$kwsets"
if [ $stage -le 2 ] ; then

  for set in $kwsets ; do

    #This will set up the basic files and converts the F4DE files into Kaldi-native format
    local/search/setup.sh $dev10h_ecf_file $dev10h_rttm_file  "${dev10h_kwlists[$set]}" \
      $data $lang  $data/kwset_${set}

    cat data/dev10h.pem/kwset_${set}/categories | \
      local/search/normalize_categories.pl --one-per-line | \
      grep OOV | sed 's/OOV/BaseOOV/g' | cat - $data/kwset_kwlist/categories | \
      local/search/normalize_categories.pl > $data/kwset_kwlist/categories.new
    mv $data/kwset_kwlist/categories.new $data/kwset_kwlist/categories

    # we will search for the IV words normally (i.e. will look for the specificsequence
    # of the words
    local/search/compile_keywords.sh  --filter "OOV=0&&Characters>2"\
      $data/kwset_${set}  $lang   $data/kwset_${set}/tmp.2

    # in addition to the direct search of the IV words, we will set up the proxy
    # search as well -- we will use lower nbest, compared to OOV=1
    #-- local/search/compile_proxy_keywords.sh --cmd "$decode_cmd" --category "OOV=0" \
    #--   --beam 5 --nbest 10 --nj 64  --confusion-matrix exp/conf_matrix/confusions.txt  \
    #--   ${data}/kwset_${set} ${lang} ${data}/${set}_oov_kws/tmp/L1.lex  \
    #--   ${data}/${set}_oov_kws/tmp/L1.lex ${data}/kwset_${set}/tmp.3

    local/search/compile_proxy_keywords.sh --cmd "$decode_cmd" --filter "OOV=1&&Characters>4"\
      --beam 5 --nbest 100 --nj 64  --confusion-matrix exp/conf_matrix/confusions.txt  \
      ${data}/kwset_${set} ${lang} data/local/dict.phn/lexiconp.txt exp/g2p \
      ${data}/kwset_${set}/tmp.4
  done
fi

if [ $stage -le 3 ] ; then
  for set in $kwsets ; do
    fsts-union scp:<(sort ${data}/kwset_${set}/tmp*/keywords.scp) \
      ark,t:"|gzip -c >${data}/kwset_${set}/keywords.fsts.gz"
  done
fi
exit
if [ $stage -le 4 ] ; then
  for set in $kwsets ; do
    for it in $(seq 1 4); do
      system=exp/sgmm5_mmi_b0.1/decode_fmllr_$(basename $data)_it$it
      local/search/search.sh --cmd "$decode_cmd" --min-lmwt 9 --max-lmwt 12  \
        --extraid ${set} --indices-dir $system/kws_indices ${lang} ${data} $system
    done
  done
fi

if [ $stage -le 5 ] ; then
  for set in $kwsets ; do
    system=exp/nnet3/lstm_bidirectional_sp/decode_dev10h.phn.pem
    local/search/search.sh --cmd "$decode_cmd" --min-lmwt 10 --max-lmwt 12  \
      --extraid ${set} --indices-dir $system/kws_indices $lang $data $system
  done
fi

if [ $stage -le 6 ] ; then
  for set in $kwsets ; do
    system=exp/nnet3/lstm_bidirectional_sp/decode_dev10h.phn.pem_17_8.5
    local/search/search.sh --cmd "$decode_cmd" --min-lmwt 10 --max-lmwt 12  \
      --extraid ${set} --indices-dir $system/kws_indices $lang $data $system
  done
fi

if [ $stage -le 7 ] ; then
  for set in $kwsets ; do
    system=exp/nnet3/lstm_bidirectional_sp/decode_dev10h.phn.pem.bg
    local/search/search.sh --cmd "$decode_cmd" --min-lmwt 10 --max-lmwt 12  \
      --extraid ${set} --indices-dir $system/kws_indices $lang $data $system
  done
fi

if [ $stage -le 8 ] ; then
  for set in $kwsets ; do
    system=exp/tri6_nnet/decode_dev10h.phn.pem
    local/search/search.sh --cmd "$decode_cmd" --min-lmwt 10 --max-lmwt 12  \
      --extraid ${set} --indices-dir $system/kws_indices $lang $data $system
  done
fi

