#!/bin/bash
# Copyright (c) 2016, Johns Hopkins University ( Yenda Trmal <jtrmal@gmail.com> )
# License: Apache 2.0

# Begin configuration section.
stage=2
# End configuration section
. ./utils/parse_options.sh
set -e -o pipefail
set -o nounset                              # Treat unset variables as an error

. ./conf/common_vars.sh
. ./lang.conf

#Example script how to run keyword search using the Kaldi-native pipeline


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
      data/dev10h.pem/ data/lang/ data/dev10h.pem/kwset_${set}

    # we will search for the IV words normally (i.e. will look for the specificsequence
    # of the words
    local/search/compile_keywords.sh  --filter "OOV=0&&Characters>2"\
      data/dev10h.pem/kwset_${set}  data/lang   data/dev10h.pem/kwset_${set}/tmp.2

    # in addition to the direct search of the IV words, we will set up the proxy
    # search as well -- we will use lower nbest, compared to OOV=1
    #-- local/search/compile_proxy_keywords.sh --cmd "$decode_cmd" --category "OOV=0" \
    #--   --beam 5 --nbest 10 --nj 64  --confusion-matrix exp/conf_matrix/confusions.txt  \
    #--   data/dev10h.pem/kwset_${set} data/lang data/dev10h.pem/${set}_oov_kws/tmp/L1.lex  \
    #--   data/dev10h.pem/${set}_oov_kws/tmp/L1.lex data/dev10h.pem/kwset_${set}/tmp.3

    local/search/compile_proxy_keywords.sh --cmd "$decode_cmd" --filter "OOV=1&&Characters>2"\
      --beam 5 --nbest 50 --nj 64  --confusion-matrix exp/conf_matrix/confusions.txt  \
      data/dev10h.pem/kwset_${set} data/lang data/local/lexiconp.txt exp/g2p \
      data/dev10h.pem/kwset_${set}/tmp.4
  done
fi

if [ $stage -le 3 ] ; then
  for set in $kwsets ; do
    fsts-union scp:<(sort data/dev10h.pem/kwset_${set}/tmp*/keywords.scp) \
      ark,t:"|gzip -c >data/dev10h.pem/kwset_${set}/keywords.fsts.gz"
  done
fi




if [ $stage -le 4 ] ; then
  for set in $kwsets ; do
    for it in $(seq 1 4); do
      system=exp/sgmm5_mmi_b0.1/decode_fmllr_dev10h.pem_it$it
      local/search/search.sh --cmd "$decode_cmd" --min-lmwt 9 --max-lmwt 12  \
        --extraid ${set} --indices-dir $system/kws_indices \
        data/lang data/dev10h.pem $system
    done
  done
fi

if [ $stage -le 5 ] ; then
  for set in $kwsets ; do
    system=exp/nnet3/lstm_bidirectional_sp/decode_dev10h.pem
    local/search/search.sh --cmd "$decode_cmd" --min-lmwt 9 --max-lmwt 12  \
      --extraid ${set} --indices-dir $system/kws_index \
      data/lang data/dev10h.pem $system
  done
fi

if [ $stage -le 6 ] ; then
  for set in $kwsets ; do
    system=exp/nnet3/lstm_bidirectional_sp/decode_dev10h.pem_17_8.5/
    local/search/search.sh --cmd "$decode_cmd" --min-lmwt 9 --max-lmwt 12  \
      --extraid ${set} --indices-dir $system/kws_index \
      data/lang data/dev10h.pem $system
  done
fi

