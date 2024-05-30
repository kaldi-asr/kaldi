#!/usr/bin/env bash
# Copyright (c) 2016, Johns Hopkins University ( Yenda Trmal <jtrmal@gmail.com> )
# License: Apache 2.0

# Begin configuration section.
stage=2
dir=dev10h.pem
# End configuration section
. ./conf/common_vars.sh
. ./utils/parse_options.sh
set -e -o pipefail
set -o nounset                              # Treat unset variables as an error

. ./lang.conf

#Example script how to run keyword search using the Kaldi-native pipeline


if [ $stage -le 0 ]; then
  local/generate_confusion_matrix.sh --nj 64 --cmd "$decode_cmd" \
    exp/sgmm5_denlats/dengraph/ exp/sgmm5 exp/sgmm5_ali exp/sgmm5_denlats exp/conf_matrix
fi

if [ $stage -le 1 ] ; then
  local/train_g2p.sh --cmd "$decode_cmd" data/local/lexicon.txt  exp/g2p
fi

dataset=${dir%%.*}
datatype=${dir#*.}

lang=data/lang.phn
if [ "$dir" == "$dataset" ]; then
  data=data/${dataset}.phn
else
  data=data/${dataset}.phn.${datatype}
fi

set +o nounset
eval kwsets=${!dataset_kwlists[@]}
eval my_ecf_file=\$${dataset}_ecf_file
eval my_rttm_file=\$${dataset}_rttm_file
set -o nounset

my_array_name=${dataset}_kwlists

eval kwsets=\( \${!$my_array_name[@]} \)
declare -p kwsets
for set in ${kwsets[@]} ; do
  eval my_kwlist=\${$my_array_name[$set]}
  declare -p my_kwlist
done
declare -p my_ecf_file
declare -p my_rttm_file

if [ $stage -le 2 ] ; then

  for set in ${kwsets[@]} ; do

    eval my_kwlist=\${$my_array_name[$set]}

    #This will set up the basic files and converts the F4DE files into Kaldi-native format
    local/search/setup.sh $my_ecf_file $my_rttm_file  "${my_kwlist}" \
      $data $lang  $data/kwset_${set}

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

    if [ -d data/local/extend ]; then
      echo "Detected extended lexicon system..."
      local/search/compile_proxy_keywords.sh \
        --cmd "$decode_cmd" --nj 64 --filter "OOV=1&&Characters>4" \
        --beam $extlex_proxy_beam --nbest $extlex_proxy_nbest \
        --phone-beam $extlex_proxy_phone_beam --phone-nbest $extlex_proxy_phone_nbest\
        --confusion-matrix exp/conf_matrix/confusions.txt  \
        ${data}/kwset_${set} ${lang} data/local/dict.phn/lexiconp.txt exp/g2p \
        ${data}/kwset_${set}/tmp.4
    else
      local/search/compile_proxy_keywords.sh \
        --cmd "$decode_cmd" --nj 64 --filter "OOV=1&&Characters>4" \
        --beam $proxy_beam --nbest $proxy_nbest \
        --phone-beam $proxy_phone_beam --phone-nbest $proxy_phone_nbest\
        --confusion-matrix exp/conf_matrix/confusions.txt  \
        ${data}/kwset_${set} ${lang} data/local/dict.phn/lexiconp.txt exp/g2p \
        ${data}/kwset_${set}/tmp.4
    fi

    # and finally, replace the categories by the word-level categories
    cp data/${dir}/kwset_${set}/categories $data/kwset_${set}/categories
  done
fi

if [ $stage -le 3 ] ; then
  for set in ${kwsets[@]} ; do
    fsts-union scp:<(sort $data/kwset_${set}/tmp*/keywords.scp) \
      ark,t:"|gzip -c >$data/kwset_${set}/keywords.fsts.gz"
  done
fi


echo "Directories are set up -- running run-4-phn-anydecode.sh will take care of the rest"
exit 0

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

