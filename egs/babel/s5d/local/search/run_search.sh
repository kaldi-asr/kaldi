#!/usr/bin/env bash
# Copyright (c) 2016, Johns Hopkins University ( Yenda Trmal <jtrmal@gmail.com> )
# License: Apache 2.0

# Begin configuration section.
stage=2
dir=dev10h.pem
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

dataset=${dir%%.*}

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
      data/$dir/ data/lang/ data/$dir/kwset_${set}

    # we will search for the IV words normally (i.e. will look for the specificsequence
    # of the words
    local/search/compile_keywords.sh  --filter "OOV=0&&Characters>2"\
      data/$dir/kwset_${set}  data/lang   data/$dir/kwset_${set}/tmp.2

    # in addition to the direct search of the IV words, we will set up the proxy
    # search as well -- we will use lower nbest, compared to OOV=1
    #-- local/search/compile_proxy_keywords.sh --cmd "$decode_cmd" --category "OOV=0" \
    #--   --beam 5 --nbest 10 --nj 64  --confusion-matrix exp/conf_matrix/confusions.txt  \
    #--   data/dev10h.pem/kwset_${set} data/lang data/dev10h.pem/${set}_oov_kws/tmp/L1.lex  \
    #--   data/dev10h.pem/${set}_oov_kws/tmp/L1.lex data/dev10h.pem/kwset_${set}/tmp.3
    if [ -d data/local/extend ]; then
      echo "Detected extended lexicon system..."
      local/search/compile_proxy_keywords.sh --filter "OOV=1&&Characters>2"\
        --cmd "$decode_cmd --mem 24G --max-jobs-run 64" --nj 128 \
        --beam $extlex_proxy_beam --nbest $extlex_proxy_nbest \
        --phone-beam $extlex_proxy_phone_beam --phone-nbest $extlex_proxy_phone_nbest\
        --confusion-matrix exp/conf_matrix/confusions.txt  \
        data/$dir/kwset_${set} data/lang data/local/lexiconp.txt exp/g2p \
        data/$dir/kwset_${set}/tmp.4
    else
      local/search/compile_proxy_keywords.sh --cmd "$decode_cmd" --filter "OOV=1&&Characters>2"\
        --beam 5 --nbest 50 --nj 64  --confusion-matrix exp/conf_matrix/confusions.txt  \
        data/$dir/kwset_${set} data/lang data/local/lexiconp.txt exp/g2p \
        data/$dir/kwset_${set}/tmp.4
    fi

    cut -f 1  data/local/filtered_lexicon.txt | uconv -f utf8 -t utf8 -x Any-Lower | sort -u | \
      nl | awk '{print $2, $1;}' > data/$dir/kwset_${set}/base_words.txt
    paste <(cut -f 1  data/$dir/kwset_${set}/keywords.txt ) \
          <(cut -f 2  data/$dir/kwset_${set}/keywords.txt | \
        uconv -f utf8 -t utf8 -x Any-Lower ) | \
        local/kwords2indices.pl --map-oov 0 data/$dir/kwset_${set}/base_words.txt |\
      perl -ane '
        if (grep (/^0$/, @F[1..$#F])) {print  "$F[0] BaseOOV=1\n";}
        else { print "$F[0] BaseOOV=0\n";}' |\
      cat - data/$dir/kwset_${set}/categories | sort -u |\
      local/search/normalize_categories.pl > data/$dir/kwset_${set}/categories.2
      mv data/$dir/kwset_${set}/categories data/$dir/kwset_${set}/categories.bak
      mv data/$dir/kwset_${set}/categories.2 data/$dir/kwset_${set}/categories

      echo >&2 "Kwset $set processed successfully..."
  done
fi

if [ $stage -le 3 ] ; then
  for set in ${kwsets[@]} ; do
    fsts-union scp:<(sort data/$dir/kwset_${set}/tmp*/keywords.scp) \
      ark,t:"|gzip -c >data/$dir/kwset_${set}/keywords.fsts.gz"
  done
fi


exit

if [ $stage -le 4 ] ; then
  for set in $kwsets ; do
    for it in $(seq 1 4); do
      system=exp/sgmm5_mmi_b0.1/decode_fmllr_$dir_it$it
      local/search/search.sh --cmd "$decode_cmd" --min-lmwt 9 --max-lmwt 12  \
        --extraid ${set} --indices-dir $system/kws_indices \
        data/lang data/$dir $system
    done
  done
fi

if [ $stage -le 5 ] ; then
  for set in $kwsets ; do
    system=exp/nnet3/lstm_bidirectional_sp/decode_$dir
    local/search/search.sh --cmd "$decode_cmd" --min-lmwt 9 --max-lmwt 12  \
      --extraid ${set} --indices-dir $system/kws_indices \
      data/lang data/$dir $system
  done
fi

if [ $stage -le 6 ] ; then
  for set in $kwsets ; do
    system=exp/nnet3/lstm_sp/decode_$dir
    local/search/search.sh --cmd "$decode_cmd" --min-lmwt 10 --max-lmwt 12  \
      --extraid ${set} --indices-dir $system/kws_indices \
      data/lang data/$dir $system
  done
fi
