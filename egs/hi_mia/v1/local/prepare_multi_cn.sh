#!/bin/bash

set -e
. ./path.sh

stage=0
aidatatang_url=www.openslr.org/resources/62
aishell_url=www.openslr.org/resources/33
magicdata_url=www.openslr.org/resources/68
primewords_url=www.openslr.org/resources/47
stcmds_url=www.openslr.org/resources/38
thchs_url=www.openslr.org/resources/18

. ./utils/parse_options.sh

dbase=$1

mkdir -p $dbase/{aidatatang,aishell,magicdata,primewords,stcmds,thchs}

if [ $stage -le 0 ]; then
  # download all training data
  local/aidatatang_download_and_untar.sh $dbase/aidatatang $aidatatang_url aidatatang_200zh || exit 1;
  local/aishell_download_and_untar.sh $dbase/aishell $aishell_url data_aishell || exit 1;
  local/magicdata_download_and_untar.sh $dbase/magicdata $magicdata_url train_set || exit 1;
  local/primewords_download_and_untar.sh $dbase/primewords $primewords_url || exit 1;
  local/stcmds_download_and_untar.sh $dbase/stcmds $stcmds_url || exit 1;
  local/thchs_download_and_untar.sh $dbase/thchs $thchs_url data_thchs30 || exit 1;

  # download all test data
  local/thchs_download_and_untar.sh $dbase/thchs $thchs_url test-noise || exit 1;
  local/magicdata_download_and_untar.sh $dbase/magicdata $magicdata_url dev_set || exit 1;
  local/magicdata_download_and_untar.sh $dbase/magicdata $magicdata_url test_set || exit 1;
fi

if [ $stage -le 1 ]; then
  local/aidatatang_data_prep.sh $dbase/aidatatang/aidatatang_200zh data/aidatatang || exit 1;
  local/aishell_data_prep.sh $dbase/aishell/data_aishell data/aishell || exit 1;
  local/thchs-30_data_prep.sh $dbase/thchs/data_thchs30 data/thchs || exit 1;
  local/magicdata_data_prep.sh $dbase/magicdata data/magicdata || exit 1;
  local/primewords_data_prep.sh $dbase/primewords data/primewords || exit 1;
  local/stcmds_data_prep.sh $dbase/stcmds data/stcmds || exit 1;
fi

if [ $stage -le 2 ]; then
  # normalize transcripts
  utils/combine_data.sh data/train_combined \
    data/{aidatatang,aishell,magicdata,primewords,stcmds,thchs}/train || exit 1;
  utils/fix_data_dir.sh data/train_combined || exit 1;
  utils/utt2spk_to_spk2utt.pl data/train_combined/utt2spk > data/train_combined/spk2utt || exit 1;
  echo "$0: openSLR training data created. "
fi

exit 0;
