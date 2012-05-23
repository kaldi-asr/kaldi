#!/bin/bash

mkdir -p data/local
local=`pwd`/local
scripts=`pwd`/scripts

export PATH=$PATH:`pwd`/../../../tools/irstlm/bin

train_base_name=train_yesno
test_base_name=test_yesno
waves_dir=$1

cd data/local

ls -1 $waves_dir > waves_all.list
../../local/create_yesno_waves_test_train.pl waves_all.list waves.test waves.train

../../local/create_yesno_wav_scp.pl ${waves_dir} waves.test > ${test_base_name}_wav.scp

../../local/create_yesno_wav_scp.pl ${waves_dir} waves.train > ${train_base_name}_wav.scp

../../local/create_yesno_txt.pl waves.test > ${test_base_name}.txt

../../local/create_yesno_txt.pl waves.train > ${train_base_name}.txt

cp ../../input/task.arpabo lm_tg.arpa

cd ../..

echo "Data preparation succeeded"
