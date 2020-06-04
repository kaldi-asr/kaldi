#!/bin/bash

# Copyright 2020 Audio, Speech and Language Processing Group (ASLP@NPU), Northwestern Polytechnical University(Authors: Zhuoyuan Yao, Xiong Wang, Jingyong Hou, Lei Xie)
#           2020 AIShell-Foundation(Authors:Bengu WU) 
#           2020 Beijing Shell Shell Tech. Co. Ltd. (Author: Hui BU) 
# Apache 2.0

# get wav.scp utt2spk spk2utt here, we also just get _02_ _10_ mic because we don't need so much data

stage=0

data_train_root=data/kws/train
data_dev_root=data/kws/dev
data_test_root=data/kws/test

data_train=data/kws/train
data_test=data/kws/test
data_dev=data/kws/dev
mkdir -p $data_train
mkdir -p $data_test
mkdir -p $data_dev
echo "prepare kws data"
if [ $stage -le 0 ];then
    
    if [ -f $data_train_root/SPEECHDATA/train.scp ];then
        awk '{print "data/kws/train/SPEECHDATA/"$0 }' $data_train_root/SPEECHDATA/train.scp | grep -E '_02_|_10_' > $data_train_root/SPEECHDATA/train_p.scp
		mv $data_train_root/SPEECHDATA/train_p.scp $data_train_root/wav.scp
    fi
    if [ -f $data_dev_root/SPEECHDATA/dev.scp ];then
        awk '{print "data/kws/dev/SPEECHDATA/"$0 }' $data_dev_root/SPEECHDATA/dev.scp | grep -E '_02_|_10_'  > $data_dev_root/SPEECHDATA/dev_p.scp
        mv $data_dev_root/SPEECHDATA/dev_p.scp $data_dev_root/wav.scp
	fi
    awk '{print "data/kws/test/wav/"$0}' $data_test_root/wav.scp  > $data_test_root/test_p.scp
	mv $data_test_root/test_p.scp $data_test/wav.scp
fi

if [ $stage -le 1 ];then
    for i in $data_train $data_dev $data_test;do
        while read -r line;do
            line_p1=${line##*/}
            echo ${line_p1%.*} $line
        done < $i/wav.scp  > $i/wav_terminal.scp
        mv $i/wav_terminal.scp $i/wav.scp
    done
fi

if [ $stage -le 2 ];then
    for i in $data_train $data_dev $data_test;do
		awk '{split($1,array,"_");print $1,array[1]}' $i/wav.scp> $i/utt2spk
        utils/utt2spk_to_spk2utt.pl $i/utt2spk > $i/spk2utt
		utils/fix_data_dir.sh $i
	done
fi
