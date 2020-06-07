#!/bin/bash

# Copyright 2020 Audio, Speech and Language Processing Group (ASLP@NPU), Northwestern Polytechnical University (Authors: Zhuoyuan Yao, Xiong Wang, Jingyong Hou, Lei Xie)
#           2020 AIShell-Foundation (Author: Bengu WU) 
#           2020 Beijing Shell Shell Tech. Co. Ltd. (Author: Hui BU) 
# Apache 2.0

data_aishell=data
data_kws=data/kws
data_train=$data_kws/train
data_test=$data_kws/test
data_local_dict=data/local/dict
data_url=www.openslr.org/resources/33
kws_url=www.openslr.org/resources/85

ali=exp/tri3a_merge_ali
other_ali=exp/tri3a_ali
kws_ali=exp/tri3a_kws_ali
feat_dir=data/fbank

kws_dict=data/dict
kws_lang=data/lang

kws_word=你好米雅
kws_phone="n i3 h ao3 m i3 ii ia3"

stage=0

do_train_aishell1=$true

. ./cmd.sh
. ./path.sh

if [ $stage -le 0 ]; then
    mkdir -p $data_kws
	if [ $do_train_aishell1 ];then
		echo "do train aishell1"
		local/download_and_untar.sh $data_aishell $data_url data_aishell || exit 1;
		local/download_and_untar.sh $data_aishell $data_url resource_aishell || exit 1;
	fi
	local/kws_download_and_untar.sh $data_kws $kws_url dev.tar.gz || exit 1;
	local/kws_download_and_untar.sh $data_kws $kws_url test.tar.gz || exit 1;
	local/kws_download_and_untar.sh $data_kws $kws_url train.tar.gz ||exit 1;
	# You should write your own path to this script
	local/prepare_kws.sh || exit 1;
	local/aishell_data_prep.sh $data_aishell/data_aishell/wav $data/data_aishell/transcript
fi

# Prepare mfcc for aishell and kws
if [ $stage -le 2 ];then
	mfccdir=mfcc
	for x in  train dev test; do
		if [ $do_train_aishell1 ];then
			steps/make_mfcc.sh --cmd "$train_cmd" --nj 10 $data_aishell/$x exp/make_mfcc/aishell/$x $data_aishell/$mfccdir || exit 1;
			steps/compute_cmvn_stats.sh $data_aishell/$x exp/make_mfcc/aishell/$x $data_aishell/$mfccdir || exit 1;
			utils/fix_data_dir.sh $data_aishell/$x || exit 1;
		fi
		if [ ! -f $data_kws/$x/feats.scp ];then
			steps/make_mfcc.sh --cmd "$train_cmd" --nj 10 $data_kws/$x exp/make_mfcc/kws/$x $data_kws/$mfccdir || exit 1;
			steps/compute_cmvn_stats.sh $data_kws/$x exp/make_mfcc/kws/$x $data_kws/$mfccdir || exit 1;
			utils/fix_data_dir.sh $data_kws/$x || exit 1;
		fi
	done
fi

if [ $stage -le 3 ];then
	echo "stage 3"
	for i in train dev test;do
		awk '{print $1,"'$kws_word'"}' $data_kws/$i/wav.scp > $data_kws/$i/text
		# paste -d " " < awk '{print $1}' $data_kws/$i/wav.scp < echo $kws_word > $data_kws/$i/text 
	done
	for i in utt2spk spk2utt feats.scp cmvn.scp text wav.scp;do
		cat $data_kws/train/$i $data_kws/test/$i $data_kws/dev/$i > $data_kws/$i
	done

	mkdir -p data/merge
	for i in train dev test;do
		mkdir -p data/merge/$i
		for j in utt2spk spk2utt feats.scp cmvn.scp wav.scp;do
			cat $data_aishell/$i/$j $data_kws/$i/$j > data/merge/$i/$j
		done
		# utils/combine_data.sh data/merge/$i $data_aishell/$i $data_kws/$i
		awk '{print $1,"<GBG>"}' $data_aishell/$i/text > $data_aishell/$i/text.neg
		cat $data_aishell/$i/text.neg $data_kws/$i/text > data/merge/$i/text
		utils/fix_data_dir.sh data/merge/$i || exit 1;
	done

	awk '{print $1, 0}' $data_aishell/test/wav.scp > data/merge/negative
	awk '{print $1, 1}' $data_kws/test/wav.scp > data/merge/positive

	cat data/merge/negative data/merge/positive | sort > data/merge/label
	rm data/merge/negative
	rm data/merge/positive
		
fi

if [ $stage -le 4 ];then
	mkdir -p $data_local_dict
	cat <<EOF > $data_local_dict/lexicon.txt
SIL sil
<GBG> sil
$kws_word $kws_phone
EOF
	local/prepare_dict.sh $data_local_dict
	utils/prepare_lang.sh --position-dependent-phone false $data_local_dict "<GBG>" data/local/lang data/lang
	mkdir -p data/local/arpa
	cat <<EOF > data/local/arpa/arpa
\data\\
ngram 1=4
ngram 2=2
ngram 3=1

\1-grams:
-0.7781512  <unk>   0
0   <s> -5.8119392
-0.38021123 </s>    0
-0.38021123 ${kws_word}    -0.30103

\2-grams:
-0.1497623  ${kws_word} </s>   0
-3.8828972e-7   <s> ${kws_word}    -5.8119392

\3-grams:
-2.070878e-7    <s> ${kws_word} </s>

\end\\
EOF
	gzip -c data/local/arpa/arpa > data/local/arpa/arpa.gz
	utils/format_lm.sh data/lang data/local/arpa/arpa.gz \
		$data_local_dict/lexicon.txt data/lang_test
fi

if [ $stage -le 5 ];then
	steps/train_mono.sh --cmd "$train_cmd" --nj 50 \
        $data_train $kws_lang exp/mono || exit 1;
fi

# align
if [ $stage -le 6 ];then
    steps/align_si.sh --cmd "$feature_cmd" --nj 40 \
        data/merge/train $kws_lang exp/mono exp/mono_ali || exit 1 ;
fi

# make graph
if [ $stage -le 7 ];then
	utils/mkgraph.sh data/lang_test exp/mono exp/mono/graph || exit 1;
fi

# alignment lattices
if [ $stage -le 8 ];then
	steps/align_fmllr_lats.sh --stage 0 --nj 40 --cmd "${train_cmd}"\
       data/merge/train  data/lang exp/mono exp/datakws_mia_lats
fi
if [ $stage -le 9 ];then
	echo "Extracting feats & Create tr cv set"
    [ ! -d $feat_dir ] && mkdir -p $feat_dir
	mkdir -p $feat_dir/train
	mkdir -p $feat_dir/test
    cp data/merge/train/* $feat_dir/train
    cp data/merge/test/* $feat_dir/test
	rm $feat_dir/train/feats.scp
	rm $feat_dir/train/cmvn.scp
	rm $feat_dir/test/feats.scp
	rm $feat_dir/test/cmvn.scp
    steps/make_fbank.sh --cmd "$feature_cmd" --fbank-config conf/fbank71.conf --nj 50 $feat_dir/train $feat_dir/log $feat_dir/train_feat || exit 1;
    steps/make_fbank.sh --cmd "$feature_cmd" --fbank-config conf/fbank71.conf --nj 50 $feat_dir/test $feat_dir/log $feat_dir/test_feat || exit 1;
    compute-cmvn-stats --binary=false --spk2utt=ark:$feat_dir/train/spk2utt scp:$feat_dir/train/feats.scp ark,scp:$feat_dir/train_feat/cmvn.ark,$feat_dir/train/cmvn.scp || exit 1;
	compute-cmvn-stats --binary=false --spk2utt=ark:$feat_dir/test/spk2utt scp:$feat_dir/test/feats.scp ark,scp:$feat_dir/test_feat/cmvn.ark,$feat_dir/test/cmvn.scp || exit 1;
    utils/fix_data_dir.sh $feat_dir/train
    utils/fix_data_dir.sh $feat_dir/test

fi
# train chain model
if [ $stage -le 10 ];then
	local/chain/run_tdnn.sh 
fi

if [ $stage -le 11 ];then
	best_result=$(cat exp/chain/tdnn_1b_kws/decode_test/scoring_kaldi/best_wer)
	best_lmwt=$(echo ${best_result##*/} | tr '_' ' ' | awk '{print $2}')
	best_penalty=$(echo ${best_result##*/} | tr '_' ' ' | awk '{print $3}')
	local/get_roc.sh $best_lmwt $best_penalty
	python local/kws_draw_roc.py --roc result data/merge/label data/fbank/test/utt2dur
fi
exit 1;


