#!/bin/bash

# Copyright 2020 Audio, Speech and Language Processing Group (ASLP@NPU), Northwestern Polytechnical University (Authors: Zhuoyuan Yao, Xiong Wang, Jingyong Hou, Lei Xie)
#           2020 AIShell-Foundation (Author: Bengu WU) 
#           2020 Beijing Shell Shell Tech. Co. Ltd. (Author: Hui BU) 
# Apache 2.0

# if do_train_aishell is false we should prepare aishell data and tri3a ourselves

data=data
data_aishell=$data
data_kws=data/kws
data_url=www.openslr.org/resources/33
kws_url=www.openslr.org/resources/85

ali=exp/tri3a_merge_ali
other_ali=exp/tri3a_ali
kws_ali=exp/tri3a_kws_ali
feat_dir=data/fbank

kws_dict=data/dict

kws_word_split="你好 米 雅"
kws_word="你好米雅"
kws_phone="n i3 h ao3 m i3 ii ia3"

use_fst=$true

stage=0

# if false we will not train aishell model,
do_train_aishell1=$true

. ./cmd.sh
. ./path.sh

if [ $stage -le 0 ]; then
	mkdir -p $data_kws
	if [ $do_train_aishell1 ];then
		echo "do aishell 1"
		local/download_and_untar.sh $data $data_url data_aishell || exit 1;
		local/download_and_untar.sh $data $data_url resource_aishell || exit 1;
	fi
	local/kws_download_and_untar.sh $data_kws $kws_url dev.tar.gz || exit 1;
	local/kws_download_and_untar.sh $data_kws $kws_url test.tar.gz || exit 1;
	local/kws_download_and_untar.sh $data_kws $kws_url train.tar.gz ||exit 1;
	# You should write your own path to this script
	local/prepare_kws.sh || exit 1;
fi
if [ $stage -le 1 ]; then
	if [ $do_train_aishell1 ];then
		# Lexicon Preparation,
		local/aishell_prepare_dict.sh data/aishell/local || exit 1;

		# Data Preparation,
		local/aishell_data_prep.sh $data/data_aishell/wav $data/data_aishell/transcript || exit 1;

		# Phone Sets, questions, L compilation
		utils/prepare_lang.sh --position-dependent-phones false data/local/dict \
			"<SPOKEN_NOISE>" data/local/lang data/lang || exit 1;

		# LM training
		local/aishell_train_lms.sh || exit 1;

		# G compilation, check LG composition
		utils/format_lm.sh data/lang data/local/lm/3gram-mincount/lm_unpruned.gz \
			data/local/dict/lexicon.txt data/lang_test || exit 1;
	fi
fi

# Prepare mfcc for aishell and kws
if [ $stage -le 2 ];then
	mfccdir=mfcc
	for x in  train dev test; do
		if [ $do_train_aishell1 ];then
			steps/make_mfcc.sh --cmd "$train_cmd" --nj 10 data/$x exp/make_mfcc/$x $mfccdir || exit 1;
			steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir || exit 1;
			utils/fix_data_dir.sh data/$x || exit 1;
		fi
		steps/make_mfcc.sh --cmd "$train_cmd" --nj 10 $data_kws/$x exp/make_mfcc/kws/$x $data_kws/$mfccdir || exit 1;
		steps/compute_cmvn_stats.sh $data_kws/$x exp/make_mfcc/kws/$x $data_kws/$mfccdir || exit 1;
		utils/fix_data_dir.sh $data_kws/$x || exit 1;
	done
fi

if [ $do_train_aishell1 ];then

if [ $stage -le 3 ];then
	steps/train_mono.sh --cmd "$train_cmd" --nj 10 \
		data/train data/lang exp/mono || exit 1;
	steps/align_si.sh --cmd "$train_cmd" --nj 10 \
		data/train data/lang exp/mono exp/mono_ali || exit 1;
fi

if [ $stage -le 4 ];then
	steps/train_deltas.sh --cmd "$train_cmd" \
		2500 20000 data/train data/lang exp/mono_ali exp/tri1 || exit 1;
	steps/align_si.sh --cmd "$train_cmd" --nj 10 \
		data/train data/lang exp/tri1 exp/tri1_ali || exit 1;
fi

if [ $stage -le 5 ];then
	steps/train_deltas.sh --cmd "$train_cmd" \
		2500 20000 data/train data/lang exp/tri1_ali exp/tri2 || exit 1;
	steps/align_si.sh --cmd "$train_cmd" --nj 10 \
		data/train data/lang exp/tri2 exp/tri2_ali || exit 1;
fi

if [ $stage -le 6 ];then
	steps/train_lda_mllt.sh --cmd "$train_cmd" \
		2500 20000 data/train data/lang exp/tri2_ali exp/tri3a || exit 1;
fi

fi

if [ $stage -le 7 ];then
# use aishell tri3a align kws data
	for i in train dev test;do
		echo $kws_word_split
		awk '{print $1}' $data_kws/$i/wav.scp | while read -r line; do echo $line" "$kws_word_split;done >$data_kws/$i/text
		#awk -v word=$kws_word_split '{print $1,word}' $data_kws/$i/wav.scp> $data_kws/$i/text 
	done
	for i in utt2spk spk2utt feats.scp cmvn.scp text wav.scp;do
		cat $data_kws/train/$i $data_kws/test/$i $data_kws/dev/$i > $data_kws/$i
	done
	# utils/combine_data.sh data/merge/$i $data_aishell/$i $data_kws/$i
	mkdir -p data/merge
	for i in train dev test;do
		mkdir -p data/merge/$i
		for j in utt2spk spk2utt feats.scp cmvn.scp text wav.scp;do
			cat $data_aishell/$i/$j $data_kws/$i/$j > data/merge/$i/$j
		done
		utils/fix_data_dir.sh data/merge/$i || exit 1;
	done
	awk '{print $1, 0}' $data_aishell/test/wav.scp > data/merge/negative
	awk '{print $1, 1}' $data_kws/test/wav.scp > data/merge/positive
	test_merge_data=data/merge
	cat $test_merge_data/negative $test_merge_data/positive | sort > data/merge/label
	rm data/merge/negative
	rm data/merge/positive
	
	steps/align_fmllr.sh --cmd "$train_cmd" --nj 50 \
      		data/merge/train $data_aishell/lang exp/tri3a $ali || exit 1;
fi

if [ $stage -le 8 ];then
	[ ! -d $kws_dict ] && mkdir -p $kws_dict;
    echo "Prepare keyword phone & id"

	cat <<EOF > $kws_dict/lexicon.txt
sil sil 
<SPOKEN_NOISE> sil
<gbg> <GBG>
$kws_word $kws_phone
EOF
	cat <<EOF > $kws_dict/hotword.lexicon
$kws_word $kws_phone
EOF
	echo "<eps> 0
sil 1" > $kws_dict/phones.txt
	count=2
    awk '{for(i=2;i<=NF;i++){if(!match($i,"sil"))print $i}}' $kws_dict/lexicon.txt | sort | uniq  | while read -r line;do
		echo "$line $count"
		count=$(($count+1))
	done >> $kws_dict/phones.txt
	cat <<EOF > $kws_dict/words.txt
<eps> 0
<gbg> 1
$kws_word 2
EOF
fi

if [ $stage -le 9 ];then
	echo "merge and change alignment"
    awk -v hotword_phone=$kws_dict/phones.txt \
    'BEGIN {
        while (getline < hotword_phone) {
            map[$1] = $2 
        }
    }
    {
        if(!match($1, "#") && !match($1, "<")) { 
			if(match($1, "sil"))
			{
				printf("%s %s\n", $2, 1)
			}
			else
			{
				printf("%s %s\n", $2, map[$1] != "" ? map[$1] : 2)
			}
        }
    }
    ' $data_aishell/lang/phones.txt > data/phone.map
	mkdir -p exp/kws_ali_test
	cur=$(cat $ali/num_jobs)
	for x in $(seq 1 $cur);do
		gunzip -c $ali/ali.$x.gz | 
		ali-to-phones --per-frame=true exp/tri3a/final.mdl ark:- t,ark:- | 
		utils/apply_map.pl -f 2- data/phone.map |
		copy-int-vector t,ark:- ark,scp:exp/kws_ali_test/ali.$x.ark,exp/kws_ali_test/ali.$x.scp 
	done
	cat exp/kws_ali_test/ali.*.scp | sort -k 1 > exp/kws_ali_test/ali.scp
	cp $ali/final.mdl exp/kws_ali_test || exit 1;
	cp $ali/num_jobs exp/kws_ali_test || exit 1;
	cp $ali/tree exp/kws_ali_test || exit 1;
	cp $kws_dict/phones.txt exp/kws_ali_test || exit 1;
	ali=exp/kws_ali_test
fi

if [ $stage -le 10 ]; then
	    echo "python local/gen_text_fst.py data/dict/hotword.lexicon data/dict/hotword.text.fst"
		python local/gen_text_fst.py data/dict/hotword.lexicon data/dict/fst.txt 
		fstcompile \
			--isymbols=data/dict/phones.txt \
			--osymbols=data/dict/words.txt  \
			data/dict/fst.txt | fstdeterminize | fstminimizeencoded > data/dict/hotword.openfst || exit 1;
		fstprint data/dict/hotword.openfst data/dict/hotword.fst.txt
fi
if [ $stage -le 11 ];then
	echo "Extracting feats & Create tr cv set"
	[ ! -d $feat_dir ] && mkdir -p $feat_dir
    [ ! -d data/wav ] && ln -s $data_aishell/wav data/
    cp -r data/merge/train $feat_dir/train
    cp -r data/merge/test $feat_dir/test
    steps/make_fbank.sh --cmd "$train_cmd" --fbank-config conf/fbank71.conf --nj 50 $feat_dir/train $feat_dir/log $feat_dir/train_feat || exit 1;
    steps/make_fbank.sh --cmd "$train_cmd" --fbank-config conf/fbank71.conf --nj 50 $feat_dir/test $feat_dir/log $feat_dir/test_feat || exit 1;
    compute-cmvn-stats --binary=false --spk2utt=ark:$feat_dir/train/spk2utt scp:$feat_dir/train/feats.scp ark,scp:$feat_dir/train_feat/cmvn.ark,$feat_dir/train/cmvn.scp || exit 1;
	utils/fix_data_dir.sh $feat_dir/train
	utils/fix_data_dir.sh $feat_dir/test
fi
# train
if [ $stage -le 12 ];then
	num_targets=$(wc -l $kws_dict/phones.txt)
	local/nnet3/run_tdnn.sh --num_targets $num_targets
fi

# p
if [ $stage -le 13 ];then
	steps/nnet3/make_bottleneck_features.sh  \
		--use_gpu true \
		--nj 1 \
		output.log-softmax \
		data/fbank/test \
 		data/fbank/test_bnf \
 		exp/nnet3/tdnn_test_kws \
 		exp/bnf/log \
 		exp/bnf || exit 1;
fi

if [ $stage -le 14 ];then
	copy-matrix ark:exp/bnf/raw_bnfeat_test.1.ark t,ark:exp/bnf/ark.txt
	if [ $use_fst ];then
		python local/run_fst.py data/dict/hotword.fst.txt exp/bnf/ark.txt > result.txt
	else
		python local/kws_posterior_handling.py exp/bnf/ark.txt
	fi
	python local/kws_draw_roc.py --roc result.txt data/merge/label data/fbank/test/utt2dur
fi
