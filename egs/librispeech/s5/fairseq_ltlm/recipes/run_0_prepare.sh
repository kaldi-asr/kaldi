#!/bin/bash
# Copyright 2021 STC-Innovation LTD (Author: Anton Mitrofanov) 

set -e

. ./cmd.sh
. ./path.sh
[ -f fairseq_ltlm/path.sh ] && . ./fairseq_ltlm/path.sh

cmd=utils/run.pl
stage=0
nj=48


utts_per_split=50000
extra_texts=
fam_train_dirs=

fam_train_dir=   
dir_with_extra_train_dirs=


. ./utils/parse_options.sh

. ./utils/require_argument_all.sh \
		--extra_texts \
		--fam_train_dir \
		--dir_with_extra_train_dirs
	

extra_texts=${extra_texts[*]}

if [ $stage -le 0 ] ; then
	echo "$0: Stage 0: Prepare text files "
	if [ ! -f $dir_with_extra_train_dirs/.done ] ; then
		python $(dirname $0)/scripts/balance_text.py --utts_per_split $utts_per_split --add_utt_ids --out_dir $dir_with_extra_train_dirs ${extra_texts[*]}
		touch $dir_with_extra_train_dirs/.done
	fi
fi

fam_train_utts=($fam_train_utts)
if [ $stage -le 1 ] ; then
	if [ ! -f $fam_train_dir/.done ] ; then
		echo "$0: Stage 1: Making subsets from train data"
		utils/subset_data_dir.sh data/train_960_cleaned_hires $fam_train_utts $fam_train_dir 
		touch $fam_train_dir/.done
	fi
fi

if [ $stage -le 2 ] && [ ! -z $online_ivector_dir ]	; then
	all_ivectors=$online_ivector_dir/ivectors_train_960_cleaned_hires
	ivec_dir=$online_ivector_dir/ivectors_$(basename $fam_train_dir)

	if [ ! -f $ivec_dir/.done ] ; then
		[ ! -d $ivec_dir ] && mkdir -p $ivec_dir
		cp $all_ivectors/ivector_period $ivec_dir
		utils/filter_scp.pl $fam_train_dir/utt2spk $all_ivectors/ivector_online.scp > $ivec_dir/ivector_online.scp
		touch $ivec_dir/.done
	fi
fi
