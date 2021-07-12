#!/bin/bash


# Recipe for training lattice-transformer on Libripeech data 


scrips_basedir=$(dirname $0)

stage=0
set -e

. ./utils/parse_options.sh


if [ $stage -le 0 ] ; then
	echo "$0: Stage 0: Prepare data"
	$scrips_basedir/run_0_prepare.sh --config $scrips_basedir/config.sh
fi


if [ $stage -le 1 ] ; then
	echo "$0: Stage 1: Decode train set and generate lattices from text in parallel."
	./run_1_decode_train.sh  --config $scrips_basedir/config.sh &
	./run_1_generate.sh  --config $scrips_basedir/config.sh
fi

wait

if [ $stage -le 2 ] ; then
	echo "$0: Stage 2: Prepare egs"
	./run_2_prepare_egs.sh  --config $scrips_basedir/config.sh
fi

if [ $stage -le 3 ] ; then 
	echo "$0: Stage 3: Training LT-LM"
	$scrips_basedir/tuning/train_small_ltlm_PAPER.sh --config $scrips_basedir/config.sh
fi

if [ $stage -le 4 ] ; then 
	echo "$0: Stage 4: Evaluating model"
	$scrips_basedir/run_4_eval.sh --config $scrips_basedir/config.sh --lt_model_dir exp_librispeech/fam_dev_clean_other/lt_small
fi
