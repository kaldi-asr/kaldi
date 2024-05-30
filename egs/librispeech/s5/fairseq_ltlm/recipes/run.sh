#!/bin/bash
# Copyright 2021 STC-Innovation LTD (Author: Anton Mitrofanov) 

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
	$scrips_basedir/run_1_decode_train.sh  --config $scrips_basedir/config.sh &
	$scrips_basedir/run_1_generate.sh  --config $scrips_basedir/config.sh
fi

wait

if [ $stage -le 2 ] ; then 
	echo "$0: Stage 2: rescoring tests with 4gram lm"
	$scrips_basedir/

if [ $stage -le 3 ] ; then
	echo "$0: Stage 3: Prepare egs"
	$scrips_basedir/run_3_prepare_egs.sh  --config $scrips_basedir/config.sh
fi

if [ $stage -le 4 ] ; then 
	echo "$0: Stage 4: Training LT-LM"
	$scrips_basedir/tuning/train_small_ltlm.lr1e-4.ddp2.sh --config $scrips_basedir/config.sh
fi

if [ $stage -le 5 ] ; then 
	echo "$0: Stage 5: Evaluating model"
	$scrips_basedir/run_5_eval.sh --config $scrips_basedir/config.sh --lt_model_dir exp/ltlm/lr0.0001_wd1e-6_cl1.25_btz22_uf8_spe8
fi
