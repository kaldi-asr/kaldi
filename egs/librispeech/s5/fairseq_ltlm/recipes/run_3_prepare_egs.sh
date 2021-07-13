#!/bin/bash
# Copyright 2021 STC-Innovation LTD (Author: Anton Mitrofanov) 

set -e

. ./path.sh
. ./fairseq_ltlm/path.sh


# Parameters from default config
cmd_cpu='slurm.pl --max-jobs-run 100 --config conf/slurm_cpu.conf'
stage=0


# Parameters:
lmwt=			# Lm scale for prunning
prune_beam=4	
training_type='oracle_path' # Can be oracle_path or choices_at_fork
egs_basename=lt_egs_$training_type
max_len=600

#epoch_max_MB=4000

# Parameters from Data config
lang=					
filter=

# Parameters from AM config
graph=

# Parameters from Experiment config
exp_dir=


. ./utils/parse_options.sh

#checking parameters
. ./utils/require_argument_all.sh \
		--lmwt \
		--prune_beam \
		--training_type \
		--egs_basename \
		--lang \
		--filter \
		--graph \
		--exp_dir 


# Checking what decode and generate files exists
# This files should be generated in run_1_*.
# Files containg pairs "data_dir lats_dir"
test_decoded=$exp_dir/test.decoded
train_decoded=$exp_dir/train.decoded
train_generated=$exp_dir/train.generated
out_test_egs=$exp_dir/test.egs

if [ ! -f $test_decoded ] ; then 
	echo "$0: Error: $test_decoded is missing. This file should be generated in stage run_1_decode.sh"
	exit 1
fi
if [ ! -f $train_decoded ] && [ ! -f $train_generated ] ; then
	echo "$0: Error: Both $train_decoded and $train_generated is missing. Check run_1_decode.sh and run_1_generate.sh"
	exit 1
fi

# Getting unk word
unk=$(cat $lang/oov.txt)


if [ $stage -le 0 ] ; then
	echo "$0: Stage 0: Get egs for tests"
	cat $test_decoded | while read -a data_X_lats ; do 
		dir=${data_X_lats[0]}
		lats=${data_X_lats[1]}
		egs_dir=$lats/$egs_basename
		if [ ! -f $egs_dir/.done ] ; then
			bash  $(dirname $0)/scripts/prepare_egs.sh \
					--cmd "$cmd_cpu" \
					--unk "$unk" \
					--data_dir $dir \
					--lats_dir $lats \
					--prune_lmscale $lmwt \
					--prune_beam $prune_beam \
					--training_type $training_type \
					--out_dir $egs_dir \
					--filter $filter \
					--max_len $max_len \
 					--skip_scoring false \
					--lang $lang 
			echo "$dir $egs_dir" >> $out_test_egs
			touch $egs_dir/.done
		fi
	done
fi

if [ $stage -le 1 ] ; then
	echo "$0: Stage 1: Get egs for train sets"
		cat $train_decoded $train_generated | while read -a data_X_lats ; do 
		
		dir=${data_X_lats[0]}
		 lats=${data_X_lats[1]}
		 egs_dir=$lats/$egs_basename
		 if [ ! -f $egs_dir/.done ] ; then
			echo "$dir"
			bash  $(dirname $0)/scripts/prepare_egs.sh \
					--cmd "$cmd_cpu --mem 32G" \
					--unk "$unk" \
					--data_dir $dir \
					--lats_dir $lats \
					--prune_lmscale $lmwt \
					--prune_beam $prune_beam \
					--max_len $max_len \
					--training_type $training_type \
					--out_dir $egs_dir \
					--filter $filter \
					--skip_scoring true \
					--lang $lang 
			touch $egs_dir/.done
		fi
	done
fi

#if [ $stage -le 2 ] ; then 
#	echo "$0: Stage 1: Get egs for generated train sets"
#	run.pl JOB=1:


#if [ $stage -le 2 ] ; then 
#	echo "$0: Generate $exp_dir/data_config.json"
#	train_d=$(cat $train_decoded | while read -a data_X_lats ; do echo -n "${data_X_lats[1]}/$egs_basename,$(basename ${data_X_lats[1]})" ; done)
#	train_g=$(cat $train_generated | while read -a data_X_lats ; do echo -n "${data_X_lats[1]}/$egs_basename,$(basename ${data_X_lats[1]}) " ; done)
#	valid=$(head -1 $test_decoded | while read -a data_X_lats ; do echo -n "${data_X_lats[1]}/$egs_basename,${data_X_lats[0]}/text_filtered " ; done )
#	test=$(cat $test_decoded | while read -a data_X_lats ; do echo -n "${data_X_lats[1]}/$egs_basename,${data_X_lats[0]}/text_filtered " ; done )
#	out=$exp_dir/balanced_data_config.json
#
#	python ../../lattice_transformer/pyscripts/get_balanced_data_config.py --epoch_max_MB $epoch_max_MB \
#			--train_decoded $train_d \
#			--train_generated $train_g \
#			--valid $valid \
#			--test $test \
#			--out $out
#fi
