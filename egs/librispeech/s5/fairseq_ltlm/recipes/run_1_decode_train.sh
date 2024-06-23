#!/bin/bash
# Copyright 2021 STC-Innovation LTD (Author: Anton Mitrofanov) 

set -e

. ./path.sh


# Parameters from default config
cmd_cpu='local/slurm.pl --max-jobs-run 100 --config conf/slurm_cpu.conf'
cmd_gpu='local/slurm.pl --max-jobs-run 12'
decode_nj=30
stage=0


# Parameters:

# Parameters from Data config
acoustic_train_dirs=
test_dirs=
filter=


# Parameters from AM config
extra_left_context=0
extra_right_context=0
frames_per_chunk=150
online_ivector_dir= 	# not required
model_dir=
graph=

# Parameters for Decode 
acwt=1.0
post_decode_acwt=10.0
decode_num_threads=12
decode_use_gpu=true
decode_train_nj=1200

# Parameters frin Experiment config
exp_dir=

. ./utils/parse_options.sh

#checking parameters
. ./utils/require_argument_all.sh \
		--test_dirs \
		--model_dir \
		--graph \
		--exp_dir \
		--filter \
		--acoustic_train_dirs

# Configurate cmd
if $decode_use_gpu ; then
	decode_cmd=$cmd_gpu
else
	decode_cmd=$cmd_cpu
fi

# Converting to list
acoustic_train_dirs=(${acoustic_train_dirs[*]})
test_dirs=(${test_dirs[*]})

# Output lists with 2 columns <data-dir> <lats-dir>
# In stage 2 egs will be generated for this data
out_test_decoded=$exp_dir/test.decoded
out_train_decoded=$exp_dir/train.decoded


if [ $stage -le 0 ] ; then
	echo "$0: Stage 0: Decode test sets"
	for dir in ${test_dirs[*]} ; do 
		decode_dir=${model_dir}/decode_$(basename $dir)_$(basename $graph)
		ivec_dir=''
		if [ ! -z $online_ivector_dir ] ; then
			ivec_dir=$online_ivector_dir/ivectors_$(basename $dir)
		fi
		if [ ! -f $decode_dir/.done ] ; then
			steps/nnet3/decode.sh \
					--nj $decode_nj \
					--cmd "$decode_cmd" \
					--use-gpu $decode_use_gpu \
					--num_threads $decode_num_threads	\
					--acwt $acwt \
					--post_decode_acwt $post_decode_acwt \
					--frames_per_chunk $frames_per_chunk \
					--extra_left_context $extra_left_context \
					--extra_right_context $extra_right_context \
					--skip_scoring false \
					--online_ivector_dir "$ivec_dir" \
					$graph $dir $decode_dir
			echo "$dir $decode_dir" >> $out_test_decoded
			touch $decode_dir/.done
		fi
	done
fi


if [ $stage -le 1 ] ; then
	echo "$0: Stage 1: Decode train sets"
	for dir in ${acoustic_train_dirs[*]} ; do
		decode_dir=${model_dir}/decode_$(basename $dir)_$(basename $graph)
		if [ ! -z $online_ivector_dir ] ; then
			ivec_dir=$online_ivector_dir/ivectors_$(basename $dir)
		fi

		if [ ! -f $decode_dir/.done ] ; then
			steps/nnet3/decode.sh \
					--nj $decode_train_nj \
					--cmd "$decode_cmd" \
					--use-gpu $decode_use_gpu \
					--num_threads $decode_num_threads \
					--acwt $acwt \
					--post_decode_acwt $post_decode_acwt \
					--frames_per_chunk $frames_per_chunk \
					--extra_left_context $extra_left_context \
					--extra_right_context $extra_right_context \
					--skip_scoring true \
					--online_ivector_dir "$ivec_dir" \
					$graph $dir $decode_dir
			echo "$dir $decode_dir" >> $out_train_decoded
			touch $decode_dir/.done
		fi
	done
fi

