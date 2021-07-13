#!/bin/bash
# Copyright 2021 STC-Innovation LTD (Author: Anton Mitrofanov)
set -e

echo "$0 $@"  # Print the command line for logging

cmd=local/slurm.pl
use_gpu=true
extra_left_context=0
extra_right_context=0
frames_per_chunk=50
stage=0
online_ivector_dir=
nj=4

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
    echo "Usage: $0 <data-dir> <lang-dir> <model-dir> <out-dir>"
    exit 1;
fi

data=$1
lang=$2
model=$3
dir=$4

#write_per_frame_acoustic_loglikes="ark:| gzip -c > $dir/loglikes.JOB.gz"

if [ $stage -le 1 ]; then
	echo "$0:Stage 1: make align"
# --write_per_frame_acoustic_loglikes="$write_per_frame_acoustic_loglikes" \
	if [ ! -f $dir/done.ali ] ; then
		steps/nnet3/align.sh --extra_left_context=$extra_left_context \
			--nj $nj \
			--online_ivector_dir=$online_ivector_dir \
			--extra_right_context=$extra_right_context \
			--frames_per_chunk=$frames_per_chunk \
			--cmd="$cmd" \
			--use_gpu=$use_gpu \
			--scale-opts '--transition-scale=1.0 --acoustic-scale=1.0 --self-loop-scale=1.0' \
			$data $lang $model $dir 
		touch $dir/done.ali
	fi
fi

if [ $stage -le 2 ]; then
	echo "$0:Stage 2: Compute AM outputs"
	fsf=1
	if [ -f $model/frame_subsampling_factor ] ; then 
		fsf=$(cat $model/frame_subsampling_factor)
		echo "Using frame_subsampling_factor $fsf"
	fi
	if [ ! -f $dir/done.output ] ; then
	steps/nnet3/compute_output.sh --extra_left_context=$extra_left_context \
		--nj $nj \
		--online_ivector_dir=$online_ivector_dir \
		--extra_right_context=$extra_right_context \
		--frames_per_chunk=$frames_per_chunk \
		--frame_subsampling_factor=$fsf \
		--cmd="$cmd" \
		--use_gpu=$use_gpu \
		$data $model $dir 
	touch $dir/done.output
	fi
fi

if [ $stage -le 3 ]; then
	echo "$0:Stage 3: Convert ali to pdf"
	nj=$(cat $dir/num_jobs)
	utils/run.pl JOB=1:$nj $dir/log/ali2pdf.JOB.log \
			ali-to-pdf $model/final.mdl ark:"gunzip -c $dir/ali.JOB.gz|" ark:"| gzip -c > $dir/ali_pdf.JOB.gz"
fi
