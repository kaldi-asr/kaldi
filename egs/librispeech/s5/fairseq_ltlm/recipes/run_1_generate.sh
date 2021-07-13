#!/bin/bash
# Copyright 2021 STC-Innovation LTD (Author: Anton Mitrofanov) 

set -e

. ./path.sh
[ -f fairseq_ltlm/path.sh ] && . ./fairseq_ltlm/path.sh


# Parameters from default config
cmd_cpu='slurm.pl --max-jobs-run 100 --config conf/slurm_cpu.conf'
decode_nj=30
stage=0


# Parameters:
fam_smoothing=0       	# if not zero then fam = avg(subset) + one_hot*smoothing
beam=13					# Decode beam for generating lattices. Recommended 11 < beam < 15
max_active=3000			# max active hpys for generating lattices.	
generate_nj=1200

# Parameters from Data config
fam_train_dir=
lang=					#Used for getting align
acoustic_train_dirs=
dir_with_extra_train_dirs=
test_dirs=
filter=


# Parameters from AM config
extra_left_context=0
extra_right_context=0
frames_per_chunk=150
online_ivector_dir= 	# not required
model_dir=
graph=

# Parameters frin Experiment config
exp_dir=


# egs params
prune_beam=4
training_type='oracle_path'
max_len=600
egs_basename=lt_egs_$training_type



. ./utils/parse_options.sh

#checking parameters
. ./utils/require_argument_all.sh \
		--fam_train_dir \
		--lang \
		--test_dirs \
		--model_dir \
		--graph \
		--exp_dir \
		--filter

# train dirs
. ./utils/require_argument_all.sh \
		--acoustic_train_dirs \
		--dir_with_extra_train_dirs


# Converting to list
acoustic_train_dirs=(${acoustic_train_dirs[*]})
extra_train_dirs=($dir_with_extra_train_dirs/*)
test_dirs=(${test_dirs[*]})


ali_dir=$exp_dir/align_$(basename $fam_train_dir)
fam_dir=$exp_dir/fam_s${fam_smoothing}
fam_model=$fam_dir/fam.ark
stretch_model=$fam_dir/sali.pkl


if [ $stage -le 0 ] ; then
	echo "$0: Stage 0: make ali saving loglikes----------"
	if [ ! -f $ali_dir/.done ] ; then
		if [ ! -z $online_ivector_dir ] ; then
			ivec=$online_ivector_dir/ivectors_$(basename $fam_train_dir)
		else
			ivec=''
		fi
		$(dirname $0)/genlats/ali_and_post.sh --extra_left_context=$extra_left_context \
				--online_ivector_dir="$ivec" \
				--extra_right_context=$extra_right_context \
				--frames_per_chunk=$frames_per_chunk \
				--cmd "$cmd_gpu" \
				--nj 8 \
				$fam_train_dir $lang $model_dir $ali_dir
		touch $ali_dir/.done
	fi
fi

num_pdf=$(tree-info $model_dir/tree |grep num-pdfs|awk '{print $2}')
if [ $stage -le 1 ] ; then
	echo "$0: Stage 1: Train FAM model--------------"
	if [ ! -f $fam_dir/.done ] ; then
			python $(dirname $0)/genlats/generate_fake_am_model.py \
				--label_smoothing $fam_smoothing \
				--stretch_model_path $stretch_model \
				--id2ll_model_path $fam_model --max_pdf $num_pdf $ali_dir
		touch $fam_dir/.done
	fi
fi


if [ $stage -le 2 ] ; then
		echo "$0: Stage 2: Generate Lattice for one test. Checking everything is ok"
		dir=${test_dirs[0]}
		decode_dir=$fam_dir/decode_fake_$(basename $dir)
		if [ ! -f $decode_dir/.done ] ; then
				bash $(dirname $0)/genlats/generate_fake_lats.sh \
						--skip-scoring false \
						--filter "$filter" \
						--beam $beam \
						--max_active $max_active \
						--cmd "$cmd_cpu" \
						--nj $decode_nj \
						--data $dir \
						--lang $lang \
						--tree_dir $model_dir \
						--fam_dir $fam_dir \
						--graph $graph \
						--dir $decode_dir 
				touch $decode_dir/.done
		fi
fi


unk=$(cat $lang/oov.txt)  

get_egs() {
		#dir=${data_X_lats[0]}
		#lats=${data_X_lats[1]}
		egs_dir=$decode_dir/$egs_basename
		if [ ! -f $egs_dir/.done ] ; then
				echo "Generating egs for $decode_dir $dir in background"
				bash  $(dirname $0)/scripts/prepare_egs.sh \
						--cmd "$cmd_cpu --max-jobs-run 40" \
						--unk "$unk" \
						--data_dir $dir \
						--lats_dir $decode_dir \
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
}

if [ $stage -le 3 ] ; then
	echo "$0: Stage 3: Generate Lattices for train --------------"
	for dir in ${acoustic_train_dirs[*]} ${extra_train_dirs[*]} ; do
		decode_dir=$fam_dir/decode_fake_$(basename $dir)
		if [ ! -f $decode_dir/.done ] ; then
				bash $(dirname $0)/genlats/generate_fake_lats.sh \
					--skip-scoring true \
					--beam $beam \
					--max_active $max_active \
					--cmd "$cmd_cpu" \
					--nj $generate_nj \
					--data $dir \
					--lang $lang \
					--tree_dir $model_dir \
					--fam_dir $fam_dir \
					--graph $graph \
					--dir $decode_dir 
			echo "$dir $decode_dir" >> $exp_dir/train.generated
			touch $decode_dir/.done
		fi
		get_egs &
	done
	wait
fi
echo "Done"
