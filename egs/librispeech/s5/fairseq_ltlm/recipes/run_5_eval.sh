#!/bin/bash
# Copyright 2021 STC-Innovation LTD (Author: Anton Mitrofanov) 

set -e

. ./path.sh

cmd_cpu='utils/slurm.pl --config conf/slurm_cpu.conf'

# Parameters
lmwt=                    # LM scale for prunning
transformer_weight=
rescoring_strategy=base  # [base, norm, only_forks, global_prob]
egs_basename=lt_egs_$training_type
# Parameters from Data config
filter=

# Parameters from AM config
graph=

# Parameters from Experiment config
lt_model_dir=

cpt=best

use_gpu=true

. ./utils/parse_options.sh

. ./utils/require_argument_all.sh \
		--lmwt \
		--filter \
		--graph \
		--exp_dir \
		--lt_model_dir

if $use_gpu ; then
	cmd="$cmd_gpu --num-threads 12 --gpu 1"
	device=cuda
else
	cmd="$cmd_cpu --num-threads 12"
	device=cpu
fi

[ -z $transformer_weight ] && transformer_weight=$lmwt # usually gives good results


test_decoded=$exp_dir/test.decoded

if [ ! -f $test_decoded ] ; then
	echo "$0: Error: $test_decoded is missing. This file should be generated in stage run_1_decode.sh"
	exit 1
fi

cat $test_decoded | while read -a data_X_lats ; do
	dir=${data_X_lats[0]}
	lats=${data_X_lats[1]}
	egs_dir=$lats/$egs_basename
	test=$(basename $lats)

	echo "Evaluating model. Logging in $lt_model_dir/eval_${cpt}/$rescoring_strategy/eval_${test}.log"
	$cmd $lt_model_dir/eval_${cpt}/$rescoring_strategy/eval_${test}.log python fairseq_ltlm/ltlm/eval.py \
			--device $device \
			--max_len 600 \
			--lmwt $lmwt \
			--tokenizer_fn $graph/words.txt \
			--model_weight $transformer_weight \
			--data "$egs_dir" \
			--hyp_filter $filter \
			--strategy $rescoring_strategy \
			$lt_model_dir/checkpoint_${cpt}.pt \
			$dir/text
done
wait
