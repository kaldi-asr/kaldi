#!/bin/bash

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

. ./utils/parse_options.sh

. ./utils/require_argument_all.sh \
		--lmwt \
		--transformer_weight \
		--filter \
		--graph \
		--exp_dir \
		--lt_model_dir

cmd="$cmd_cpu --num-threads 12"

transformer_weight=$lmwt # usually gives good results

cpt=best

test_decoded=$exp_dir/test.decoded

if [ ! -f $test_decoded ] ; then
	echo "$0: Error: $test_decoded is missing. This file should be generated in stage run_1_decode.sh"
	exit 1
fi

cat $test_decoded | while read -a data_X_lats ; do
	dir=${data_X_lats[0]}
	test=$(basename $dir)
	lats=${data_X_lats[1]}
	egs_dir=$lats/$egs_basename

	echo "Evaluating model. Logging in $lt_model_dir/eval_${cpt}/$rescoring_strategy/eval_${test}.log"
	$cmd $lt_model_dir/eval_${cpt}/$rescoring_strategy/eval_${test}.log python ../../lattice_transformer/eval.py \
			--device cpu \
			--max_len 600 \
			--lmwt $lmwt \
			--tokenizer_fn $graph/words.txt \
			--model_weight $transformer_weight \
			--data "$egs_dir" \
			--hyp_filter $filter \
			--strategy $rescoring_strategy \
			--keep_tmp true \
			$lt_model_dir/checkpoint_${cpt}.pt \
			$dir/text_filtered 
done
wait
