#!/bin/bash
# Copyright 2021 STC-Innovation LTD (Author: Anton Mitrofanov) 

set -e

. ./path.sh


# Parameters from default config
cmd_cpu='local/slurm.pl --max-jobs-run 100 --config conf/slurm_cpu.conf'
cmd_gpu='local/slurm.pl --max-jobs-run 12'

# Parameters:

# Parameters from Data config
test_dirs=

# Parameters from AM config
model_dir=
graph=

exp_dir=

. ./utils/parse_options.sh

#checking parameters
. ./utils/require_argument_all.sh \
		--test_dirs \
		--model_dir \
		--graph \
		--exp_dir

out_test_decoded=$exp_dir/test.decoded

# Configurate cmd
if $decode_use_gpu ; then
	decode_cmd=$cmd_gpu
else
	decode_cmd=$cmd_cpu
fi

# Converting to list
test_dirs=(${test_dirs[*]})

echo "$0: Stage 0: Rescore test sets"
for dir in ${test_dirs[*]} ; do 
	decode_dir=${model_dir}/decode_$(basename $dir)_$(basename $graph)
	decode_set=$(basename $dir)
	mark=$model_dir/decode_${decode_set}_graph_tgsmall_tgmed/.done
	if [ ! -f $mark ] ; then
		steps/lmrescore.sh \
			--cmd "$cmd_cpu" \
			--self-loop-scale 1.0 \
			data/lang_test_{tgsmall,tgmed} \
			$dir \
			$model_dir/decode_${decode_set}_graph_tgsmall{,_tgmed} || exit 1
		touch $mark
	fi
	mark=$model_dir/decode_${decode_set}_graph_tgsmall_tglarge/.done
	if [ ! -f $mark ] ; then
		steps/lmrescore_const_arpa.sh \
			--cmd "$cmd_cpu" \
			data/lang_test_{tgsmall,tglarge} \
			$dir \
			$model_dir/decode_${decode_set}_graph_tgsmall{,_tglarge} || exit 1
		echo "$dir $model_dir/decode_${decode_set}_graph_tglarge" >> $out_test_decoded
		touch $mark
	fi
done


