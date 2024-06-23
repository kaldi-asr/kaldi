#!/bin/bash
# Copyright 2021 STC-Innovation LTD (Author: Anton Mitrofanov)


########### Results #############

######### eval best. < 170 epoch ####################
#####################################################
### dev_clean # dev_other # test_clean # test_other #
###   2.49    #   7.37    #   3.01     #   7.67     #
#####################################################
###
### INFO | ltlm.tasks.rescoring_task | Epoch 205. Valid wer is %WER 2.46 [ 1340 / 54402, 186 ins, 109 del, 1045 sub ]
######### eval best. 205 epoch ####################
#####################################################
### dev_clean # dev_other # test_clean # test_other #
###   2.46    #  7.41     #   3.01     #    7.66    #
#####################################################


set -e
. ./path.sh
. ./fairseq_ltlm/path.sh

stage=0

# Exp options
cmd_gpu=
exp_dir=

model_dir=
graph=

#Scales
lmwt=12
transformer_weight=12
filter=

# LTLM training params
epoch=6 # REAL all data epoch
lr=0.0001 # learning rate
wd=1e-6 # weight decay
c=1.25  # gradient clipping

valid_data_dir=data/dev_clean_hires
btz=22
uf=8
split_per_epoch=8

. ./utils/parse_options.sh


# out dir
base_exp_dir=$exp_dir/lt_small/
ltlm_dir=$base_exp_dir/lr${lr}_wd${wd}_cl${c}_btz${btz}_uf${uf}_spe${split_per_epoch}

# Data
train_decoded_lats=$(cut -d' ' -f2 $exp_dir/train.decoded)
train_generated_lats=$(cut -d' ' -f2 $exp_dir/train.generated)
train_egs=$(for lat in $train_generated_lats ; do echo $lat/lt_egs_$training_type; done)
num_egs=$(echo $train_egs | wc -w )
#echo $num_egs

# generated data epoch + real decode data epoch
part_epochs=$(($epoch * (num_egs + $split_per_epoch - 1) / $split_per_epoch + $epoch))

#echo "$part_epochs"
#exit



valid_egs=$model_dir/decode_$(basename $valid_data_dir)_graph_tgsmall_tglarge/lt_egs_$training_type

###


if [ $stage -le 0 ] ; then 
	echo "Generate data_config.json"
	python fairseq_ltlm/ltlm/pyscripts/get_data_config.py \
			--train $train_decoded_lats/lt_egs_$training_type,'',0 ${train_egs}  \
			--split_per_epoch ${split_per_epoch} \
			--valid $valid_egs,$valid_data_dir/text \
			--out $ltlm_dir/data_config.json
fi

if [ $stage -le 1 ] ; then
	echo "Training model. "
	sbatch  --output $ltlm_dir/full_log.out --error $ltlm_dir/full_log.err --nodes=2 \
	fairseq_ltlm/recipes/tuning/sbatch_ddp_train.sh \
				--user-dir fairseq_ltlm/ltlm/ \
				--criterion bce_loss \
				--task rescoring_task \
				--arch lt_small \
				--tokenizer_fn $graph/words.txt \
				--data_json $ltlm_dir/data_config.json \
				--lmwt $lmwt \
				--model_weight $transformer_weight \
				--hyp_filter $filter \
				--max_len 600 \
				--max-sentences $btz \
				--optimizer adam \
				--lr $lr \
				--curriculum 2 \
				--weight-decay $wd \
				--clip-norm $c \
				--max-epoch $part_epochs \
				--num-workers 1 \
				--no-epoch-checkpoints \
				--warmup-updates 4000 \
				--log-interval 100 \
				--fp16 \
				--dropout 0.2 \
				--attention-dropout 0.05 \
				--update-freq $uf \
				--save-dir $ltlm_dir \
				#--validate-interval 10 \
				#--grad-checkpointing \
fi

echo "Done"

