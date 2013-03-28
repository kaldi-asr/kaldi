#!/bin/bash


. conf/common_vars.sh
. ./lang.conf
 
# Wait till the main run.sh gets to the stage where's it's 
# finished aligning the tri5 model.
echo "Watiting till exp/tri5_ali/.done exists...."
while [ ! -d exp/tri5_ali/.done ]; do sleep 30; done
echo "...done waiting for exp/tri5_ali/.done"


if [ ! -f exp/tri6_nnet/.done ]; then
  steps/train_nnet_cpu.sh  \
    --mix-up "$dnn_mixup" \
    --initial-learning-rate "$dnn_initial_learning_rate" \
    --final-learning-rate "$dnn_final_learning_rate" \
    --num-hidden-layers "$dnn_num_hidden_layers" \
    --num-parameters "$dnn_num_parameters" \
     $dnn_extra_opts \
    --num-jobs-nnet $dnn_num_jobs \
    --num-threads 8 --parallel-opts "-pe smp 7" \
    --cmd "queue.pl -l arch=*64,mem_free=4.0G,ram_free=0.75G" \
    data/train data/lang exp/tri5_ali exp/tri6_nnet  || exit 1;
  touch exp/tri6_nnet/.done
fi

if [ ! -f exp/tri6_nnet/decode_dev2h_dev2h/.done ]; then
  steps/decode_dev2h_nnet_cpu.sh --cmd "$decode_cmd" --nj $decode_nj \
    --transform-dir exp/tri5/decode_dev2h \
    exp/tri5/graph data/dev2h exp/tri6_nnet/decode_dev2h 
  touch exp/tri6_nnet/decode_dev2h/.done
fi

if [ ! -f exp/combine_2/decode_dev2h/.done ]; then
  for iter in 1 2 3 4; do
    local/score_combine.sh --cmd "queue.pl -l mem_free=2.0G,ram_free=1.0G" \
      data/dev2h data/lang exp/tri6_nnet/decode_dev2h exp/sgmm5_mmi_b0.1/decode_dev2h_fmllr_it$iter exp/combine_2/decode_dev2h_it$iter
    touch exp/combine_2/decode_dev2h/.done 
  done
fi

# This assumes the exp_BNF stuff is done..
if [ ! -f exp/combine_3/decode_dev2h/.done ]; then
  for iter in 1 2 3 4; do
    if [ ! -f exp_BNF/sgmm7_mmi_b0.1/decode_dev2h_fmllr_it$iter/.done ]; then
      echo "BNF decode in exp_BNF/sgmm7_mmi_b0.1/decode_dev2h_fmllr_it$iter is not done, skipping this step."
    fi
    local/score_combine.sh --cmd "queue.pl -l mem_free=2.0G,ram_free=1.0G" \
      data/dev2h data/lang exp_BNF/sgmm7_mmi_b0.1/decode_dev2h_fmllr_it$iter:10 \
      exp/sgmm5_mmi_b0.1/decode_dev2h_fmllr_it$iter exp/tri5_nnet/decode_dev2h exp/combine_3/decode_dev2h_it$iter
    touch exp_BNF/sgmm7_mmi_b0.1/decode_dev2h_fmllr_it$iter/.done
  done
fi

exit 0
