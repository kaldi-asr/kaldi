#!/usr/bin/env bash


. conf/common_vars.sh
. ./lang.conf

set -e
set -o pipefail
set -u

# Wait till the main run.sh gets to the stage where's it's 
# finished aligning the tri5 model.
echo "Waiting till exp/tri5_ali/.done exists...."
while [ ! -f exp/tri5_ali/.done ]; do sleep 30; done
echo "...done waiting for exp/tri5_ali/.done"


if [ ! -f exp/tri6_nnet/.done ]; then
  steps/train_nnet_cpu.sh  \
    --mix-up "$dnn_mixup" \
    --initial-learning-rate "$dnn_initial_learning_rate" \
    --final-learning-rate "$dnn_final_learning_rate" \
    --num-hidden-layers "$dnn_num_hidden_layers" \
    --num-parameters "$dnn_num_parameters" \
    --num-jobs-nnet $dnn_num_jobs \
    --cmd "$train_cmd" \
    "${dnn_train_extra_opts[@]}" \
    data/train data/lang exp/tri5_ali exp/tri6_nnet  || exit 1;
  touch exp/tri6_nnet/.done
fi

#The following has been commented as the 5-anydecode.sh script takes care about
#decoding and keyword search in more complete/systematic fashion

#[ ! -f exp/tri5/graph/.done ] && utils/mkgraph.sh data/lang exp/tri5 exp/tri5/graph && touch exp/tri5/graph/.done

#-if [ ! -f exp/tri6_nnet/decode_dev2h/.done ]; then
#-  [ ! -f exp/tri5/graph/.done ] && utils/mkgraph.sh data/lang exp/tri5 exp/tri5/graph && touch exp/tri5/graph/.done
#-  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj $decode_nj \
#-  "${decode_extra_opts[@]}" --transform-dir exp/tri5/decode_dev2h \
#-    exp/tri5/graph data/dev2h exp/tri6_nnet/decode_dev2h 
#-  touch exp/tri6_nnet/decode_dev2h/.done
#-fi
#-if [ ! -f exp/tri6_nnet/decode_dev2h/.kws.done ]; then
#-  local/kws_search.sh --cmd "$decode_cmd" --duptime $duptime \
#-      data/lang data/dev2h exp/tri6_nnet/decode_dev2h
#-  touch exp/tri6_nnet/decode_dev2h/.kws.done 
#-fi  


exit 0
