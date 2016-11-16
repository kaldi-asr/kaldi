. cmd.sh

minibatch_size=64
shuffle_buffer_size=5000
max_param_change=20

eg_number=1
learning_rate=0.001
for i in `seq 1 40`; do
  echo iter $i ...
  old=$[$i-1]
  $cuda_cmd train-rnnlm-sp-$i.log rnnlm-train --max-param-change=$max_param_change "rnnlm-copy --learning-rate=$learning_rate $old.mdl -|" \
   "ark:nnet3-shuffle-egs --srand=$i --buffer-size=$shuffle_buffer_size ark:data/sdm1/rnn-0.01-0.001-1.03-64-200-4-/egs/train.$eg_number.egs ark:- | nnet3-merge-egs --minibatch-size=$minibatch_size ark:- ark:- |"\
    $i.mdl

 grep parse train-rnnlm-sp-$i.log | awk -F '-' '{print "PPL on train is: " exp($NF)}'

 eg_number=$[$eg_number+1]
 if [ $eg_number -eq 4 ]; then
   eg_number=1
 fi

 learning_rate=`echo $learning_rate | awk '{print $1/1.03}'`
done


rnnlm-init --binary=false rnnlm.config 0.mdl
$cuda_cmd train-rnnlm-sp.log rnnlm-train --max-param-change=$max_param_change "rnnlm-copy --learning-rate=0.008 0.mdl -|" \
   "ark:nnet3-shuffle-egs --buffer-size=$shuffle_buffer_size ark:data/sdm1/rnn-0.01-0.001-1.03-128-200-20-/train.egs ark:- | nnet3-merge-egs --minibatch-size=$minibatch_size ark:- ark:- |" 1.mdl

exit

rnnlm-train --use-gpu=no "rnnlm-copy --learning-rate=0.01 0.mdl -|" \
   "ark:nnet3-shuffle-egs --buffer-size=$shuffle_buffer_size ark:data/sdm1/rnn-0.01-0.001-1.03-128-200-20-/train.egs ark:- | nnet3-merge-egs --minibatch-size=$minibatch_size ark:- ark:- |" 1.mdl
