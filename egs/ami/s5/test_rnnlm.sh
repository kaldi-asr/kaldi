. cmd.sh

minibatch_size=64
shuffle_buffer_size=5000
max_param_change=20

rnnlm-init --binary=false rnnlm.config 0.mdl
$cuda_cmd train-rnnlm-sp.log rnnlm-train --max-param-change=$max_param_change "rnnlm-copy --learning-rate=0.008 0.mdl -|" \
   "ark:nnet3-shuffle-egs --buffer-size=$shuffle_buffer_size ark:data/sdm1/rnn-0.01-0.001-1.03-128-200-20-/train.egs ark:- | nnet3-merge-egs --minibatch-size=$minibatch_size ark:- ark:- |" 1.mdl

exit

rnnlm-train --use-gpu=no "rnnlm-copy --learning-rate=0.01 0.mdl -|" \
   "ark:nnet3-shuffle-egs --buffer-size=$shuffle_buffer_size ark:data/sdm1/rnn-0.01-0.001-1.03-128-200-20-/train.egs ark:- | nnet3-merge-egs --minibatch-size=$minibatch_size ark:- ark:- |" 1.mdl
