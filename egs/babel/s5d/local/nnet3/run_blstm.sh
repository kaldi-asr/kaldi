 
stage=0
train_stage=-10

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

local/nnet3/run_lstm.sh --affix bidirectional \
                         --stage $stage \
                         --train-stage $train_stage \
                         --lstm-delay " [-1,1] [-2,2] [-3,3] " \
                         --label-delay 0 \
                         --cell-dim 1024 \
                         --recurrent-projection-dim 128 \
                         --non-recurrent-projection-dim 128 \
                         --chunk-left-context 40 \
                         --chunk-right-context 40
