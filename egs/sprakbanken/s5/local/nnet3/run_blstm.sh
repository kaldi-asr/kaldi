stage=0
train_stage=-10
affix=bidirectional
nnet3_affix=
common_egs_dir=
remove_egs=true
train_set=train
gmm=tri3b


# BLSTM params
cell_dim=1024
rp_dim=128
nrp_dim=128
chunk_left_context=40
chunk_right_context=40

# training options
srand=0
num_jobs_initial=3
num_jobs_final=15
samples_per_iter=20000
num_epochs=6
echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

local/nnet3/run_lstm.sh --affix $affix \
                         --srand $srand \
                         --stage $stage \
                         --train-stage $train_stage \
                         --train-set $train_set \
                         --gmm $gmm \
                         --lstm-delay " [-1,1] [-2,2] [-3,3] " \
                         --label-delay 0 \
                         --cell-dim $cell_dim \
                         --recurrent-projection-dim $rp_dim \
                         --non-recurrent-projection-dim $nrp_dim \
                         --common-egs-dir "$common_egs_dir" \
                         --chunk-left-context $chunk_left_context \
                         --chunk-right-context $chunk_right_context \
                         --num-jobs-initial $num_jobs_initial \
                         --num-jobs-final $num_jobs_final \
                         --samples-per-iter $samples_per_iter \
                         --num-epochs $num_epochs \
                         --remove-egs $remove_egs

