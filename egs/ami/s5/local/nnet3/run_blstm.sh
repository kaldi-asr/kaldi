stage=0
train_stage=-10
mic=ihm
affix=bidirectional
common_egs_dir=
remove_egs=true
use_ihm_ali=false
use_sat_alignments=true

# BLSTM params
cell_dim=512
rp_dim=128
nrp_dim=128
chunk_left_context=40
chunk_right_context=40

# training options
num_jobs_initial=2
num_jobs_final=12
samples_per_iter=20000
realign_times=
echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

local/nnet3/run_lstm.sh --affix $affix \
                         --stage $stage \
                         --train-stage $train_stage \
                         --lstm-delay " [-1,1] [-2,2] [-3,3] " \
                         --label-delay 0 \
                         --cell-dim $cell_dim \
                         --recurrent-projection-dim $rp_dim \
                         --non-recurrent-projection-dim $nrp_dim \
                         --common-egs-dir "$common_egs_dir" \
                         --chunk-left-context $chunk_left_context \
                         --chunk-right-context $chunk_right_context \
                         --mic $mic \
                         --num-jobs-initial $num_jobs_initial \
                         --num-jobs-final $num_jobs_final \
                         --samples-per-iter $samples_per_iter \
                         --use-ihm-ali $use_ihm_ali \
                         --use-sat-alignments $use_sat_alignments \
                         --realign-times "$realign_times" \
                         --remove-egs $remove_egs

