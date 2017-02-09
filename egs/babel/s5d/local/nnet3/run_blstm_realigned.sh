stage=0
train_stage=-10
cell_dim=512
rp_dim=128
nrp_dim=128
affix=bidirectional
multicondition=false
common_egs_dir=
num_epochs=8
align_model_dir=exp/nnet3/tdnn_sp
extra_align_opts=

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

local/nnet3/run_lstm_realigned.sh --affix $affix \
                         --stage $stage \
                         --train-stage $train_stage \
                         --num-epochs $num_epochs \
                         --lstm-delay " [-1,1] [-2,2] [-3,3] " \
                         --label-delay 0 \
                         --cell-dim $cell_dim \
                         --recurrent-projection-dim $rp_dim \
                         --non-recurrent-projection-dim $nrp_dim \
                         --common-egs-dir "$common_egs_dir" \
                         --multicondition  $multicondition \
                         --chunk-left-context 40 \
                         --chunk-right-context 40 \
                         --extra-align-opts "$extra_align_opts" \
                         --align-model-dir "$align_model_dir"
