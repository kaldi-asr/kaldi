#!/bin/bash

# this is a script to train the nnet3 blstm acoustic model
# it is based on blstm used in fisher_swbd recipe

stage=7 # assuming you already ran the TDNN system ; local/nnet3/run_tdnn.sh
affix=bidirectional
train_stage=-10
egs_stage=0
common_egs_dir=
remove_egs=true

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

local/nnet3/run_lstm.sh  --stage $stage --train-stage $train_stage --egs-stage "$egs_stage" \
                         --affix $affix --lstm-delay " [-1,1] [-2,2] [-3,3] " --label-delay 0 \
                         --cell-dim 1024 --recurrent-projection-dim 128 --non-recurrent-projection-dim 128 \
                         --chunk-left-context 40 --chunk-right-context 40 \
                         --extra-left-context 50 --extra-right-context 50 \
                         --common-egs-dir "$common_egs_dir" --remove-egs "$remove_egs"


#ASpIRE decodes
  local/nnet3/prep_test_aspire.sh --stage 1 --decode-num-jobs 30 --affix "v7" \
   --extra-left-context 40 --extra-right-context 40 --frames-per-chunk 20 \
   --sub-speaker-frames 6000 --window 10 --overlap 5 --max-count 75 --pass2-decode-opts "--min-active 1000" \
   --ivector-scale 0.75  dev_aspire data/lang exp/tri5a/graph_pp exp/nnet3/lstm_bidirectional

exit 0;

# final result
# %WER 29.4 | 2120 27210 | 77.0 14.7 8.3 6.4 29.4 77.9 | -1.227 | exp/nnet3/lstm_bidirectional/decode_dev_aspire_whole_uniformsegmented_win10_over5_v7_iterfinal_pp_fg/score_13/penalty_0.25/ctm.filt.filt.sys

#%WER 35.4 | 2120 27216 | 71.2 19.2 9.6 6.6 35.4 80.8 | -0.956 | exp/nnet3/lstm_bidirectional/decode_dev_aspire_whole_uniformsegmented_win10_over5_v7_iter200_pp_fg/score_14/penalty_0.0/ctm.filt.filt.sys
#%WER 33.6 | 2120 27215 | 72.7 18.1 9.2 6.3 33.6 79.1 | -1.018 | exp/nnet3/lstm_bidirectional/decode_dev_aspire_whole_uniformsegmented_win10_over5_v7_iter300_pp_fg/score_12/penalty_0.0/ctm.filt.filt.sys
#%WER 33.0 | 2120 27215 | 73.3 17.4 9.3 6.3 33.0 80.6 | -1.127 | exp/nnet3/lstm_bidirectional/decode_dev_aspire_whole_uniformsegmented_win10_over5_v7_iter400_pp_fg/score_12/penalty_0.25/ctm.filt.filt.sys
#%WER 31.6 | 2120 27216 | 74.5 16.4 9.1 6.1 31.6 79.4 | -1.119 | exp/nnet3/lstm_bidirectional/decode_dev_aspire_whole_uniformsegmented_win10_over5_v7_iter700_pp_fg/score_13/penalty_0.25/ctm.filt.filt.sys
#%WER 31.8 | 2120 27220 | 74.9 16.3 8.8 6.7 31.8 80.1 | -1.233 | exp/nnet3/lstm_bidirectional/decode_dev_aspire_whole_uniformsegmented_win10_over5_v7_iter800_pp_fg/score_12/penalty_0.25/ctm.filt.filt.sys
#%WER 31.6 | 2120 27222 | 75.0 16.1 8.8 6.7 31.6 80.7 | -1.208 | exp/nnet3/lstm_bidirectional/decode_dev_aspire_whole_uniformsegmented_win10_over5_v7_iter900_pp_fg/score_12/penalty_0.5/ctm.filt.filt.sys
#%WER 30.0 | 2120 27212 | 76.0 15.4 8.6 6.1 30.0 79.4 | -1.193 | exp/nnet3/lstm_bidirectional/decode_dev_aspire_whole_uniformsegmented_win10_over5_v7_iter1700_pp_fg/score_13/penalty_0.25/ctm.filt.filt.sys
#%WER 30.2 | 2120 27211 | 76.4 15.3 8.3 6.7 30.2 79.4 | -1.099 | exp/nnet3/lstm_bidirectional/decode_dev_aspire_whole_uniformsegmented_win10_over5_v7_iter1800_pp_fg/score_14/penalty_0.0/ctm.filt.filt.sys
#%WER 30.3 | 2120 27215 | 76.8 15.5 7.8 7.1 30.3 78.8 | -1.317 | exp/nnet3/lstm_bidirectional/decode_dev_aspire_whole_uniformsegmented_win10_over5_v7_iter1900_pp_fg/score_11/penalty_0.25/ctm.filt.filt.sys
#%WER 29.8 | 2120 27215 | 76.8 15.0 8.2 6.6 29.8 78.6 | -1.219 | exp/nnet3/lstm_bidirectional/decode_dev_aspire_whole_uniformsegmented_win10_over5_v7_iter1996_pp_fg/score_13/penalty_0.25/ctm.filt.filt.sys
#%WER 30.1 | 2120 27213 | 76.3 14.7 9.0 6.4 30.1 79.2 | -1.204 | exp/nnet3/lstm_bidirectional/decode_dev_aspire_whole_uniformsegmented_win10_over5_v7_iter2124_pp_fg/score_14/penalty_0.5/ctm.filt.filt.sys
