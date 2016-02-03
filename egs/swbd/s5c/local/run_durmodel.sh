#!/bin/bash

[ -f ./path.sh ] && . ./path.sh; # source the path.
. cmd.sh
. parse_options.sh || exit 1;

srcdir=exp/tri4
durmodel_dir=$srcdir/durmodel

steps/dur_model/train.sh --cmd "$decode_cmd" \
                         --max-duration 60 \
                         --left-context 5 --right-context 2 \
                         data/lang/phones $srcdir $durmodel_dir || exit 1;


steps/dur_model/rescore.sh --cmd "$decode_cmd" \
                           --duration-model-scale 0.7 \
                           $durmodel_dir/final_nnet_dur_model.mdl \
                           $srcdir/decode_eval2000_sw1_tg \
                           $srcdir/decode_eval2000_sw1_tg_durmodel_rescored || exit 1;

if [ -d $srcdir/decode_eval2000_sw1_fsh_fg ]; then
  steps/dur_model/rescore.sh --cmd "$decode_cmd" \
                             --duration-model-scale 0.7 \
                             $durmodel_dir/final_nnet_dur_model.mdl \
                             $srcdir/decode_eval2000_sw1_fsh_fg \
                             $srcdir/decode_eval2000_sw1_fsh_fg_durmodel_rescored &
fi

wait
