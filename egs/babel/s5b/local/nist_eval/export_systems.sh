#!/usr/bin/env bash
set -e
set -o pipefail

. ./cmd.sh; . ./path.sh; 


#(
#bash filter_data.sh  --cmd "$decode_cmd"  data/shadow.uem eval.uem exp/sgmm5_mmi_b0.1/decode_*shadow.uem_it* 
#bash filter_data.sh  --cmd "$decode_cmd"  data/shadow.uem eval.uem exp_bnf/sgmm7_mmi_b0.1/decode_*shadow.uem_it*
#) &
#bash filter_data.sh  --cmd "$decode_cmd"  data/shadow.uem eval.uem exp/tri6*_nnet*/decode_shadow.uem*
#wait

(
bash filter_data.sh  --cmd "$decode_cmd"  data/shadow.uem dev10h.uem exp_bnf/sgmm7_mmi_b0.1/decode_*shadow.uem_it*
#bash filter_data.sh  --cmd "$decode_cmd"  data/shadow.uem dev10h.uem exp/sgmm5_mmi_b0.1/decode_*shadow.uem_it* 
) &
bash filter_data.sh  --cmd "$decode_cmd"  data/shadow.uem dev10h.uem exp/tri6*_nnet*/decode_shadow.uem 
wait

wait
exit

bash make_release.sh --dryrun false --dir exp/sgmm5_mmi_b0.1  --data data/shadow.uem --master dev10h.uem lang.conf ./release
bash make_release.sh --dryrun false --dir exp/tri6b_nnet  --data data/shadow.uem --master dev10h.uem lang.conf ./release
bash make_release.sh --dryrun false --dir exp_bnf/sgmm7_mmi_b0.1  --data data/shadow.uem --master dev10h.uem lang.conf ./release

bash make_release.sh --dryrun false --dir exp/sgmm5_mmi_b0.1 --extrasys "NEWJHU"  --data data/dev10h.uem --master dev10h.uem lang.conf ./release
bash make_release.sh --dryrun false --dir exp/tri6b_nnet --extrasys "NEWJHU"  --data data/dev10h.uem --master dev10h.uem lang.conf ./release
bash make_release.sh --dryrun false --dir exp_bnf/sgmm7_mmi_b0.1 --extrasys "NEWJHU"  --data data/dev10h.uem --master dev10h.uem lang.conf ./release


