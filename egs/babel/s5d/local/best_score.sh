#!/bin/bash                                                                        
# Copyright (c) 2015, Johns Hopkins University ( Yenda Trmal <jtrmal@gmail.com> )
# License: Apache 2.0

# Begin configuration section.  
# End configuration section
set -e -o pipefail 
set -o nounset                              # Treat unset variables as an error


grep Avg exp/tri5/decode_dev10h.pem/score*/*.sys | utils/best_wer.sh
grep Avg exp/sgmm5_mmi_b0.1/decode_fmllr_dev10h.pem_it*/*/*.sys | utils/best_wer.sh
grep Avg exp/tri6_nnet/decode_dev10h.pem/score*/*.sys | utils/best_wer.sh
grep Avg exp_bnf/sgmm7_mmi_b0.1/decode_fmllr_dev10h.pem_it*/*/*.sys | utils/best_wer.sh
grep Avg exp_bnf/tri7_nnet/decode_dev10h.pem/*/*.sys  | utils/best_wer.sh



(
find exp/sgmm5_mmi_b0.1/decode_fmllr_dev10h.pem_it*/kws* -name "metrics.txt" | xargs grep MTWV | sort -k3,3gr | head -n 1
find exp/tri6_nnet/decode_dev10h.pem/kws* -name "metrics.txt" | xargs grep MTWV | sort -k3,3gr | head -n 1                 
find exp_bnf/sgmm7_mmi_b0.1/decode_fmllr_dev10h.pem_it*/kws* -name "metrics.txt" | xargs grep MTWV | sort -k3,3gr | head -n 1
find exp_bnf/tri7_nnet/decode_dev10h.pem*/kws* -name "metrics.txt" | xargs grep MTWV | sort -k3,3gr | head -n 1
) | column -t

for elem in  `cat data/dev10h.pem/extra_kws_tasks` ; do
  (
  find exp/sgmm5_mmi_b0.1/decode_fmllr_dev10h.pem_it*/${elem}_kws* -name "metrics.txt" | xargs grep MTWV | sort -k3,3gr | head -n 1
  find exp/tri6_nnet/decode_dev10h.pem/${elem}_kws* -name "metrics.txt" | xargs grep MTWV | sort -k3,3gr | head -n 1                 
  find exp_bnf/sgmm7_mmi_b0.1/decode_fmllr_dev10h.pem_it*/${elem}_kws* -name "metrics.txt" | xargs grep MTWV | sort -k3,3gr | head -n 1
  find exp_bnf/tri7_nnet/decode_dev10h.pem*/${elem}_kws* -name "metrics.txt" | xargs grep MTWV | sort -k3,3gr | head -n 1
  ) | column -t
done

