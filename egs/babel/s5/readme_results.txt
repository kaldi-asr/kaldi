The results are by default to be found in <your-decode_directory>/decode_* where the individual <your-decode_directory>/decode_* directory correspond to the language model weight.

An easthetically pleasing table with the results can be obtained for example like this (YMMV, as well as your aesthetic feeling):
find exp/sgmm5_mmi_b0.1 -name "*.ctm.sys" -not -name "*char.ctm.sys" -ipath "*fmllr_eval.pem*" | xargs grep 'Sum/Avg' | sed 's/:* *| */ /g' | sed 's/  */ /g' |  sort  -n -k 9 | column -t

