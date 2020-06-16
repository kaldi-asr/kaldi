#!/bin/bash

# Copyright 2020 Audio, Speech and Language Processing Group (ASLP@NPU), Northwestern Polytechnical University (Authors: Zhuoyuan Yao, Xiong Wang, Jingyong Hou, Lei Xie)
#           2020 AIShell-Foundation (Author: Bengu WU) 
#           2020 Beijing Shell Shell Tech. Co. Ltd. (Author: Hui BU) 
# Apache 2.0


# Get result file to draw a roc curve

dir=exp/chain/tdnn_1b_kws_sp/decode_test
best_penalty=0.0
best_lmwt=16
best_result_log=$dir/scoring_kaldi/penalty_${best_penalty}/log/best_path.${best_lmwt}.log

grep "lattice-best-path.cc:99" $best_result_log | awk '{print $5,$12}' > score.txt
sed 's/,//g' score.txt > score.txt.tmp
mv score.txt.tmp score.txt
awk '{if(NF==1)print $1,"0";if(NF==2)print $1,"1"}' $best_result_log > result.txt
sed 's/# 0//g' result.txt > result.txt.tmp
mv result.txt.tmp result.txt
local/process_result.py score.txt  result.txt > result
rm score.txt
rm result.txt

