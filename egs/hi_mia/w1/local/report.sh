#!/bin/bash

# Copyright 2020 Audio, Speech and Language Processing Group (ASLP@NPU), Northwestern Polytechnical University (Authors: Zhuoyuan Yao, Xiong Wang, Jingyong Hou, Lei Xie)
#           2020 AIShell-Foundation (Author: Bengu WU) 
#           2020 Beijing Shell Shell Tech. Co. Ltd. (Author: Hui BU) 
# Apache 2.0


# when we train the model by nnet3 or chain, we can use this script to show the log property or accuracy

steps/nnet3/report/generate_plots.py --is-chain false exp/nnet3/tdnn_test_kws  report
