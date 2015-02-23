This is a kaldi setup for 3rd CHiME challenge.
See http://spandh.dcs.shef.ac.uk/chime_challenge/ for more detailed information.

Quick instruction:
1) Download kaldi (version 4710)

svn co -r 4710 https://svn.code.sf.net/p/kaldi/code/trunk kaldi-trunk-r4710

2) specify kaldi in path.sh
e.g.,
export KALDI_ROOT=`pwd`/../../..

3) specify data path of CHiME3 corpus in run_init.sh
e.g.,
chime3_data=/local_data/archive/speech-db/original/public/CHiME3/

4) execute run.sh

5) if you have your own enhanced speech data for training and test data, you can evaluate the performance 
of GMM and DNN systems by
local/run_gmm.sh <enhancement method> <enhanced speech directory>
local/run_dnn.sh <enhancement method> <enhanced speech directory>

You don't have to execute local/run_init.sh twice.

6) You can find result at
enhan=<enhancement method>
GMM: exp/tri3b_tr05_sr_$enhan/best_wer_$enhan.result
DNN: exp/tri4a_dnn_tr05_sr_${enhan}_smbr_i1lats/best_wer_${enhan}.result
