This is a kaldi setup for 3rd CHiME challenge.
See http://spandh.dcs.shef.ac.uk/chime_challenge/ for more detailed information.

If you use these data in a publication, please cite:

Jon Barker, Ricard Marxer, Emmanuel Vincent, and Shinji Watanabe, The
third 'CHiME' Speech Separation and Recognition Challenge: Dataset,
task and baselines, submitted to IEEE 2015 Automatic Speech Recognition
and Understanding Workshop (ASRU), 2015.

Quick instruction:
1) Download CHiME3 data

Check the download page of http://spandh.dcs.shef.ac.uk/chime_challenge/

2) move to Kaldi CHiME3 directory, e.g.,

cd kaldi-trunk/egs/chime3/s5

3) specify CHiME3 root directory in run.sh e.g.,

chime3_data=<your CHiME3 directory>/CHiME3

4) execute run.sh

./run.sh

4*) we suggest to use the following command to save the main log file

nohup ./run.sh > run.log

5) if you have your own enhanced speech data for training and test data, you can evaluate the performance of GMM and DNN systems by

local/run_gmm.sh <enhancement method> <enhanced speech directory>
local/run_dnn.sh <enhancement method> <enhanced speech directory>
local/run_lmrescore.sh <your CHiME3 directory> <enhancement method>

You can put <enhanced speech directory> in your working directory.
But please make sure to use the same directory structure and naming convention with those of the
example enhanced speech directory in CHiME3/data/audio/16kHz/enhanced

You don't have to execute local/run_init.sh twice.

6) You can find result at

enhan=<enhancement method>
GMM clean training: exp/tri3b_tr05_orig_clean/best_wer_$enhan.result
GMM multi training: exp/tri3b_tr05_multi_$enhan/best_wer_$enhan.result
DNN multi training: exp/tri4a_dnn_tr05_multi_${enhan}_smbr_i1lats/best_wer_${enhan}.result
DNN multi training with LM rescoring: exp/tri4a_dnn_tr05_multi_${enhan}_smbr_i1lats_lmrescore/best_wer_${enhan}_rnnlm_5k_h300_w0.5_n100.result

Note that training on clean data means original WSJ0 data only (no booth data)
