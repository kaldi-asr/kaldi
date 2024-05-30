This recipe builds a Speech Activity Detection (SAD)system.
The rats_sad corpus is used to train and evaluate the SAD models.
The rats_sad corpus is available at the LDC (LDC2015S02).

Although scripts to do SAD are available in other Kaldi recipes like aspire and chime6, they are embedded in recipes  that are focused on larger tasks.
there is no recipe dedicated to SAD. 
The rats_sad corpus was created specifically for the SAD task.
Documentation on the rats_sad corpus is available at:
https://catalog.ldc.upenn.edu/docs/LDC2015S02/README.txt

Note:
The recipe follows the conventions set by other recipes in the Kaldi repository available at:
https://github.com/kaldi_asr/kaldi.git
Next, the files and directories are described.
- s5
The whole recipe is contained in this directory.
- s5/cmd.sh
This file sets variables used for job management.
- s5/conf/mfcc_hires.conf
This file contains parameters used in extracting Mel FrequencyCepstral Coeficients from input audio data.
- s5/run.sh
Invoking this script builds and evaluates the entire system.
The location of the rats_sad corpus must be set in the rats_sad_data_dir variable.
The variables in cmd.sh might also need to be adjusted in order to run this script.
The run.sh script invokes several scripts located under 3 directories:
1. s5/steps
This directory is symlinked to ../../wsj/s5/steps
It contains standard Kaldi scripts that have been optimized over many years.
2. s5/utils
This directory is symlinked to:
../../wsj/s5/utils
It also contains scripts that have been optimized over many years.
3. local
This directory contains scripts that were taken from other Kaldi recipes and modified to work for the rats_sad corpus.
- s5/local/convert_rttm_to_utt2spk_and_segments.py
- s5/local/prepare_data.py
- s5/local/rats_sad_tab_prep.sh
These 3 scripts prepare rats_sad data that is given in text format.
- s5/local/get_speech_targets.py
- s5/local/get_transform_probs_mat.py
- s5/local/make_mfcc.sh
These 3 scripts do data preparation on audio signal data.
- s5/local/detect_speech_activity.sh
this script performs SAD with the trained models.
- s5/local/segmentation/tuning/tdnn_lstm_sad_1a.sh
This file contains parameter settings and the command used to train a Long Short-Term Memory Neural Network model for SAD.
the afix 1a denotes that this file represents an experiment that uses the parameter settings given in the file.
This file is symlinked to:
- s5/local/segmentation/run_lstm.sh

According to the pattern set in other Kaldi recipes a new experiment with different parameter settings would be written in a file named something lie:
s5/local/segmentation/tuning/tdnn_lstm_sad_1b.sh
and symlinked to
s5/local/segmentation/run_lstm.sh
- path.sh
This file sets path variables to make Kaldi accessible to the other scripts.
