This s5b recipe for MALACH data is a modified version of the s5b
recipe for ami. 

You need to download the malach data to get started. For information about the MALACH database see : 
USC-SFI MALACH Interviews and Transcripts English - Speech Recognition Edition
https://catalog.ldc. upenn.edu/LDC2019S11

Once the data is unloaded and untar-ed, you need to run:

run_prepare_shared.sh - prepares most of the data for the system
run.sh - builds the system

Beforehand, you need to edit BOTH scripts to point to 
where you downloaded and untar-ed the data. Find the lines in
run_prepare_shared.sh and run.sh that say:

malach_dir=dummy_directory

Replace "dummy_directory" with the fully-qualified location of the actual data
data. For example, let's say you copied the data distribution tar file to 
/user/jdoe/malach and untar-ed it there. That would create a high level directory called 
/user/jdoe/malach/malach_eng_speech/recognition. You would then change the above line to read:

malach_dir=/user/doe/malach/malach_eng_speech_recognition/data

Note that the scripts were "tweaked" to always use sclite scoring
(vs. default kaldi scoring).

Other issues that we have run up against in setting up this recipe
that may or may not impact you:

On the system on which these scripts were developed, we run python 2.7
and a relatively older version of CUDA by default. We had to modify
path.sh to point to the right load libraries for both python 3 (a
number of the scripts use python three) and an appropriate library
consistent with the level of CUDA we were using. Please modify path.sh
accordingly.

You may also have to modify "configure" line 405 in
/speech7/picheny5_nb/forked_kaldi/kaldi/src to point to where your
version of CUDA lives. 

Basic pipeline results summary:

tri2:

ascore_7 /dev.ctm.filt.sys: | Sum/Avg  |  843   12345 | 64.8   29.4    5.8    9.3   44.6   81.6 | -0.765 |
ascore_8 /dev.ctm.filt.sys: | Sum/Avg  |  843   12345 | 65.7   28.2    6.2    8.5   42.8   79.6 | -0.589 |
ascore_9 /dev.ctm.filt.sys: | Sum/Avg  |  843   12345 | 66.2   27.3    6.5    7.7   41.5   77.9 | -0.451 |
ascore_10/dev.ctm.filt.sys: | Sum/Avg  |  843   12345 | 66.6   26.4    7.0    7.0   40.4   76.7 | -0.351 |
ascore_11/dev.ctm.filt.sys: | Sum/Avg  |  843   12345 | 66.9   25.7    7.4    6.6   39.7   76.0 | -0.293 |
ascore_12/dev.ctm.filt.sys: | Sum/Avg  |  843   12345 | 66.8   25.4    7.8    6.2   39.3   75.0 | -0.257 |
ascore_13/dev.ctm.filt.sys: | Sum/Avg  |  843   12345 | 66.5   25.1    8.3    5.7   39.1   74.0 | -0.230 |
ascore_14/dev.ctm.filt.sys: | Sum/Avg  |  843   12345 | 66.0   25.0    9.0    5.5   39.4   74.3 | -0.231 |
ascore_15/dev.ctm.filt.sys: | Sum/Avg  |  843   12345 | 65.6   24.8    9.6    5.2   39.6   74.0 | -0.218 |


tri3.si:
ascore_7 /dev.ctm.filt.sys: | Sum/Avg  |  843   12345 | 63.0   30.5    6.6    8.5   45.6   79.8 | -1.738 |
ascore_8 /dev.ctm.filt.sys: | Sum/Avg  |  843   12345 | 63.3   29.6    7.1    7.9   44.6   78.8 | -1.505 |
ascore_9 /dev.ctm.filt.sys: | Sum/Avg  |  843   12345 | 63.4   29.2    7.4    7.4   44.0   77.6 | -1.336 |
ascore_10/dev.ctm.filt.sys: | Sum/Avg  |  843   12345 | 63.5   28.7    7.7    7.0   43.4   77.1 | -1.212 |
ascore_11/dev.ctm.filt.sys: | Sum/Avg  |  843   12345 | 63.5   28.3    8.2    6.6   43.1   77.1 | -1.136 |
ascore_12/dev.ctm.filt.sys: | Sum/Avg  |  843   12345 | 63.4   28.0    8.5    6.3   42.8   76.9 | -1.079 |
ascore_13/dev.ctm.filt.sys: | Sum/Avg  |  843   12345 | 63.1   27.8    9.1    6.0   43.0   76.7 | -1.046 |
ascore_14/dev.ctm.filt.sys: | Sum/Avg  |  843   12345 | 62.6   27.9    9.4    5.9   43.2   77.2 | -1.039 |
ascore_15/dev.ctm.filt.sys: | Sum/Avg  |  843   12345 | 62.3   28.0    9.7    5.6   43.4   77.1 | -1.045 |

tri3:
ascore_7 /dev.ctm.filt.sys: | Sum/Avg  |  843   12345 | 70.0   25.2    4.8    8.4   38.3   75.3 | -1.022 |
ascore_8 /dev.ctm.filt.sys: | Sum/Avg  |  843   12345 | 70.5   24.3    5.2    7.8   37.3   74.0 | -0.819 |
ascore_9 /dev.ctm.filt.sys: | Sum/Avg  |  843   12345 | 70.9   23.8    5.4    7.2   36.3   73.4 | -0.684 |
ascore_10/dev.ctm.filt.sys: | Sum/Avg  |  843   12345 | 71.2   23.2    5.6    6.7   35.5   72.5 | -0.581 |
ascore_11/dev.ctm.filt.sys: | Sum/Avg  |  843   12345 | 71.2   22.9    5.8    6.4   35.1   71.6 | -0.513 |
ascore_12/dev.ctm.filt.sys: | Sum/Avg  |  843   12345 | 71.2   22.7    6.2    6.0   34.9   70.9 | -0.438 |
ascore_13/dev.ctm.filt.sys: | Sum/Avg  |  843   12345 | 71.0   22.4    6.6    5.7   34.7   70.7 | -0.410 |
ascore_14/dev.ctm.filt.sys: | Sum/Avg  |  843   12345 | 70.8   22.3    6.9    5.5   34.6   70.1 | -0.392 |
ascore_15/dev.ctm.filt.sys: | Sum/Avg  |  843   12345 | 70.7   22.1    7.1    5.2   34.5   69.2 | -0.398 |

tri3_cleaned.si:
ascore_7 /dev.ctm.filt.sys: | Sum/Avg  |  843   12345 | 63.1   30.4    6.5    8.8   45.7   82.1 | -1.805 |
ascore_8 /dev.ctm.filt.sys: | Sum/Avg  |  843   12345 | 63.6   29.6    6.8    8.3   44.7   81.4 | -1.527 |
ascore_9 /dev.ctm.filt.sys: | Sum/Avg  |  843   12345 | 63.6   29.1    7.3    7.8   44.2   81.1 | -1.346 |
ascore_10/dev.ctm.filt.sys: | Sum/Avg  |  843   12345 | 63.8   28.7    7.6    7.4   43.7   80.2 | -1.243 |
ascore_11/dev.ctm.filt.sys: | Sum/Avg  |  843   12345 | 63.8   28.4    7.8    7.0   43.2   79.6 | -1.149 |
ascore_12/dev.ctm.filt.sys: | Sum/Avg  |  843   12345 | 63.6   28.2    8.2    6.7   43.1   79.0 | -1.095 |
ascore_13/dev.ctm.filt.sys: | Sum/Avg  |  843   12345 | 63.2   28.2    8.7    6.4   43.2   79.2 | -1.064 |
ascore_14/dev.ctm.filt.sys: | Sum/Avg  |  843   12345 | 62.9   28.1    9.0    6.1   43.2   79.0 | -1.038 |
ascore_15/dev.ctm.filt.sys: | Sum/Avg  |  843   12345 | 62.6   28.0    9.5    5.8   43.3   78.3 | -1.041 |

tri3_cleaned:
ascore_7 /dev.ctm.filt.sys: | Sum/Avg  |  843   12345 | 69.3   25.9    4.8    8.7   39.4   77.1 | -1.004 |
ascore_8 /dev.ctm.filt.sys: | Sum/Avg  |  843   12345 | 70.0   25.0    5.0    8.0   37.9   75.7 | -0.815 |
ascore_9 /dev.ctm.filt.sys: | Sum/Avg  |  843   12345 | 70.7   24.1    5.2    7.3   36.6   74.5 | -0.672 |
ascore_10/dev.ctm.filt.sys: | Sum/Avg  |  843   12345 | 71.0   23.6    5.4    6.9   35.9   72.7 | -0.576 |
ascore_11/dev.ctm.filt.sys: | Sum/Avg  |  843   12345 | 70.9   23.3    5.8    6.6   35.7   72.2 | -0.511 |
ascore_12/dev.ctm.filt.sys: | Sum/Avg  |  843   12345 | 70.8   23.0    6.2    6.3   35.5   72.2 | -0.467 |
ascore_13/dev.ctm.filt.sys: | Sum/Avg  |  843   12345 | 70.7   22.8    6.5    6.1   35.3   71.6 | -0.432 |
ascore_14/dev.ctm.filt.sys: | Sum/Avg  |  843   12345 | 70.6   22.6    6.8    5.8   35.3   71.3 | -0.415 |
ascore_15/dev.ctm.filt.sys: | Sum/Avg  |  843   12345 | 70.4   22.4    7.2    5.6   35.2   70.7 | -0.407 |

chain:
ascore_5/ dev_hires.ctm.filt.sys: | Sum/Avg  |  843   12345 | 78.4   18.3    3.3    6.3   27.9   65.6 | -0.425 |
ascore_6/ dev_hires.ctm.filt.sys: | Sum/Avg  |  843   12345 | 79.4   17.1    3.5    5.8   26.4   63.9 | -0.239 |
ascore_7/ dev_hires.ctm.filt.sys: | Sum/Avg  |  843   12345 | 80.1   16.2    3.7    5.2   25.1   63.7 | -0.104 |
ascore_8/ dev_hires.ctm.filt.sys: | Sum/Avg  |  843   12345 | 80.5   15.5    4.0    4.9   24.4   62.5 | -0.018 |
ascore_9/ dev_hires.ctm.filt.sys: | Sum/Avg  |  843   12345 | 80.6   15.1    4.3    4.5   24.0   62.0 |  0.030 |
ascore_10/dev_hires.ctm.filt.sys: | Sum/Avg  |  843   12345 | 80.4   15.0    4.7    4.1   23.7   61.9 |  0.063 |
ascore_11/dev_hires.ctm.filt.sys: | Sum/Avg  |  843   12345 | 79.9   14.9    5.3    3.8   24.0   61.9 |  0.098 |
ascore_12/dev_hires.ctm.filt.sys: | Sum/Avg  |  843   12345 | 79.4   14.7    5.9    3.6   24.2   61.9 |  0.098 |
ascore_13/dev_hires.ctm.filt.sys: | Sum/Avg  |  843   12345 | 79.0   14.6    6.4    3.4   24.4   61.3 |  0.092 |
ascore_14/dev_hires.ctm.filt.sys: | Sum/Avg  |  843   12345 | 78.2   14.6    7.2    3.2   25.0   62.9 |  0.095 |
ascore_15/dev_hires.ctm.filt.sys: | Sum/Avg  |  843   12345 | 77.6   14.7    7.8    3.0   25.4   63.7 |  0.092 |


In addition to the basic Kaldi recipe above, one can also rescore with an rnnlm. There are three different variations 
for training the language model:

local/rnnlm/tuning/run_lstm_tdnn_1a.sh - rnnlm
local/rnnlm/tuning/run_lstm_tdnn_1b.sh - rnnlm plus L2 regularization
local/rnnlm/tuning/run_lstm_tdnn_bs_1a.sh - rnnlm plus backstitch training

rescoring is done like this (for the 1a rnnlm version; similarly for 1b, and bs_1a):

rnnlm/lmrescore_pruned.sh data/lang_malach.o4g.kn.pr1-9/ exp/rnnlm_lstm_tdnn_1a data/dev exp/chain_cleaned/tdnn1i_sp_bi/decode_dev exp/chain_cleaned/tdnn1i_sp_bi/decode_dev_rnnlm_lstm_tdnn_1a

Results:

rnnlm_lstm_tdnn_1a:
ascore_7/ dev.ctm.filt.sys:| Sum/Avg  |  843   12345  | 81.7   14.5    3.8    4.1   22.4   56.8 | -0.408 |
ascore_8/ dev.ctm.filt.sys:| Sum/Avg  |  843   12345  | 82.2   13.8    4.1    3.7   21.5   56.0 | -0.328 |
ascore_9/ dev.ctm.filt.sys:| Sum/Avg  |  843   12345  | 82.2   13.3    4.5    3.4   21.1   55.5 | -0.274 |
ascore_10/dev.ctm.filt.sys:| Sum/Avg  |  843   12345  | 81.9   13.1    5.0    3.1   21.2   55.3 | -0.241 |
ascore_11/dev.ctm.filt.sys:| Sum/Avg  |  843   12345  | 81.4   13.1    5.5    2.9   21.5   55.3 | -0.237 |
ascore_12/dev.ctm.filt.sys:| Sum/Avg  |  843   12345  | 80.9   13.0    6.1    2.7   21.8   55.3 | -0.227 |
ascore_13/dev.ctm.filt.sys:| Sum/Avg  |  843   12345  | 80.3   13.1    6.6    2.6   22.3   55.6 | -0.240 |
ascore_14/dev.ctm.filt.sys:| Sum/Avg  |  843   12345  | 79.6   13.2    7.2    2.5   22.9   56.5 | -0.263 |
ascore_15/dev.ctm.filt.sys:| Sum/Avg  |  843   12345  | 78.9   13.3    7.7    2.4   23.5   57.7 | -0.274 |

rnnlm_lstm_tdnn_1b:
ascore_7 /dev.ctm.filt.sys:| Sum/Avg  |  843   12345  | 81.8   14.5    3.7    4.3   22.5   56.8 | -0.199 |
ascore_8 /dev.ctm.filt.sys:| Sum/Avg  |  843   12345  | 82.3   13.8    3.9    3.9   21.7   56.2 | -0.102 |
ascore_9 /dev.ctm.filt.sys:| Sum/Avg  |  843   12345  | 82.6   13.3    4.1    3.6   21.1   55.5 | -0.065 |
ascore_10/dev.ctm.filt.sys:| Sum/Avg  |  843   12345  | 82.5   13.1    4.5    3.3   20.9   55.5 | -0.033 |
ascore_11/dev.ctm.filt.sys:| Sum/Avg  |  843   12345  | 82.2   12.8    5.0    3.1   20.8   55.4 | -0.034 |
ascore_12/dev.ctm.filt.sys:| Sum/Avg  |  843   12345  | 81.7   12.7    5.6    2.9   21.2   56.1 | -0.022 |
ascore_13/dev.ctm.filt.sys:| Sum/Avg  |  843   12345  | 80.9   12.8    6.3    2.7   21.8   56.3 | -0.015 |
ascore_14/dev.ctm.filt.sys:| Sum/Avg  |  843   12345  | 80.4   12.9    6.7    2.6   22.2   57.3 | -0.020 |
ascore_15/dev.ctm.filt.sys:| Sum/Avg  |  843   12345  | 79.6   13.1    7.2    2.5   22.9   57.9 | -0.017 |

rnnlm_lstm_tdnn_bs_1a:
ascore_7/ dev.ctm.filt.sys:| Sum/Avg  |  843   12345  | 82.0   14.3    3.7    4.3   22.3   56.9 | -0.212 |
ascore_8/ dev.ctm.filt.sys:| Sum/Avg  |  843   12345  | 82.5   13.6    3.9    3.8   21.4   56.3 | -0.130 |
ascore_9/ dev.ctm.filt.sys:| Sum/Avg  |  843   12345  | 82.6   13.1    4.3    3.5   20.9   55.9 | -0.084 |
ascore_10/dev.ctm.filt.sys:| Sum/Avg  |  843   12345  | 82.5   12.8    4.7    3.2   20.7   55.9 | -0.052 |
ascore_11/dev.ctm.filt.sys:| Sum/Avg  |  843   12345  | 82.1   12.8    5.1    3.1   20.9   55.3 | -0.037 |
ascore_12/dev.ctm.filt.sys:| Sum/Avg  |  843   12345  | 81.6   12.6    5.8    2.8   21.3   55.6 | -0.036 |
ascore_13/dev.ctm.filt.sys:| Sum/Avg  |  843   12345  | 81.0   12.7    6.3    2.7   21.7   55.6 | -0.028 |
ascore_14/dev.ctm.filt.sys:| Sum/Avg  |  843   12345  | 80.5   12.7    6.9    2.5   22.1   56.7 | -0.039 |
ascore_15/dev.ctm.filt.sys:| Sum/Avg  |  843   12345  | 79.8   12.8    7.4    2.4   22.6   56.6 | -0.050 |



