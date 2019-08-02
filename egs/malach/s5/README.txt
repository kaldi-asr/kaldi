# Copyright 2019  IBM Corp. (Author: Michael Picheny) Adapted AMI recipe to MALACH corpus

This s5 recipe for MALACH data is a modified version of the s5b
recipe for AMI. 

You need to download the malach data to get started. For information about the MALACH database see : 
USC-SFI MALACH Interviews and Transcripts English - Speech Recognition Edition
https://catalog.ldc.upenn.edu/LDC2019S11

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
/user/jdoe/malach/malach_eng_speech_recognition. You would then change the above line to read:

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
%WER 39.1 | 843 12345 | 66.5 25.1 8.3 5.7 39.1 74.0 | -0.230 | exp/tri2/decode_dev_malach.o4g.kn.pr1-9/ascore_13/dev.ctm.filt.sys

tri3.si:
%WER 42.8 | 843 12345 | 63.4 28.0 8.5 6.3 42.8 76.9 | -1.079 | exp/tri3/decode_dev_malach.o4g.kn.pr1-9.si/ascore_12/dev.ctm.filt.sys

tri3:
%WER 34.5 | 843 12345 | 70.7 22.1 7.1 5.2 34.5 69.2 | -0.398 | exp/tri3/decode_dev_malach.o4g.kn.pr1-9/ascore_15/dev.ctm.filt.sys

tri3_cleaned.si:
%WER 43.1 | 843 12345 | 63.6 28.2 8.2 6.7 43.1 79.0 | -1.095 | exp/tri3_cleaned/decode_dev_malach.o4g.kn.pr1-9.si/ascore_12/dev.ctm.filt.sys

tri3_cleaned:
%WER 35.1 | 843 12345 | 71.0 22.6 6.4 6.1 35.1 72.7 | -0.431 | exp/tri3_cleaned/decode_dev_malach.o4g.kn.pr1-9/ascore_13/dev.ctm.filt.sys

Results using the chain model, and rescoring the chain model with various LSTMs, can be found in s5/local/chain/run_tdnn.sh


