#!/bin/bash
# Copyright 2012  Navdeep Jaitly

# This program allows you to export log filterbank data from 
# KALDI to HTK format. Also exported is the force alignment 
# data, from the gmm alignment.
# HTK files are created, one per input file. 
# alignment file: ali is create one for the entire set (test/dev/train).
# Can be used for offline neural network training if you don't use
# the abilities of Kaldi to do so.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.



config=conf/mfcc.conf
data=data
#out_path=/ais/gobi2/ndjaitly/Data/Kaldi/Spectrograms/
#out_path=/ais/gobi2/ndjaitly/Data/Kaldi/FBANKS/
out_path=/ais/gobi2/ndjaitly/Data/Kaldi/export/FBANKS_25_10/
num_mel_bins=40
power_spectrum_only=0
frame_length=25
frame_shift=10
 
#for test in train test dev ; do 
for test in test dev ; do 
   scp=$data/$test/wav.scp
   out_dir=$out_path/$test/
   out_scp=$out_path/$test/htk.scp
   out_ali=$out_path/$test/ali
   mkdir -p $out_dir
   cat $scp | awk -v outdir=$out_dir '{ printf $1 " " outdir $1 ".htk\n"; }'  > $out_scp
   compute-fbank-feats --frame-length=$frame_length  --frame-shift=$frame_shift \
                 --num-mel-bins=$num_mel_bins --output-format=htk --verbose=2 \
                 --config=$config scp:$scp  scp:$out_scp
   ali-to-pdf exp/mono/final.mdl ark:exp/mono_ali_$test/ali t,ark:- > $out_ali
done
