#!/bin/bash -u

# Copyright 2012  Arnab Ghoshal

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

exit 1;
# This script shows the steps needed to build a phone recognizer for TIMIT.

# This recipe follows the setup first described in:
#   K. F. Lee and H. W. Hon, "Speaker-independent phone recognition using hidden Markov models," 1988 
# where the training set is mapped to 48 phones and the results are presented 
# on a 39-phone subset of that.

# Set WORKDIR to someplace with enough disk space. That is where MFCCs will 
# get created, as well as the LM in ARPA & FST formats.
WORKDIR=/path/with/disk/space
mkdir -p $WORKDIR
cp -r conf local utils steps path.sh $WORKDIR
cd $WORKDIR
. path.sh
[ -z "$KALDIROOT" ] && echo "ERROR: Must specify the KALDIROOT env varaible" && exit 1;

local/timit_data_prep.sh --config-dir=$PWD/conf --corpus-dir=/path/to/TIMIT --work-dir=$WORKDIR

local/timit_format_data.sh --hmm-proto=conf/topo.proto --work-dir=$PWD

# Now make MFCC features.
mfccdir=$WORKDIR/data/MFCC
for x in train dev test; do
  steps/make_mfcc.sh --num-jobs 6 data/$x exp/make_mfcc/$x $mfccdir
done

decode_cmd="qsub -q all.q@@blade -l ram_free=500M,mem_free=500M"
train_cmd="qsub -q all.q@@blade -l ram_free=200M,mem_free=200M"

steps/train_mono.sh --num-jobs 10 --qcmd "$train_cmd" \
  data/train data/lang exp/mono
utils/mkgraph.sh --mono data/lang_test_phone_bg exp/mono exp/mono/graph_bg
steps/decode_deltas.sh --accwt 1.0 --beam 20.0 --latgen --num-jobs 6 \
  --qcmd "$decode_cmd" exp/mono/graph_bg data/dev exp/mono/decode_dev_bg
utils/score_lats.sh exp/mono/decode_dev_bg exp/mono/graph_bg/words.txt \
  data/dev conf/phones.60-48-39.map 
opt_accwt=`grep WER exp/mono/decode_dev_bg/wer_* \
  | sed -e 's?.*wer_??' -e 's?:%WER??' -e 's?\[.*??' | sort -k2,2 -g \
  | head -1 | awk '{print 1/$1}'`
steps/decode_deltas.sh --accwt $opt_accwt --beam 20.0 --num-jobs 4 \
  --qcmd "$decode_cmd" exp/mono/graph_bg data/test exp/mono/decode_test_bg
utils/score_text.sh exp/mono/decode_test_bg exp/mono/graph_bg/words.txt \
  data/test conf/phones.60-48-39.map 

steps/align_deltas.sh --num-jobs 10 --qcmd "$train_cmd" \
  data/train data/lang exp/mono exp/mono_ali

steps/train_deltas.sh --num-jobs 10 --qcmd "$train_cmd" \
  2000 10000 data/train data/lang exp/mono_ali exp/tri1

utils/mkgraph.sh data/lang_test_phone_bg exp/tri1 exp/tri1/graph_bg
steps/decode_deltas.sh --accwt 1.0 --beam 20.0 --latgen --num-jobs 6 \
  --qcmd "$decode_cmd" exp/tri1/graph_bg data/dev exp/tri1/decode_dev_bg
utils/score_lats.sh exp/tri1/decode_dev_bg exp/tri1/graph_bg/words.txt \
  data/dev conf/phones.60-48-39.map 
opt_accwt=`grep WER exp/tri1/decode_dev_bg/wer_* \
  | sed -e 's?.*wer_??' -e 's?:%WER??' -e 's?\[.*??' | sort -k2,2 -g \
  | head -1 | awk '{print 1/$1}'`
steps/decode_deltas.sh --accwt $opt_accwt --beam 20.0 --num-jobs 4 \
  --qcmd "$decode_cmd" exp/tri1/graph_bg data/test exp/tri1/decode_test_bg
utils/score_text.sh exp/tri1/decode_test_bg exp/tri1/graph_bg/words.txt \
  data/test conf/phones.60-48-39.map 

