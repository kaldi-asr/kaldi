
# Copyright 2010-2011 Microsoft Corporation

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

# Decode the testing data.

# this is the hardest test set, see
# http://www.itl.nist.gov/iad/mig/tests/rt/ASRhistory/pdf/resource_management_92eval.pdf

dir=exp/decode_tri_mixup
mkdir -p $dir
srcdir=exp/tri_mixup
model=$srcdir/25.mdl
graphdir=exp/graph_tri_mixup


../src/bin/faster-decode-gmm --acoustic-scale=0.08333 --word-symbol-table=data/words.txt  $model $graphdir/HCLG.fst data/test_sep92.scp  $dir/word_transcripts.txt $dir/alignments.txt > $dir/decode.out

../src/bin/compute-wer --symbol-table=data/words.txt  data_prep/test_sep92_trans.txt $dir/word_transcripts.txt  > $dir/wer

