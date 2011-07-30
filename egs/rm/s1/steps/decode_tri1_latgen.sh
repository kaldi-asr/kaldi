
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


# To view the lattices, a suitable command (after running this) is:
# gunzip -c exp/decode_tri1_latgen/test_feb89.lat.gz | scripts/int2sym.pl --field 3 data/words.txt | less

if [ -f path.sh ]; then . path.sh; fi
dir=exp/decode_tri1_latgen
tree=exp/tri1/tree
model=exp/tri1/final.mdl
graphdir=exp/graph_tri1

mkdir -p $dir

scripts/mkgraph.sh $tree $model $graphdir

for test in mar87 oct87 feb89 oct89 feb91 sep92; do
 (
  feats="ark:add-deltas --print-args=false scp:data/test_${test}.scp ark:- |"

  gmm-latgen-simple --beam=20.0 --acoustic-scale=0.08333 --word-symbol-table=data/words.txt $model $graphdir/HCLG.fst "$feats" "ark,t:|gzip -c > $dir/test_${test}.lat.gz" ark,t:$dir/test_${test}.tra ark,t:$dir/test_${test}.ali  2> $dir/decode_${test}.log

 # the ,p option lets it score partial output without dying..
  scripts/sym2int.pl --ignore-first-field data/words.txt data_prep/test_${test}_trans.txt | \
  compute-wer --mode=present ark:-  ark,p:$dir/test_${test}.tra >& $dir/wer_${test}

 # Now rescore lattices with various acoustic scales, and compute the WER.
 for inv_acwt in  6 7 8 9 10 11 12 13; do
   acwt=`perl -e "print (1.0/$inv_acwt);"`
   lattice-best-path --acoustic-scale=$acwt --word-symbol-table=data/words.txt \
      "ark:gunzip -c $dir/test_${test}.lat.gz|" ark:$dir/test_${test}.acwt${inv_acwt}.tra \
      2>$dir/rescore_${inv_acwt}.log
   scripts/sym2int.pl --ignore-first-field data/words.txt data_prep/test_${test}_trans.txt | \
   compute-wer --mode=present ark:-  ark,p:$dir/test_${test}.acwt${inv_acwt}.tra \
     >& $dir/wer_${test}_${inv_acwt}
 done

 ) &
done

wait



for inv_acwt in "" _6 _7 _8 _9 _10 _11 _12 _13; do
 grep WER $dir/wer_{mar87,oct87,feb89,oct89,feb91,sep92}${inv_acwt} | \
  awk '{n=n+$4; d=d+$6} END{ printf("Average WER is %f (%d / %d) \n", 100.0*n/d, n, d); }' \
   > $dir/wer${inv_acwt}
done
