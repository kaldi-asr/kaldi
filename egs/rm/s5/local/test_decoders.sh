#!/usr/bin/env bash


dir=exp/tri1/decode/tmp
mkdir -p $dir
acwt=0.083333
beam=15.0
n=100 # number of utts to decode

. ./path.sh

gmm-latgen-faster --beam=$beam --lattice-beam=6.0 --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=exp/tri1/graph/words.txt exp/tri1/final.mdl exp/tri1/graph/HCLG.fst "ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:data/test/utt2spk scp:data/test/cmvn.scp 'scp:head -n $n data/test/feats.scp|' ark:- | add-deltas ark:- ark:- |" "ark:|lattice-1best --acoustic-scale=$acwt ark:- ark:- | gzip -c > $dir/lat.1.gz" 2>$dir/decode_latgen_faster.log &

gmm-decode-faster --beam=$beam --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=exp/tri1/graph/words.txt exp/tri1/final.mdl exp/tri1/graph/HCLG.fst "ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:data/test/utt2spk scp:data/test/cmvn.scp 'scp:head -n $n data/test/feats.scp|' ark:- | add-deltas ark:- ark:- |" ark:/dev/null ark:/dev/null "ark:|gzip -c > $dir/lat.2.gz" 2>$dir/decode_faster.log &

gmm-decode-simple --beam=$beam --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=exp/tri1/graph/words.txt exp/tri1/final.mdl exp/tri1/graph/HCLG.fst "ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:data/test/utt2spk scp:data/test/cmvn.scp 'scp:head -n $n data/test/feats.scp|' ark:- | add-deltas ark:- ark:- |" ark:/dev/null ark:/dev/null "ark:|gzip -c > $dir/lat.3.gz" 2>$dir/decode_simple.log &

wait


! lattice-equivalent --max-error-proportion=0.02 "ark:gunzip -c $dir/lat.1.gz|" "ark:gunzip -c $dir/lat.2.gz|" 2>$dir/equivalent_1_2.log && \
   echo "Decoders were not equivalent, check $dir/equivalent_1_2.log and contact maintainers" && exit 1;

! lattice-equivalent --max-error-proportion=0.02 "ark:gunzip -c $dir/lat.1.gz|" "ark:gunzip -c $dir/lat.3.gz|" 2>$dir/equivalent_2_3.log && \
   echo "Decoders were not equivalent, check $dir/equivalent_2_3.log and contact maintainers" && exit 1;

echo "$0: decoder comparison test succeeded"
exit 0;
