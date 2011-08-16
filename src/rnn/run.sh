#./rnnrescore --n=100 words.txt WSJ.35M.200cl.350h.kaldi.rnn ark,t:/mnt/matylda5/kombrink/src/kaldi/trunk/egs/wsj/s1/exp/decode_tri1kn_latgen_tgpr_eval92/1.lats ark,t:nbest.lats
#../../tools/openfst/bin/fstreverse

ln -s /mnt/matylda5/kombrink/src/kaldi/debug/src/rnnlm-rescore/Eigen/ .
ln -s /mnt/matylda5/kombrink/src/kaldi/trunk/src/rnnrescore_/1.lats .
ln -s /mnt/matylda5/kombrink/src/kaldi/trunk/src/rnnrescore_/words.txt .
ln -s /mnt/matylda5/kombrink/src/kaldi/trunk/src/rnnrescore_/WSJ.35M.200cl.350h.kaldi.rnn .
ln -s /mnt/matylda5/kombrink/src/kaldi/trunk/src/rnnrescore_/WSJ.35M.200cl.350h.rnn .

make clean
make

./rnnrescore --acoustic-scale=0.0625 --lm-scale=1 --n=1000 words.txt ark,t:1.lats WSJ.35M.200cl.350h.kaldi.rnn ark,t:nbest.lats

cat nbest.lats | awk 'BEGIN{while (getline<"words.txt")v[$2]=$1}{$4="";if ($3 in v)$3=v[$3];print $0}' | less
