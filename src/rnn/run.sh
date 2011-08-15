#./rnnrescore --n=100 words.txt WSJ.35M.200cl.350h.kaldi.rnn ark,t:/mnt/matylda5/kombrink/src/kaldi/trunk/egs/wsj/s1/exp/decode_tri1kn_latgen_tgpr_eval92/1.lats ark,t:nbest.lats
#../../tools/openfst/bin/fstreverse

ln -s /mnt/matylda5/kombrink/src/kaldi/debug/src/rnnlm-rescore/Eigen/ .
ln -s /mnt/matylda5/kombrink/src/kaldi/trunk/src/rnnrescore_/1.lats .
ln -s /mnt/matylda5/kombrink/src/kaldi/trunk/src/rnnrescore_/words.txt .
ln -s /mnt/matylda5/kombrink/src/kaldi/trunk/src/rnnrescore_/WSJ.35M.200cl.350h.kaldi.rnn .
ln -s /mnt/matylda5/kombrink/src/kaldi/trunk/src/rnnrescore_/WSJ.35M.200cl.350h.rnn .

make clean
make
