#./rnnrescore --n=100 words.txt WSJ.35M.200cl.350h.kaldi.rnn ark,t:/mnt/matylda5/kombrink/src/kaldi/trunk/egs/wsj/s1/exp/decode_tri1kn_latgen_tgpr_eval92/1.lats ark,t:nbest.lats
#../../tools/openfst/bin/fstreverse

ln -sf /mnt/matylda5/kombrink/src/kaldi/debug/src/rnnlm-rescore/Eigen/ .
ln -sf /mnt/matylda5/kombrink/src/kaldi/trunk/egs/wsj/s1/exp/decode_tri1ckn_latgen_tgpr_eval92/1.lats .
ln -sf /mnt/matylda5/kombrink/src/kaldi/trunk/src/rnnrescore_/words.txt .
ln -sf /mnt/matylda5/kombrink/src/kaldi/trunk/src/rnnrescore_/WSJ.35M.200cl.350h.kaldi.rnn .
ln -sf /mnt/matylda5/kombrink/src/kaldi/trunk/src/rnnrescore_/WSJ.35M.200cl.350h.rnn .

make clean
make

./rnn-rescore --acoustic-scale=0.0625 --lm-scale=1 --n=1000 words.txt ark,t:1.lats WSJ.35M.200cl.350h.kaldi.rnn ark,t:nbest.lats

cat nbest.lats | awk 'BEGIN{while (getline<"words.txt")v[$2]=$1}{$4="";if ($3 in v)$3=v[$3];print $0}' | less
#paste ngrscores rnnscores ams | awk 'BEGIN{lambda=0.75}{lmmix=2.3*lambda*$2 + (1-lambda)*$1; ams=$3; print ams+lmmix}'> combinedScores

paste ngrscores rnnscores ams | awk 'BEGIN{lambda=0.75}{lmmix=2.3*lambda*$2 + (1-lambda)*$1; ams=$3; print ams+lmmix}'> combinedScores
paste combinedScores debug3 | awk '{if ((utt_score[$2]>$1)||(utt_score[$2]=="")){utt_score[$2]=$1;utt[$2]=$0}}END{for (u in utt_score){print utt[u]}}' | awk '{$1="";print $0}' | sed 's/^ //'| sort > newbest.txt

compute-wer --text --mode=present ark:/mnt/matylda5/kombrink/src/kaldi/trunk/egs/wsj/s1/exp/decode_tri1ckn_latgen_tgpr_eval92/test_trans.filt ark:newbest.txt 


 time /mnt/matylda5/kombrink/src/rnnlm-0.3a/rnnlm -rnnlm /mnt/matylda5/imikolov/0WSJ/3/v4/2/model  -nbest -test debug3 -debug 0 | tee tomsrnnscore

# score all
for f in 1best*.txt; do echo $f;compute-wer --text --mode=present ark:/mnt/matylda5/kombrink/src/kaldi/trunk/egs/wsj/s1/exp/decode_tri1ckn_latgen_tgpr_eval92/test_trans.filt ark:$f; echo ""; done > results.txt
