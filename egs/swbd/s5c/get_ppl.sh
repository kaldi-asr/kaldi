
. path.sh
. cmd.sh

# variables for lattice rescoring
run_rescore=false
ac_model_dir=exp/chain/tdnn_lstm_1e_sp
decode_dir_suffix=rnnlm_adaptation_dan_formula_max_2
ngram_order=4 # approximate the lattice-rescoring by limiting the max-ngram-order
              # if it's set, it merges histories in the lattice if they share
              # the same ngram history and this prevents the lattice from 
              # exploding exponentially

src_dir=exp/chain/tdnn_lstm_1e_sp/decode_eval2000_fsh_sw1_fg
stage=2

. parse_options.sh

if [ $stage -le 1 ]; then
#  cat data/train/text | cut -d " " -f2- | sed "s= =\n=g" | grep . | sort | uniq -c | awk '{print $2, $1}' > train.unigram

#  cat data/train/text | cut -d " " -f2- | sym2int.pl --map-oov "<unk>" data/lang/words.txt | sed "s= =\n=g" | grep . | sort | uniq -c | awk '{print $2, $1}' | sort -k1n > train.unigram
#  cat data/train/text | sym2int.pl -f 2- --map-oov "<unk>" data/lang/words.txt > train.txt
  cat data/eval2000/text | cut -d " " -f2- | tr A-Z a-z > test.raw
  cat data/eval2000/text | cut -d " " -f1 > test.head
  paste test.head test.raw | sym2int.pl -f 2- --map-oov "<unk>" data/lang/words.txt > test.txt
#  lattice-arc-post --acoustic-scale=0.1 "ark:gunzip -c $src_dir/lat.*.gz|" post.txt
#
#  cat post.txt | sed 's=_= =g' | awk '{print $1"_"$2,$6,$7}' | awk '{a[$1][$3]+=$2}END{for(i in a) for(j in a[i]) print i, j, a[i][j]}' > maps
#
#  cat data/train/text | cut -d " " -f2- | sed "s= =\n=g" | grep . | sort | uniq -c | awk '{print $2, $1}' > train.unigram
#
fi

dir=exp/rnnlm_lstm_1c

word_embedding="rnnlm-get-word-embedding $dir/word_feats.txt $dir/feat_embedding.final.mat -|"

rnnlm-nbest-probs-adjust $(cat $dir/special_symbol_opts.txt) $dir/final.raw "$word_embedding" test.txt data/eval2000/utt2spk train.unigram

exit

if [ $stage -le 2 ]; then
  echo Perform lattice-rescoring on $ac_model_dir
  LM=fsh_sw1_tg
  for decode_set in eval2000; do
    decode_dir=${ac_model_dir}/decode_${decode_set}_$LM

    # Lattice rescoring
    rnnlm/lmrescore_rnnlm_lat_adapt.sh \
      --cmd "$decode_cmd --mem 4G -l hostname='[bc]*'" \
      --weight 0.5 --max-ngram-order $ngram_order \
      data/lang_$LM $dir \
      data/${decode_set}_hires ${decode_dir} \
      ${decode_dir}_${decode_dir_suffix} data/eval2000/utt2spk train.unigram

  done
fi
