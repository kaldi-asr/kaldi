#!/usr/bin/env bash

# This script demonstrates how to use the grammar-decoding framework to build
# graphs made out of more than one part.  It demonstrates using `fstequivalent`
# that the graph constructed this way is equivalent to what you would create if
# you had the LM all as a single piece.  This uses the command line tools to
# expand to a regular FST (--write-as-grammar=false) In practice you might not
# want do to that, since the result might be large, and since writing the entire
# thing might take too much time.  The code itself allows you to construct these
# GrammarFst objects in lightweight way and decode using them.

# Unfortunately the filenames here are not very well thought through.  I hope to
# rework this when I have time.

stage=0
run_g2p=false  # set this to true to run the g2p stuff, it's slow so
               # by default we fake it by providing what it previously output
set -e

. ./path.sh
. utils/parse_options.sh


tree_dir=exp/chain/tree_sp
lang_base=data/lang_basevocab
lang_ext=data/lang_extvocab

# For the purposes of this script we just need a biphone tree and associated
# transition-model for testing, because we're testing it at the graph level,
# i.e. testing equivalence of compiled HCLG graphs; there is no decoding
# involved here.

# We're doing this with the "no-silprobs" dictionary dir for now, as we
# need to write some scripts to support silprobs with this.

# For reference, here is how we could create the 'lang' dir for the
# baseline.
#utils/prepare_lang.sh data/local/dict \
#   "<UNK>" data/local/lang_tmp data/lang

if [ $stage -le 0 ]; then
  cp -r data/local/dict data/local/dict_basevocab
  echo "#nonterm:unk" > data/local/dict_basevocab/nonterminals.txt

  utils/prepare_lang.sh data/local/dict_basevocab \
       "<UNK>" data/local/lang_tmp $lang_base
fi

if [ $stage -le 1 ]; then
  # note: <UNK> does appear in that arpa file, with a reasonable probability
  # (0.0)...  presumably because the vocab that the arpa file was built with was
  # not vast, so there were plenty of OOVs.  It would be possible to adjust its
  # probability with adjust_unk_arpa.pl, but for now we just leave it as-is.
  # The <UNK> appears quite a few times in the ARPA.  In the language model we
  # replaced it with #nonterm:unk, which will later expand to our custom graph
  # of new words.

  # We don't want the #nonterm:unk on the output side of G.fst, or it would
  # appear in the decoded output, so we remove it using the 'fstrmsymbols' command.

  nonterm_unk=$(grep '#nonterm:unk' $lang_base/words.txt | awk '{print $2}')

  gunzip -c  data/local/lm/tgsmall.arpa.gz | \
    sed 's/<UNK>/#nonterm:unk/g' | \
    arpa2fst --disambig-symbol=#0 \
             --read-symbol-table=$lang_base/words.txt - | \
    fstrmsymbols --remove-from-output=true "echo $nonterm_unk|" - $lang_base/G.fst
fi


if [ $stage -le 2 ]; then
  # make the top-level part of the graph.
  utils/mkgraph.sh --self-loop-scale 1.0 $lang_base $tree_dir $tree_dir/extvocab_top
fi

if [ $stage -le 3 ] && $run_g2p; then
  # you may have to do some stuff manually to install sequitur, to get this to work.
  dict=data/local/dict_basevocab
  #steps/dict/train_g2p.sh --silence-phones $dict/silence_phones.txt $dict/lexicon.txt  $tree_dir/extvocab_g2p
  local/g2p/train_g2p.sh --silence-phones $dict/silence_phones.txt $dict/lexicon.txt  $tree_dir/extvocab_g2p
fi


if [ $stage -le 4 ]; then
  # Create data/local/dict_newvocab as a dict-dir containing just the
  # newly created vocabulary entries (but the same phone list as our old setup, not
  # that it matters)

  mkdir -p $tree_dir/extvocab_lexicon

  # First find a list of words in the test set that are out of vocabulary.
  # Of course this is totally cheating.
  awk -v w=data/lang/words.txt 'BEGIN{while(getline <w) seen[$1] = $1} {for(n=2;n<=NF;n++) if(!($n in seen)) oov[$n] = 1}
                                END{ for(k in oov) print k;}' < data/dev/text > $tree_dir/extvocab_lexicon/words
  echo "$0: generating g2p entries for $(wc -l <$tree_dir/extvocab_lexicon/words) words"

  if $run_g2p; then
    steps/dict/apply_g2p.sh $tree_dir/extvocab_lexicon/words $tree_dir/extvocab_g2p  $tree_dir/extvocab_lexicon
  else
    cat <<EOF >$tree_dir/extvocab_lexicon//lexicon.lex
abalone	0.962436	aa bb aa ll oo nn
babaganoush	0.162048	bb aa bb aa gg aa nn ou ch
cabalistique	0.133349	kk aa bb aa ll ii ss tt ii kk
EOF
  fi

  # extend_lang.sh needs it to have basename 'lexiconp.txt'.
  mv $tree_dir/extvocab_lexicon/lexicon.lex $tree_dir/extvocab_lexicon/lexiconp.txt

  [ -f data/lang_extvocab/G.fst ] && rm data/lang_extvocab/G.fst
  utils/lang/extend_lang.sh  data/lang_basevocab $tree_dir/extvocab_lexicon/lexiconp.txt  data/lang_extvocab
fi

if [ $stage -le 5 ]; then
  # make the G.fst for the extra words.  Just assign equal probabilities to all of
  # them.  The words will all transition from state 1 to 2.
  cat <<EOF > $lang_ext/G.txt
0    1    #nonterm_begin <eps>
2    3    #nonterm_end <eps>
3
EOF
  lexicon=$tree_dir/extvocab_lexicon/lexiconp.txt
  num_words=$(wc -l <$lexicon)
  cost=$(perl -e "print log($num_words)");
  awk -v cost=$cost '{print 1, 2, $1, $1, cost}' <$lexicon >>$lang_ext/G.txt
  fstcompile --isymbols=$lang_ext/words.txt --osymbols=$lang_ext/words.txt <$lang_ext/G.txt | \
    fstarcsort --sort_type=ilabel >$lang_ext/G.fst
fi

if [ $stage -le 6 ]; then
  # make the part of the graph that will be included.
  # Refer to the 'compile-graph' commands in ./simple_demo.sh for how you'd do
  # this in code.
  utils/mkgraph.sh --self-loop-scale 1.0 $lang_ext $tree_dir $tree_dir/extvocab_part
fi

if [ $stage -le 7 ]; then
  offset=$(grep nonterm_bos $lang_ext/phones.txt | awk '{print $2}')
  nonterm_unk=$(grep nonterm:unk $lang_ext/phones.txt | awk '{print $2}')

  mkdir -p $tree_dir/extvocab_combined
  [ -d $tree_dir/extvocab_combined/phones ] && rm -r $tree_dir/extvocab_combined/phones
  # the decoding script expects words.txt and phones/, copy them from the extvocab_part
  # graph directory where they will have suitable values.
  cp -r $tree_dir/extvocab_part/{words.txt,phones.txt,phones/} $tree_dir/extvocab_combined

  # the following, due to --write-as-grammar=false, compiles it into an FST
  # which can be decoded by our normal decoder.
  make-grammar-fst --write-as-grammar=false --nonterm-phones-offset=$offset $tree_dir/extvocab_top/HCLG.fst \
                   $nonterm_unk $tree_dir/extvocab_part/HCLG.fst  $tree_dir/extvocab_combined/HCLG.fst

  # the following compiles it and writes as GrammarFst.  The size is 176M, vs. 182M for HCLG.fst.
  # In other examples, of course the difference might be more.

  make-grammar-fst --write-as-grammar=true --nonterm-phones-offset=$offset $tree_dir/extvocab_top/HCLG.fst \
                $nonterm_unk $tree_dir/extvocab_part/HCLG.fst  $tree_dir/extvocab_combined/HCLG.gra
fi


if [ $stage -le 8 ]; then
  # OK, now we actually decode the test data.  For reference, the command which was used to
  # decode the test data in the current (at the time of writing) chain TDNN system
  # local/chain/run_tdnn.sh (as figured out by running it from that stage), was:
  # steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 --frames-per-chunk 140 --nj 23 \
  #   --cmd "queue.pl --mem 4G --num-threads 4" --online-ivector-dir exp/nnet3/ivectors_dev_clean_2_hires \
  #   exp/chain/tree_sp/graph_tgsmall data/dev_clean_2_hires exp/chain/tdnn1h_sp/decode_tgsmall_dev_clean_2

  # We just replace the graph with the one in $treedir/extvocab_combined.

  steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 --frames-per-chunk 140 --nj 23 \
    --cmd "run.pl --mem 4G --num-threads 4" --online-ivector-dir exp/nnet3/ivectors_dev_hires \
    exp/chain/tree_sp/extvocab_combined data/dev_hires exp/chain/tdnn1a_sp/decode_tgsmall_dev_ev_comb



#  grep WER exp/chain/tdnn1h_sp/decode_tgsmall_dev_ev_comb/wer_* | utils/best_wer.sh


 #.. versus the baseline below note, the baseline is not 100% comparable as it used the
 #   silence probabilities, which the grammar-decoding does not (yet) support...
 # s5: grep WER exp/chain/tdnn1h_sp/decode_tgsmall_dev_clean_2/wer_* | utils/best_wer.sh
 # %WER 12.01 [ 2418 / 20138, 244 ins, 307 del, 1867 sub ] exp/chain/tdnn1h_sp/decode_tgsmall_dev_clean_2/wer_13_0.0
fi

if [ $stage -le 9 ]; then
  steps/nnet3/decode_grammar.sh --acwt 1.0 --post-decode-acwt 10.0 --frames-per-chunk 140 --nj 23 \
    --cmd "run.pl --mem 4G --num-threads 4" --online-ivector-dir exp/nnet3/ivectors_dev_hires \
    exp/chain/tree_sp/extvocab_combined data/dev_hires exp/chain/tdnn1a_sp/decode_tgsmall_dev_ev_comb_gra
fi
