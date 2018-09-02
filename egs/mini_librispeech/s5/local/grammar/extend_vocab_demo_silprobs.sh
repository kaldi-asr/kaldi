#!/usr/bin/env bash

# This script demonstrates how to use the grammar-decoding framework to build
# graphs made out of more than one part.  (This version uses word-specific
# silence probabilities). It demonstrates using `fstequivalent`
# that the graph constructed this way is equivalent to what you would create if
# you had the LM all as a single piece.  This uses the command line tools to
# expand to a regular FST (--write-as-grammar=false) In practice you might not
# want do to that, since the result might be large, and since writing the entire
# thing might take too much time.  The code itself allows you to construct these
# GrammarFst objects in lightweight way and decode using them.

# Unfortunately the filenames here are not very well through through.  I hope to
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

  gunzip -c  data/local/lm/lm_tgsmall.arpa.gz | \
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
  steps/dict/train_g2p.sh --silence-phones $dict/silence_phones.txt $dict/lexicon.txt  $tree_dir/extvocab_g2p
fi


if [ $stage -le 4 ]; then
  # Create data/local/dict_newvocab as a dict-dir containing just the
  # newly created vocabulary entries (but the same phone list as our old setup, not
  # that it matters)

  mkdir -p $tree_dir/extvocab_lexicon

  # First find a list of words in the test set that are out of vocabulary.
  # Of course this is totally cheating.
  awk -v w=data/lang/words.txt 'BEGIN{while(getline <w) seen[$1] = $1} {for(n=2;n<=NF;n++) if(!($n in seen)) oov[$n] = 1}
                                END{ for(k in oov) print k;}' < data/dev_clean_2/text > $tree_dir/extvocab_lexicon/words
  echo "$0: generating g2p entries for $(wc -l <$tree_dir/extvocab_lexicon/words) words"

  if $run_g2p; then
    steps/dict/apply_g2p.sh $tree_dir/extvocab_lexicon/words $tree_dir/extvocab_g2p  $tree_dir/extvocab_lexicon
  else
    cat <<EOF >$tree_dir/extvocab_lexicon//lexicon.lex
HARDWIGG	0.962436	HH AA1 R D W IH1 G
SUDVESTR	0.162048	S AH1 D V EY1 S T R
SUDVESTR	0.133349	S AH1 D V EH1 S T R
SUDVESTR	0.114376	S AH1 D V EH1 S T ER0
VINOS	0.558345	V IY1 N OW0 Z
VINOS	0.068883	V AY1 N OW0 Z
VINOS	0.068431	V IY1 N OW0 S
DOMA	0.645714	D OW1 M AH0
DOMA	0.118255	D UW1 M AH0
DOMA	0.080682	D OW0 M AH0
GWYNPLAINE'S	0.983053	G W IH1 N P L EY1 N Z
SHIMERDA	0.610922	SH IH0 M EH1 R D AH0
SHIMERDA	0.175678	SH IY0 M EH1 R D AH0
SHIMERDA	0.069785	SH AY1 M ER1 D AH0
MYRDALS	0.479183	M IH1 R D AH0 L Z
MYRDALS	0.135225	M ER1 D AH0 L Z
MYRDALS	0.115478	M IH1 R D L Z
HEUCHERA	0.650042	HH OY1 K IH1 R AH0
HEUCHERA	0.119363	HH OY1 K EH1 R AH0
HEUCHERA	0.077907	HH OY1 K ER0 AH0
IMPARA	0.906222	IH0 M P AA1 R AH0
VERLOC'S	0.564847	V ER0 L AA1 K S
VERLOC'S	0.173540	V ER1 L AH0 K S
VERLOC'S	0.050543	V ER1 L AA1 K S
UNTRUSSING	0.998019	AH0 N T R AH1 S IH0 NG
DARFHULVA	0.317057	D AA2 F UH1 L V AH0
DARFHULVA	0.262882	D AA2 F HH UH1 L V AH0
DARFHULVA	0.064055	D AA2 F HH UW1 L V AH0
FINNACTA	0.594586	F IH1 N AH0 K T AH0
FINNACTA	0.232454	F IH1 N AE1 K T AH0
FINNACTA	0.044733	F IH1 N IH0 K T AH0
YOKUL	0.845279	Y OW1 K AH0 L
YOKUL	0.051082	Y OW2 K AH0 L
YOKUL	0.029435	Y OW0 K AH0 L
CONGAL	0.504228	K AA1 NG G AH0 L
CONGAL	0.151648	K AA2 NG G AH0 L
CONGAL	0.137837	K AH0 N JH AH0 L
DELECTASTI	0.632180	D IH0 L EH0 K T EY1 S T IY0
DELECTASTI	0.203808	D IH0 L EH1 K T EY1 S T IY0
DELECTASTI	0.066722	D IH0 L EH0 K T AE1 S T IY0
YUNDT	0.975077	Y AH1 N T
QUINCI	0.426115	K W IH1 N S IY0
QUINCI	0.369324	K W IH1 N CH IY0
QUINCI	0.064507	K W IY0 N CH IY0
BIRDIKINS	0.856979	B ER1 D IH0 K AH0 N Z
BIRDIKINS	0.045315	B ER1 D AH0 K AH0 N Z
SNEFFELS	0.928413	S N EH1 F AH0 L Z
FJORDUNGR	0.130629	F Y AO1 R D UW0 NG G R
FJORDUNGR	0.125082	F Y AO1 R D AH0 NG G R
FJORDUNGR	0.111035	F Y AO1 R D UH1 NG R
YULKA	0.540253	Y UW1 L K AH0
YULKA	0.295588	Y AH1 L K AH0
YULKA	0.076631	Y UH1 L K AH0
LACQUEY'S	0.987908	L AE1 K IY0 Z
OSSIPON'S	0.651400	AA1 S AH0 P AA2 N Z
OSSIPON'S	0.118444	AA1 S AH0 P AA0 N Z
OSSIPON'S	0.106377	AA1 S AH0 P AH0 N Z
SAKNUSSEMM	0.060270	S AE1 K N AH1 S EH1 M
SAKNUSSEMM	0.044992	S AE1 K N AH0 S EH1 M
SAKNUSSEMM	0.044084	S AA0 K N AH1 S EH1 M
CONGAL'S	0.618287	K AA1 NG G AH0 L Z
CONGAL'S	0.185952	K AA2 NG G AH0 L Z
CONGAL'S	0.115143	K AH0 N G AH0 L Z
TARRINZEAU	0.159153	T AA1 R IY0 N Z OW1
TARRINZEAU	0.136536	T AA1 R AH0 N Z OW1
TARRINZEAU	0.100924	T EH1 R IY0 N Z OW1
SHIMERDAS	0.230819	SH IH0 M EH1 R D AH0 Z
SHIMERDAS	0.216235	SH IH0 M EH1 R D AH0 S
SHIMERDAS	0.073311	SH AY1 M ER1 D AH0 Z
RUGGEDO'S	0.821285	R UW0 JH EY1 D OW0 Z
RUGGEDO'S	0.166825	R AH1 G AH0 D OW0 Z
CORNCAKES	0.934118	K AO1 R N K EY2 K S
VENDHYA	0.616662	V EH0 N D Y AH0
VENDHYA	0.178349	V EH1 N D Y AH0
VENDHYA	0.160768	V AA1 N D Y AH0
GINGLE	0.919815	G IH1 NG G AH0 L
STUPIRTI	0.422653	S T UW0 P IH1 R T IY0
STUPIRTI	0.126925	S T UW1 P IH0 R T IY0
STUPIRTI	0.078422	S T UW1 P AH0 R T IY0
HERBIVORE	0.950887	HH ER1 B IH0 V AO2 R
BRION'S	0.838326	B R AY1 AH0 N Z
BRION'S	0.140310	B R IY0 AH0 N Z
DELAUNAY'S	0.993259	D EH1 L AO0 N EY0 Z
KHOSALA	0.920908	K OW0 S AA1 L AH0
BRANDD	0.827461	B R AE1 N D
BRANDD	0.085646	B R AE2 N D
GARDAR	0.598675	G AA0 R D AA1 R
GARDAR	0.289831	G AA1 R D AA2 R
GARDAR	0.057983	G AA0 R D AA2 R
MACKLEWAIN	0.570209	M AE1 K AH0 L W EY0 N
MACKLEWAIN	0.101477	M AH0 K AH0 L W EY0 N
MACKLEWAIN	0.067905	M AE1 K AH0 L W EY2 N
LIBANO	0.993297	L IY0 B AA1 N OW0
MOLING	0.782578	M OW1 L IH0 NG
MOLING	0.059362	M OW2 L IH0 NG
MOLING	0.056217	M AA1 L IH0 NG
BENNYDECK'S	0.583859	B EH1 N IY0 D EH0 K S
BENNYDECK'S	0.276699	B EH1 N IH0 D EH0 K S
BENNYDECK'S	0.028343	B EH1 N IH0 D IH0 K S
MACKLEWAIN'S	0.615766	M AE1 K AH0 L W EY0 N Z
MACKLEWAIN'S	0.109585	M AH0 K AH0 L W EY0 N Z
MACKLEWAIN'S	0.039423	M AE1 K AH0 L W AH0 N Z
PRESTY	0.616071	P R EH1 S T IY0
PRESTY	0.288701	P R AH0 S T IY0
BREADHOUSE	0.995874	B R EH1 D HH AW2 S
BUZZER'S	0.992495	B AH1 Z ER0 Z
BHUNDA	0.502439	B UW1 N D AH0
BHUNDA	0.267733	B AH0 N D AH0
BHUNDA	0.193772	B UH1 N D AH0
PINKIES	0.998440	P IH1 NG K IY0 Z
TROKE	0.723320	T R OW1 K
TROKE	0.269707	T R OW2 K
OSSIPON	0.728486	AA1 S AH0 P AA2 N
OSSIPON	0.098752	AA1 S AH0 P AH0 N
OSSIPON	0.033957	AA1 S AH0 P AO0 N
RIVERLIKE	0.991731	R IH1 V ER0 L AY2 K
NICLESS	0.478183	N IH1 K L AH0 S
NICLESS	0.159889	N IH0 K L AH0 S
NICLESS	0.120611	N IH1 K L IH0 S
TRAMPE	0.959184	T R AE1 M P
VERLOC	0.610461	V ER0 L AA1 K
VERLOC	0.128479	V ER1 L AH0 K
VERLOC	0.073687	V ER1 L AA0 K
GANNY	0.991703	G AE1 N IY0
AMBROSCH	0.302906	AE0 M B R OW1 SH
AMBROSCH	0.201163	AE0 M B R AO1 SH
AMBROSCH	0.109274	AE1 M B R AO1 SH
FIBI	0.619154	F IH1 B IY0
FIBI	0.163168	F IY1 B IY0
FIBI	0.083443	F AY1 B IY0
IROLG	0.823123	IH0 R OW1 L G
IROLG	0.053196	IH0 R OW1 L JH
IROLG	0.021038	IH0 R OW1 L JH IY1
BALVASTRO	0.251546	B AA0 L V AA1 S T R OW0
BALVASTRO	0.213351	B AE0 L V AE1 S T R OW0
BALVASTRO	0.133005	B AA0 L V AE1 S T R OW0
BOOLOOROO	0.676757	B UW1 L UW1 R UW0
BOOLOOROO	0.173653	B UW1 L UH2 R UW0
BOOLOOROO	0.086501	B UW1 L UH0 R UW0
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
  # steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 --frames-per-chunk 140 --nj 38 \
  #   --cmd "queue.pl --mem 4G --num-threads 4" --online-ivector-dir exp/nnet3/ivectors_dev_clean_2_hires \
  #   exp/chain/tree_sp/graph_tgsmall data/dev_clean_2_hires exp/chain/tdnn1h_sp/decode_tgsmall_dev_clean_2

  # We just replace the graph with the one in $treedir/extvocab_combined.

  steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 --frames-per-chunk 140 --nj 38 \
    --cmd "queue.pl --mem 4G --num-threads 4" --online-ivector-dir exp/nnet3/ivectors_dev_clean_2_hires \
    exp/chain/tree_sp/extvocab_combined data/dev_clean_2_hires exp/chain/tdnn1h_sp/decode_tgsmall_dev_clean_2_ev_comb

  # s5: grep WER exp/chain/tdnn1h_sp/decode_tgsmall_dev_clean_2_ev_comb/wer_* | utils/best_wer.sh
  # %WER 11.42 [ 2300 / 20138, 227 ins, 275 del, 1798 sub ] exp/chain/tdnn1h_sp/decode_tgsmall_dev_clean_2_ev_comb/wer_12_0.0

  #.. versus the baseline below:
  # s5: grep WER exp/chain/tdnn1h_sp/decode_tgsmall_dev_clean_2/wer_* | utils/best_wer.sh
  # %WER 12.01 [ 2418 / 20138, 244 ins, 307 del, 1867 sub ] exp/chain/tdnn1h_sp/decode_tgsmall_dev_clean_2/wer_13_0.0
fi

if [ $stage -le 9 ]; then
 steps/nnet3/decode_grammar.sh --acwt 1.0 --post-decode-acwt 10.0 --frames-per-chunk 140 --nj 38 \
    --cmd "queue.pl --mem 4G --num-threads 4" --online-ivector-dir exp/nnet3/ivectors_dev_clean_2_hires \
    exp/chain/tree_sp/extvocab_combined data/dev_clean_2_hires exp/chain/tdnn1h_sp/decode_tgsmall_dev_clean_2_ev_comb_gra

 # WER with grammar decoding is exactly the same as decoding from the converted FST.
 # grep WER exp/chain/tdnn1h_sp/decode_tgsmall_dev_clean_2_ev_comb_gra/wer_* | utils/best_wer.sh
 # %WER 11.42 [ 2300 / 20138, 227 ins, 275 del, 1798 sub ] exp/chain/tdnn1h_sp/decode_tgsmall_dev_clean_2_ev_comb_gra/wer_12_0.0
fi
