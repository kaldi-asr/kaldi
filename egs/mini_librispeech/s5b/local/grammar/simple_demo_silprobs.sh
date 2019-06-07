#!/usr/bin/env bash

# simple_demo_silprobs.sh is a version of simple_demo.sh that uses a lexicon
# with word-specific silence probabilities.

# These scripts demonstrate how to use the grammar-decoding framework to build
# graphs made out of more than one part.  It demonstrates using `fstequivalent`
# that the graph constructed this way is equivalent to what you would create if
# you had the LM all as a single piece.  This uses the command line tools to
# expand to a regular FST (--write-as-grammar=false) In practice you might not
# want do to that, since the result might be large, and since writing the entire
# thing might take too much time.  The code itself allows you to construct these
# GrammarFst objects in lightweight way and decode using them.

stage=0
set -e
. ./path.sh
. utils/parse_options.sh


tree_dir=exp/chain/tree_sp

# For the purposes of this script we just need a biphone tree and associated
# transition-model for testing, because we're testing it at the graph level,
# i.e. testing equivalence of compiled HCLG graphs; there is no decoding
# involved here.


# For reference, the original command we
#utils/prepare_lang.sh data/local/dict \
#   "<UNK>" data/local/lang_tmp data/lang

if [ $stage -le 0 ]; then
  [ -d data/local/dict_grammar1 ] && rm -r data/local/dict_grammar1
  cp -r data/local/dict data/local/dict_grammar1
  echo "#nonterm:contact_list" > data/local/dict_grammar1/nonterminals.txt

  utils/prepare_lang.sh data/local/dict_grammar1 \
       "<UNK>" data/local/lang_tmp data/lang_grammar1
fi



if [ $stage -le 1 ]; then
  # Most contents of these directories will be the same, only G.fst differs, but
  # it's our practice to make these things as directories combining G.fst with
  # everything else.
  rm -r  data/lang_grammar2{a,b} 2>/dev/null || true
  cp -r data/lang_grammar1 data/lang_grammar2a
  cp -r data/lang_grammar1 data/lang_grammar2b
fi

if [ $stage -le 2 ]; then
  # Create a simple G.fst in data/lang_grammar1, which won't
  # actually use any grammar stuff, it will be a baseline to test against.

  lang=data/lang_grammar1
  cat <<EOF | fstcompile --isymbols=$lang/words.txt --osymbols=$lang/words.txt | \
     fstarcsort --sort_type=ilabel > $lang/G.fst
0    1    GROUP  GROUP
1    2    ONE   ONE   0.69314718055994
1    2    TWO   TWO  0.69314718055994
1    2    <eps>  <eps>  5.0
2    3    ASSIST   ASSIST  0.69314718055994
2  0.69314718055994
3
EOF
  utils/mkgraph.sh --self-loop-scale 1.0 $lang $tree_dir $tree_dir/grammar1

  # test that the binary 'compile-graph' does the same thing as mkgraph.sh.
  compile-graph --read-disambig-syms=$lang/phones/disambig.int $tree_dir/tree $tree_dir/1.mdl $lang/L_disambig.fst $lang/G.fst $tree_dir/grammar1/HCLG2.fst

  if ! fstequivalent --delta=0.01 --random=true --npath=100 $tree_dir/grammar1/HCLG{,2}.fst; then
    echo "$0: two methods of producing graph in $tree_dir/grammar1 were different."
    exit 1
  fi
fi


if [ $stage -le 3 ]; then
  # create the top-level graph in data/lang_grammar2a

  # you can of course choose to put what symbols you want on the output side, as
  # long as they are defined in words.txt.  #nonterm:contact_list, #nonterm_begin
  # and #nonterm_end would be defined in this example.  This might be useful in
  # situations where you want to keep track of the structure of calling
  # nonterminals.
  lang=data/lang_grammar2a
  cat <<EOF | fstcompile --isymbols=$lang/words.txt --osymbols=$lang/words.txt | \
      fstarcsort --sort_type=ilabel > $lang/G.fst
0    1    GROUP   GROUP
1    2    #nonterm:contact_list  <eps>
2    3    ASSIST   ASSIST  0.69314718055994
2  0.69314718055994
3
EOF
  utils/mkgraph.sh --self-loop-scale 1.0 $lang $tree_dir $tree_dir/grammar2a

  # test that the binary 'compile-graph' does the same thing as mkgraph.sh.
  offset=$(grep nonterm_bos $lang/phones.txt | awk '{print $2}') # 364
  compile-graph --nonterm-phones-offset=$offset --read-disambig-syms=$lang/phones/disambig.int \
       $tree_dir/tree $tree_dir/1.mdl $lang/L_disambig.fst $lang/G.fst $tree_dir/grammar2a/HCLG2.fst

  if ! fstequivalent --delta=0.01 --random=true --npath=100 $tree_dir/grammar2a/HCLG{,2}.fst; then
    echo "$0: two methods of producing graph in $tree_dir/grammar2a were different."
    exit 1
  fi
fi

if [ $stage -le 4 ]; then
  # Create the graph for the nonterminal in data/lang_grammar2b
  # Again, we don't choose to put these symbols on the output side, but it would
  # be possible to do so.
  lang=data/lang_grammar2b
  cat <<EOF | fstcompile --isymbols=$lang/words.txt --osymbols=$lang/words.txt \
     |  fstarcsort --sort_type=ilabel > $lang/G.fst
0    1    #nonterm_begin <eps>
1    2    ONE  ONE    0.69314718055994
1    2    TWO  TWO    0.69314718055994
1    2    <eps>  <eps>  5.0
2    3    #nonterm_end <eps>
3
EOF
  utils/mkgraph.sh --self-loop-scale 1.0 $lang $tree_dir $tree_dir/grammar2b


  # test that the binary 'compile-graph' does the same thing as mkgraph.sh.
  offset=$(grep nonterm_bos $lang/phones.txt | awk '{print $2}') # 364
  compile-graph --nonterm-phones-offset=$offset --read-disambig-syms=$lang/phones/disambig.int \
       $tree_dir/tree $tree_dir/1.mdl $lang/L_disambig.fst $lang/G.fst $tree_dir/grammar2b/HCLG2.fst

  if ! fstequivalent --delta=0.01 --random=true --npath=100 $tree_dir/grammar2b/HCLG{,2}.fst; then
    echo "$0: two methods of producing graph in $tree_dir/grammar2b were different."
    exit 1
  fi
fi

if [ $stage -le 5 ]; then
  # combine the top-level graph and the sub-graph together using the command
  # line tools. (In practice you might want to do this from appliation code).

  lang=data/lang_grammar2a
  offset=$(grep nonterm_bos $lang/phones.txt | awk '{print $2}') # 364
  clist=$(grep nonterm:contact_list $lang/phones.txt | awk '{print $2}') # 368

  # the graph in $tree_dir/grammar2/HCLG.fst will be a normal FST (ConstFst)
  # that was expanded from the grammar.  (we use --write-as-grammar=false to
  # make it expand it).  This is to test equivalence to the one in
  # $tree_dir/grammar1/
  mkdir -p $tree_dir/grammar2
  make-grammar-fst --write-as-grammar=false --nonterm-phones-offset=$offset $tree_dir/grammar2a/HCLG.fst \
                   $clist $tree_dir/grammar2b/HCLG.fst  $tree_dir/grammar2/HCLG.fst
fi

if [ $stage -le 6 ]; then
  # Test equivalence using a random path.. can be useful for debugging if
  # fstequivalent fails.
  echo "$0: will print costs with the two FSTs, for one random path."
  fstrandgen $tree_dir/grammar1/HCLG.fst > path.fst
  for x in 1 2; do
    fstproject --project_output=false path.fst | fstcompose - $tree_dir/grammar${x}/HCLG.fst | fstcompose - <(fstproject --project_output=true path.fst) > composed.fst
    start_state=$(fstprint composed.fst | head -n 1 | awk '{print $1}')
    fstshortestdistance --reverse=true composed.fst | awk -v s=$start_state '{if($1 == s) { print $2; }}'
  done

fi

if [ $stage -le 7 ]; then
  echo "$0: will test equivalece using fstequivalent"
  if fstequivalent --delta=0.01 --random=true --npath=100 $tree_dir/grammar1/HCLG.fst $tree_dir/grammar2/HCLG.fst; then
    echo "$0: success: the two were equivalent"
  else
    echo "$0: failure: the two were inequivalent"
  fi
fi
