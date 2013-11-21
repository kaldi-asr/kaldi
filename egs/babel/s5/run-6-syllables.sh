#!/bin/bash 

# This script creates the data directories for us to run a syllable-based system
# in the directory with the same name as this, but with the -syllables suffix.

# dependency and build a LM on the phone level.  We get the syllable sequences from one
# of the existing systems' alignments.

set -e           #Exit on non-zero return code from any command
set -o pipefail  #Exit if any of the commands in the pipeline will 
                 #return non-zero return code
set -u           #Fail on an undefined variable

. conf/common_vars.sh || exit 1;
. ./lang.conf || exit 1;

# extra config:
syllable_ins_prob=0.01  # relative to prob of <unk>

# We will create a directory with the same name as this one, but
# with "-syllables" at the end of its name.
target=`pwd`-syllables


if [ ! -d $target ]; then
  echo ---------------------------------------------------------------------
  echo "Creating directory $target for the syllable system"
  echo ---------------------------------------------------------------------
  mkdir -p $target
  cp *.sh $target
  for x in steps utils local conf lang.conf; do
    [ ! -s $x ] && echo "No such file or directory $x" && exit 1;
    if [ -L $x ]; then # if these are links    
      cp -d $x $target # copy the link over.
    else # create a link to here.
      ln -s ../`basename $PWD`/$x $target
    fi
  done
fi

mkdir -p $target/data

for dir in raw_train_data raw_dev2h_data raw_dev10h_data; do
  ln -s `pwd`/data/$dir $target/data/ || true;
done

if [ ! -f data/local_withprob/.done ]; then
  echo -------------------------------------------------------------------------------
  echo "Creating lexicon with probabilities, data/local_withprob/lexiconp.txt on `date`"
  echo -------------------------------------------------------------------------------

  cp -rT data/local data/local_withprob
  steps/get_lexicon_probs.sh data/train data/lang exp/tri5 data/local/lexicon.txt \
    exp/tri5_lexprobs data/local_withprob/lexiconp.txt || exit 1;
  touch data/local_withprob/.done
fi


mkdir -p $target/data/local # we'll put some stuff here...

if [ ! -f $target/data/local/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Creating syllable-level lexicon in directory $target/data/local/w2s"
  echo ---------------------------------------------------------------------

  mkdir -p $target/data/local/w2s

  ! local/make_syllable_lexicon.sh --pron-probs true data/local_withprob/lexiconp.txt \
    $target/data/local/w2s/lexiconp.txt $target/data/local/lexicon.txt && \
    echo "Error creating syllable lexicon" && exit 1;

  for f in extra_questions.txt silence_phones.txt nonsilence_phones.txt optional_silence.txt; do
    cp data/local/$f $target/data/local/$f
  done

  touch $target/data/local/.done
fi

if [ ! -f $target/data/lang/.done ]; then
  echo ------------------------------------------------------------------------
  echo "Creating lang directories $target/data/lang and $target/data/lang_nopos" 
  echo ------------------------------------------------------------------------

  # create the "lang" directory for the per-syllable setup.
  # note: the position dependency will now be on the syllable level, not the phone level.
  # Now we specify zero silence prob, because we'll be putting the silence in
  # the language model in order to reduce the chances of it being inserted between syllables
  # where it shouldn't belong.
  # The oov symbols '<oov>' doesn't really matter for this setup as there will be no OOVs;
  # the scripts just require it.
  utils/prepare_lang.sh --sil-prob 0.0 \
    $target/data/local/ '<oov>' $target/data/local/tmp $target/data/lang

  utils/prepare_lang.sh --sil-prob 0.0 --position-dependent-phones false \
    $target/data/local/ '<oov>' $target/data/local/tmp $target/data/lang_nopos

  # later stages require word-symbol tables of lang and lang_nopos be the same.
  cmp $target/data/lang/words.txt $target/data/lang_nopos/words.txt  || exit 1; 

  touch $target/data/lang/.done
fi

if [ ! -f $target/data/local/w2s/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Creating word-to-syllable lexicon FSTs"
  echo --------------------------------------------------------------------

  cp data/lang/words.txt $target/data/local/w2s/words.txt
  cp $target/data/lang/words.txt $target/data/local/w2s/phones.txt

  # This FST will be used for alignment of the current system to get syllable-level
  # alignments.
  utils/make_lexicon_fst.pl --pron-probs $target/data/local/w2s/lexiconp.txt \
     0.5 `cat data/lang/phones/optional_silence.txt` | \
    fstcompile --osymbols=data/lang/words.txt --isymbols=$target/data/lang/words.txt | \
    fstarcsort --sort_type=olabel > $target/data/local/w2s/L.fst

  touch $target/data/local/w2s/.done
fi

if [ ! -f $target/data/local/w2s_extended/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Creating lexicon in $target/data/local/w2s_extended, for word decoding"
  echo "(allows penalized insertion of syllables we can't form into words)."
  echo --------------------------------------------------------------------
  dir=$target/data/local/w2s_extended
  mkdir -p $dir

  cat $target/data/local/w2s/lexiconp.txt | \
    perl -e ' $syll_prob = shift @ARGV;
    ($syll_prob > 0.0 && $syll_prob <= 1.0) || die "Bad prob $prob";
    while(<STDIN>) {
      @A = split;
      $word = shift @A;  $prob = shift @A; 
      if (!($prob > 0.0 && $prob <= 1.0)) { die "Bad pron-prob $prob in line $_"; }
      if (@A == 1) { $seen{$A[0]} = 1; } # saw this syllable as singleton.
      foreach $a ( @A ) { $is_syllable{$a} = 1; }
      print; # print out the line; all lines of the lexicon get printed, plus some more.
    }
    foreach $syllable (keys %is_syllable) {
      if (! defined $seen{$syllable}) { # did not see as singleton pron.
         print "<unk> $syll_prob $syllable\n"; # print new pron, <unk>.
      } } ' $syllable_ins_prob > $dir/lexiconp.txt
  ndisambig=`utils/add_lex_disambig.pl --pron-probs $dir/lexiconp.txt $dir/lexiconp_disambig.txt`
  ndisambig=$[$ndisambig+1]; # add one disambig symbol for silence in lexicon FST. [may not be needed here]

  ( for n in `seq 0 $ndisambig`; do echo '#'$n; done ) >$dir/disambig.txt
  cat $target/data/lang/words.txt | awk '{print $1}' | cat - <(for n in `seq $ndisambig`; do echo '#'$n; done) | \
    awk '{n=NR-1; print $1, n;}' > $dir/syllables.txt # syllable-level symbol table.
  utils/sym2int.pl $dir/syllables.txt $dir/disambig.txt >$dir/disambig.int
  cp data/lang/words.txt $dir/words.txt # word-level symbol table.

  # Make the lexicon with disambig symbols; determinize; remove disambig symbols.  
  utils/make_lexicon_fst.pl --pron-probs $dir/lexiconp_disambig.txt 0.5 SIL "#$ndisambig" | \
    fstcompile --osymbols=$dir/words.txt --isymbols=$dir/syllables.txt | \
    fstdeterminizestar | fstrmsymbols $dir/disambig.int > $dir/Ldet.fst

  # Remove the #0 symbol from the input of word-level G.fst, and remove 
  # <silence> from the output, and copy it over.  We remove <silence> from the
  # output because it won't matter for scoring and because it was causing some 
  # problems for phone-alignment which is part of word alignment. (a kind of blowup
  # from symbols getting out of sync; problems only happens if have 1 symbol per phone).
  disambig_word=`grep -w "#0" data/lang/words.txt | awk '{print $2}'`
  silence_word=`grep -w "<silence>" data/lang/words.txt | awk '{print $2}'`
  fstrmsymbols --remove-from-output=false "echo $disambig_word|" data/lang/G.fst | \
   fstrmsymbols --remove-from-output=true "echo $silence_word|"  | fstarcsort > $dir/G.fst


  mkdir -p $dir/tmp
  # remove pron-probs from syllable-level lexicon
   
  # Get syllable-level lexicon with word-position dependent phones.
  cat $target/data/lang/phones/align_lexicon.txt | cut -d ' ' -f 2- > $dir/tmp/syllable_lexicon.txt

  # remove pron-probs from word-level lexicon.
  cat $dir/lexiconp.txt | perl -ape 's/(\S+\s+)\S+\s+(.+)/$1$2/;' >$dir/lexicon.txt
   
  utils/apply_map.pl -f 2- $dir/tmp/syllable_lexicon.txt < $dir/lexicon.txt \
     >$dir/tmp/word2phone_lexicon.txt

  echo "<eps> SIL_S" >>  $dir/tmp/word2phone_lexicon.txt  # add optional silence.  It only appears
     # in the form SIL_S, because silence is always a word in the syllable setup, never optional.
  awk '{print $1, $0;}' < $dir/tmp/word2phone_lexicon.txt > $dir/word_align_lexicon.txt # duplicate 1st field

  utils/sym2int.pl -f 1-2 $dir/words.txt < $dir/word_align_lexicon.txt | \
    utils/sym2int.pl -f 3- $target/data/lang/phones.txt > $dir/word_align_lexicon.int
fi

if [ ! -f $target/data/train/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Using the data alignments to get syllable-level pronunciations"
  echo --------------------------------------------------------------------
  local/get_syllable_text.sh data/train data/lang $target/data/lang_nopos \
    $target/data/local/w2s/L.fst \
     exp/tri5_ali exp/tri5_align_syllables $target/data/train
  touch $target/data/train/.done
fi

if [ ! -f exp/tri5_dev2h_ali/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Aligning dev data using transcripts (needed for LM dev set)"
  echo --------------------------------------------------------------------

  steps/align_fmllr.sh --retry-beam 80 \
    --boost-silence $boost_sil --nj ${dev2h_nj} --cmd "$train_cmd" \
    data/dev2h data/lang exp/tri5 exp/tri5_dev2h_ali

  touch exp/tri5_dev2h_ali/.done
fi

if [ ! -f $target/data/dev2h/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Using the dev-data alignments to get syllable-level pronunciations"
  echo --------------------------------------------------------------------
  local/get_syllable_text.sh data/dev2h data/lang $target/data/lang_nopos \
    $target/data/local/w2s/L.fst \
     exp/tri5_dev2h_ali exp/tri5_align_dev2h_syllables $target/data/dev2h
  touch $target/data/dev2h/.done
fi


if [ ! -d $target/data/dev10h ]; then
  echo ---------------------------------------------------------------------
  echo "Copying data directory $target/data/dev10h/"
  echo --------------------------------------------------------------------
    ## Note: the "text" in this directory is not really correct, it is word-based
    ## not syllable based.  This doesn't really matter as we won't ever use that.
  cp -rT data/dev10h $target/data/dev10h
  rm -rf $target/data/dev10h/split*
  rm -rf $target/data/dev10h/kws
fi

for n in 2 10; do
  if [ -d data/dev${n}h/kws ] &&  [ ! -d $target/data/dev${n}h/kws/.done ]; then
    echo ---------------------------------------------------------------------
    echo "Preparing kws directory in $target/data/dev${n}h/kws"
    echo --------------------------------------------------------------------
    cp -rT data/dev${n}h/kws $target/data/dev${n}h/kws # first copy the old dir, then
      # run a script for syllable-specific setup.

    perl -ape 's/(\S+\s+)\S+\s+(.+)/$1$2/;' \
      <$target/data/local/w2s/lexiconp.txt >$target/data/local/w2s/lexicon.txt

    local/kws_data_prep_syllables.sh --silence-word SIL \
       $target/data/lang $target/data/dev${n}h $target/data/local/w2s/lexicon.txt \
       $target/data/dev${n}h/kws 
  fi
done

if [ ! -f $target/data/srilm/lm.gz ]; then
  echo ---------------------------------------------------------------------
  echo "Training LM in $target/data/srilm"
  echo --------------------------------------------------------------------
  
  local/train_lms_srilm.sh --dev-text $target/data/dev2h/text --train-text $target/data/train/text \
    --words-file $target/data/lang/words.txt $target/data $target/data/srilm
fi


if [ ! -f $target/data/lang/G.fst ]; then
  echo ---------------------------------------------------------------------
  echo "Creating $target/data/lang/G.fst"
  echo --------------------------------------------------------------------
  local/arpa2G_syllables.sh $target/data/srilm/lm.gz $target/data/lang $target/data/lang
fi



exit 0;


# At this point you go to the directory that it created, and run run-1-main.sh


