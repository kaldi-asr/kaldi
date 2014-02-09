#!/bin/bash

# Copyright 2014  Johns Hopkins University (authors: Daniel Povey, Yenda Trmal)
#           2014  Guoguo Chen
# Apache 2.0.

stage=-5
g2p_iters=5
num_prons=1000000 # number of prons to generate.
num_sent_gen=300000 # number of sents per job to generate.
                    # this times nj should exceed num_prons by
                    # a factor of at least several.
nj=40 # number of jobs to use for generation.
encoding='utf-8' # option for g2p; leave this as it is.
# the following two options are used in g2p generation.
var_counts=3  #Generate up to N variants in g2p
var_mass=0.8  #Generate enough variants to produce 80 % of the prob mass
min_phones=3  # minimum number of phones we allow in generated words
              # (very short generated words could contribute to graph blowup,
              #  and might hurt the decoding accuracy also).
skip_done=false # if true, allows us to skip over done g2p stages.
cmd=run.pl
cleanup=true

. utils/parse_options.sh
. path.sh

if [ $# -ne 2 ]; then
  echo "$0: usage: extend_lexicon.sh [options] <lexicon-in> <working-dir>"
  echo " e.g.: $0 data/local/lexicon_orig.txt data/local/extend/"
  echo "Will create in <working-dir> the files lexiconp.txt and oov2prob"
  echo "where lexiconp.txt is an extended lexicon with pronunciation"
  echo "probabilities, and oov2prob has lines <word> <prob> which divide"
  echo "the OOV probability mass among the introduced OOV words."
  echo "Important options:"
  echo " --cmd  <cmd-string>               # how to run jobs, default run.pl"
  echo " --num-prons  <n>                  # how many prons to generate, default 1000000"
  exit 1;
fi


input_lexicon=$1
toplevel_dir=$2 # e.g. data/local/extend

dir=$2/tmp  # most of our work happens in this "tmp" directory.

mkdir -p $dir

if [ ! -s $input_lexicon ]; then
  echo "$0: expected input lexicon $input_lexicon to exist";
fi

cp $input_lexicon $toplevel_dir/input_lexicon.txt  # just to have a record of what we started with.

loc=`which ngram-count`;
if [ -z $loc ]; then
  if uname -a | grep 64 >/dev/null; then # some kind of 64 bit...
    sdir=`pwd`/../../../tools/srilm/bin/i686-m64 
  else
    sdir=`pwd`/../../../tools/srilm/bin/i686
  fi
  if [ -f $sdir/ngram-count ]; then
    echo Using SRILM tools from $sdir
    export PATH=$PATH:$sdir
  else
    echo You appear to not have SRILM tools installed, either on your path,
    echo or installed in $sdir.  See tools/install_srilm.sh for installation
    echo instructions.
    exit 1
  fi
fi


if ! which g2p.py >&/dev/null; then
  if [ ! -d $KALDI_ROOT/tools/sequitur ]; then
    echo "Sequitur was not found !"
    echo "Go to $KALDI/tools and execute extras/install_sequitur.sh"
  else
    echo "Problems running sequitur.  Check that your path.sh is putting it on the path."
    echo "e.g. that it is sourcing KALDI_ROOT/tools/env.sh and that that env.sh file exists"
  fi
  exit 1;
fi

if ! which g2p.py >/dev/null ; then
  exit 1
fi


if [ $stage -le -5 ]; then
  # Map the phones to a more unambiguous representation so that when we
  # concatenate the letters of them, we won't lose information.  This will
  # also make g2p's life easier because each phone goes to a single letter,
  # which g2p will treat as a single symbol (remember, g2p is designed
  # to produce graphemes, so the tokens it produces are letters).

  cat $toplevel_dir/input_lexicon.txt | \
   awk '{for(n=2;n<=NF;n++) seen[$n]=1;} END{for (key in seen) print key;}' >$dir/phonelist

   cat $dir/phonelist | perl -e ' @ids = ("a".."z", "A".."Z");
  @map = (); while(<>) {
  chomp;  $output = "$_ ";
  @col = split("_");
  # Loop over different positions.
  for ($p = 0; $p < @col; $p++) {
    # New position that has not been assigned a hash.
    if (@map <= $p) {  push(@map, {});   }
    # Assign map for each position.
    if (!defined($map[$p]->{$col[$p]})) {
      if (@ids == 0) { # We have used all the ids... die here.
        die "Used up all the un-mapped ids\n";
      }
      $map[$p]->{$col[$p]} = shift @ids;
    }
    $output .= "$map[$p]->{$col[$p]}";
  }
  print "$output\n"; }' > $dir/phone_map
  cat $dir/phone_map | awk '{print $2, $1}' > $dir/phone_map.reverse

  cat $toplevel_dir/input_lexicon.txt | \
    local/apply_map_tab_preserving.pl -f 2- $dir/phone_map > $dir/lexicon_in.txt
fi


if [ $stage -le -4 ]; then
  cat $dir/lexicon_in.txt | perl -ane 'if (! m/^\<\S+\>\s/) { print; } ' > $dir/lexicon_in_nosil.txt

  cat $dir/lexicon_in.txt | perl -ane 's/^(\S+\s+)/${1}1.0\t/;print;' > $dir/lexiconp_in.txt
fi




if [ $stage -le -3 ]; then
  # Each syllable will be given a "word" representation; we join the phones using comma ","
  perl -e 'while(<STDIN>) { s/^\S+\s*//; s/ /,/g; print }'   <$dir/lexicon_in_nosil.txt >$dir/syllable_text.txt

  echo "$0: using SRILM to train syllable LM"

  ngram-count -lm $dir/3gram.kn022.gz -kndiscount1 -gt1min 0 -kndiscount2 -gt2min 2 -kndiscount3 -gt3min 2 -order 3 -text $dir/syllable_text.txt -sort

  rm $dir/lm.gz 2>/dev/null
  ln -s 3gram.kn022.gz $dir/lm.gz
fi


ngram=$(which ngram)

if [ $stage -le -2 ]; then
  mkdir -p $dir/log
  echo "$0: generating words from the syllable LM"

  $cmd JOB=1:$nj $dir/log/gen.JOB.log \
    $ngram -lm $dir/lm.gz -gen $num_sent_gen -seed JOB \| \
      sort -u \> $dir/sents.JOB || exit 1;
fi

if [ $stage -le -1 ]; then
  echo "$0: computing probs for the generated sentences"
  rm $dir/probs.* 2>/dev/null

  echo '#!/usr/bin/perl
while(1) { 
 $sent = <>; $line=<>; if ($line !~ m/sentences/) { $sent =~ m/^file/ || die "Bad sent $sent"; exit(0); }
 $line = <>; if ($line !~ m/logprob= (\S+)/) { die "Bad line $line"; } print "$1 $sent"; 
 $line = <>; $line eq "\n" || die "expected blank line"; }' >$dir/temp.pl
  chmod +x $dir/temp.pl

  $cmd JOB=1:$nj $dir/log/compute_prob.JOB.log \
    $ngram -debug 1 -lm $dir/lm.gz -ppl $dir/sents.JOB  \| $dir/temp.pl \| sort -gr \> $dir/probs.JOB || exit 1;

  if $cleanup; then 
    rm $dir/sents.*; 
  fi
  sort -m -gr $dir/probs.* | uniq | head -n $num_prons > $dir/probs
  if $cleanup; then 
    rm $dir/probs.*; 
  fi

  mass=$(cat $dir/probs | awk '{x += exp($1 * log(10));} END{print x}')

  echo "$0: total probability mass in generated words is $mass"
  echo " this should ideally be close to 1 (although we lose a little due to the"
  echo " empty sentence).  You can get closer by increasing --num-sent-gen and/or"
  echo " --nj"

  nl=$(cat $dir/probs | wc -l)
  if [ $nl -lt $num_prons ]; then
    echo "$0: Number of generated lines $nl is less than number of requested words $num_prons:"
    echo "  please run with larger --nj, currently $nj "
    exit 1;
  fi

fi


# Next we train a reverse g2p, which is really p2g.  Suppose a line in the lexicon is
# sugar  s uh  g ax r
# The basic idea is that we'd transform it to the following in reverse_lex.sh
# suhgaxr s u g a r
# We may lose a little information by doing this, though, because the segmentation
# into phonemes may be ambiguous.  So we create a mapping from the original phonemes
# and tags to letters of the alphabet.  Note: tags are things like s_3 for a phone: here
# s is the phone and _3 is the tag. 


if [ $stage -le 0 ]; then
  cat $dir/lexicon_in_nosil.txt | \
    perl -ane '@A = split; $w = shift @A; print join("", @A) . "\t" . join(" ", split("", $w)) . "\n";' > $dir/lexicon_reverse.txt

  echo "$0: Training the G2P model (iter 0)"
  if ! $skip_done || [ ! -f $dir/p2g.model.0 ]; then
    $cmd $dir/log/g2p.0.log \
      g2p.py -S --encoding $encoding --train $dir/lexicon_reverse.txt --devel 5% --write-model $dir/p2g.model.0 || exit 1;
  else
    echo "$0: $dir/p2g.model.0 already exists: skipping it since --skip-done is true"
  fi
fi

for i in `seq 0 $(($g2p_iters-2))`; do
  if [ $stage -le $[i+1] ]; then
    if ! $skip_done || [ ! -f $dir/p2g.model.$[$i+1] ]; then
      echo "$0: Training the G2P model (iter $[$i + 1] )"
      $cmd $dir/log/g2p.$[$i+1].log \
        g2p.py -S --encoding $encoding --model $dir/p2g.model.$i --ramp-up \
        --train $dir/lexicon_reverse.txt --devel 5% \
        --write-model $dir/p2g.model.$(($i+1))
    else
      ii=$[$i+1];
      echo "$0: $dir/p2g.model.$ii already exists: skipping it since --skip-done is true"
    fi
  fi
  rm $dir/p2g.model.final
  ln -s p2g.model.$(($i+1)) $dir/p2g.model.final
done



if [ $stage -le $g2p_iters ]; then
  # get the word-list to apply g2p to; each one is just a sequence
  # of phones, formed by appending the syllables in the "generated sentences"
  # (really generated syllable-sequences) in $dir/probs, and removing the
  # separator.

  cat $dir/probs | head -n $num_prons | awk '{$1=""; print $0}' | \
     sed "s/,//g;s/ //g;" | sort | uniq > $dir/fake_word_list.txt

  echo "$0: Applying the G2P model to wordlist $wordlist"

  $cmd JOB=1:$nj $dir/log/apply_p2g.JOB.log \
    split -n l/JOB/$nj $dir/fake_word_list.txt \| \
    g2p.py -V $var_mass --variants-number $var_counts --encoding $encoding \
      --model $dir/p2g.model.final --apply - \
    \> $dir/p2g_output.JOB || exit 1;
  cat $dir/p2g_output.* > $dir/p2g_output
  rm $dir/p2g_output.*
fi

if [ $stage -le $[$g2p_iters+1] ]; then

  # the NF >= 4 is about pruning out any empty spellings, that would
  # produce an empty word.
  # pron2spelling contains lines like ak>a 0.957937 aka
  cat $dir/p2g_output | \
     awk '{if (NF >= 4) {printf("%s %s ", $1, $3); for (n=4;n<=NF;n++) {printf("%s", $n);} printf("\n"); }}' | \
      sort | uniq > $dir/pron2spelling

  # Now remove from pron2spelling, any words that appear in $dir/lexiconp_in.txt 
  # (this also contains the excluded words like <unk>).
  cat $dir/pron2spelling | \
   perl -e 'open(F, $ARGV[0]) || die "opening $ARGV[0]"; while(<F>) { @A=split; $seen_word{$A[0]}=1; } 
        while(<STDIN>) { @A=split; if (! $seen_word{$A[2]}) { print; }} ' $dir/lexiconp_in.txt > $dir/pron2spelling.excluded
  # $dir/pron2spelling.excluded contains lines like
  #ab<a 0.957535 aba

  n1=$(cat $dir/pron2spelling | wc -l)
  n2=$(cat $dir/pron2spelling.excluded | wc -l)
  echo "$0: Removing seen words from the list of (pronunciation,spelling) combinations "
  echo " changed length of list from $n1 to $n2"


  # Now combine probs and pron2spelling to create a file words_and_prons with entries 
  # <word> <prob> syllable1 syllable2 ...
  # e.g.
  # Kuku 0.000002642 k>&u k>&u
    
  cat $dir/probs | \
     perl -e ' while(<STDIN>){ @A = split; $prob = shift @A; $pron=join("", @A);
         $pron =~ tr/,//d; print "$pron $_"; } '> $dir/probs.with_pron
   # $dir/probs.with_pron contains lines like the following:
   # ak>a -2.43244 a &k>&a
   # This is so we can get the pronunciation in the same form that we put it in, for
   # the p2g training, for easier comparison with the lines in $dir/pron2spelling.excluded

   perl -e ' ($p2s, $probs_with_pron) = @ARGV; 
     open(P2S, "<$p2s" || die);  open(PROBS, "<$probs_with_pron")||die;
    while (<P2S>) {
      @A = split;
      ($pron,$pronprob,$spelling) = @A;
      if (!defined $prons{$pron}) { $prons{$pron} = [ ]; } # new anonymous array
      $ref = $prons{$pron};
      push @$ref, "$pronprob $spelling";
    }
    $log10 = log(10.0);
    while (<PROBS>) {
       @A = split;
       $pron = shift @A; # pron in same format as used by p2g model.
       $logprob = shift @A;
       $syllable_pron = join(" ", @A); # pron separated by syllable
       $p = exp($logprob * $log10);
       $ref = $prons{$pron};
       if (defined $ref) {
          foreach $str (@$ref) {
             @B = split(" ", $str);
             ($pronprob,$spelling) = @B;
             $pair_prob = $p * $pronprob;
             print "$spelling $pair_prob $syllable_pron\n";
          }
       }
    } ' $dir/pron2spelling.excluded $dir/probs.with_pron > $dir/lexicon.oov.raw

  # $dir/lexicon.oov.raw contains lines like:
  # ukuzi 0.000342399163717093 u &k>&u &z&i

  mass=$(cat $dir/lexicon.oov.raw | awk '{x+=$2;} END{print x}')
  echo "$0: Total probability mass of unseen words (before removing prons"
  echo " shorter than $min_phones phones) is $mass"

  # the next stage does 3 things: (1) it converts the pronunciations to be
  # tab-separated lists of syllables and removes the seprator ","; (2) it limits us
  # to prons containing at least $min_phones phones; and (3) it limits to the
  # most likely $num_prons pairs of (spelling, pron)
  perl -e ' while (<STDIN>) {
       @A = split;
       $spelling = shift @A;
       $prob = shift @A;
       for ($n = 0; $n < @A; $n++) { # replace separator in syllable with space.
          $A[$n] =~ tr/,/ /d; # replace the separator with space.
       }
       $final_pron = join("\t", @A);
       print "$spelling\t$prob\t$final_pron\n";
    } ' <$dir/lexicon.oov.raw | sort -k2,2 -gr | \
     awk -v min=$min_phones '{if(NF>=min+2){print;}}' | head -n $num_prons >$dir/lexicon.oov


  mass=$(cat $dir/lexicon.oov | awk '{x+=$2;} END{print x}')
  echo "$0: Total probability mass of unseen words (after removing prons"
  echo " shorter than $min_phones phones) is $mass."


  # $dir/lexicon.oov contains lines like the following:
  #  ngisa   0.00340513074018366    N g i     s a
  # where the multiple-spaces are actually tabs.

  # Now renormalize the probability to sum to one, decompose $dir/lexicon.oov
  # into two pieces: a lexicon $dir/lexiconp_oov.txt, which contains the
  # probabilities of different spellings of words (with the most likely one at
  # 1.0), and $dir/oov2prob which contains the probabilities of the words
  # (we'll use it later to adjust the LM).

  # the uniq here shouldn't be needed, actually. [relates to a bug in a previous
  # step that is now fixed.  This script relies on the fact that lexicon.oov
  # is sorted in reverse order of probability.
  cat $dir/lexicon.oov | awk -v mass=$mass 'BEGIN{OFS=FS="\t";} {$2 = $2/mass; print;}' | uniq | \
     perl -e ' ($lexiconp,$words_probs) = @ARGV;
     open(L, "|sort -u >$lexiconp") || die "opening lexicon $lexiconp";
     open(W, "|sort -u >$words_probs") || die "opening probs file $words_probs";
     while (<STDIN>) {
        @A = split("\t", $_);
        $word = shift @A; $prob = shift @A; $pron = join("\t", @A);
        if (!defined $maxprob{$word}) { # max prob is always the first.
           $maxprob{$word} = $prob;
           print W "$word $prob\n";
        }
        $pronprob = $prob / $maxprob{$word};
        $pronprob <= 1 || die "bad pronprob $pronprob\n";
        print L "$word\t$pronprob\t$pron";
      } close(L); close(W); # wait for sort to finish. ' \
        $dir/lexiconp_oov.txt $dir/oov2prob
    
   # lexiconp_oov.txt contains lines like:
   #leyanga	0.96471840417664	l 3	 j_" a_"	 N a
   #leyanga	1	l 3	 j_" a_"	 N g a

   # oov2prob looks like this:
   #-Uni 8.77716315938887e-07
   #Adlule 9.62418179264897e-08
   #Afuna 2.23048402109824e-06
fi
  
if [ $stage -le $[$g2p_iters+2] ]; then
  # put it to the output directory $localdir e.g. data/local/
  cat $dir/lexiconp_in.txt $dir/lexiconp_oov.txt | \
    local/apply_map_tab_preserving.pl -f 3- $dir/phone_map.reverse | sort -u > $working_dir/lexiconp.txt
  cp $dir/oov2prob $working_dir/oov2prob
fi

exit 0;

###BELOW HERE IS JUST COMMENTS ###########

#cat /export/babel/data/206-zulu/release-current/conversational/reference_materials/lexicon.sub-train.txt | \
for x in data/local/filtered_lexicon.txt data/local/lexiconp.txt; do 
cat /export/babel/data/206-zulu/release-current/conversational/reference_materials/lexicon.txt  | \
 perl -e '$lex = shift @ARGV; open(L, "<$lex")||die; while(<L>){ @A=split; $seen{$A[0]}=1;}
   while(<STDIN>) { @A=split; $word=$A[0]; $tot++; if(defined $seen{$word}) { $invoc++; }}
   $oov_rate = 100.0 * (1.0 - ($invoc / $tot)); printf("Seen $invoc out of $tot tokens; OOV rate is %.2f\n", $oov_rate);  ' $x
done
# OOV rate measured on the words in the FullLP lexicon.
#Seen 13675 out of 60613 tokens; OOV rate is 77.44
#Seen 26936 out of 60613 tokens; OOV rate is 55.56

for x in data/local/filtered_lexicon.txt data/local/lexiconp.txt; do 
cat data/dev10h/text | awk '{for(n=2;n<=NF;n++) { print $n; }}' | \
 perl -e '$lex = shift @ARGV; open(L, "<$lex")||die; while(<L>){ @A=split; $seen{$A[0]}=1;}
   while(<STDIN>) { @A=split; $word=$A[0]; $tot++; if(defined $seen{$word}) { $invoc++; }}
   $oov_rate = 100.0 * (1.0 - ($invoc / $tot)); printf("Seen $invoc out of $tot tokens; OOV rate is %.2f\n", $oov_rate);  ' $x
done
# zulu limitedlp, dev10h:
# With the million-word lexicon we more than halve the per-token OOV rate of dev10h.
#Seen 44680 out of 66891 tokens; OOV rate is 33.20
#Seen 57095 out of 66891 tokens; OOV rate is 14.64
