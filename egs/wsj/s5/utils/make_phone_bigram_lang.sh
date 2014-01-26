#!/bin/bash

# Apache 2.0.  Copyright 2012, Johns Hopkins University (author: Daniel Povey)

# This script creates a "lang" directory of the "testing" type (including G.fst)
# given an existing "alignment" directory and an existing "lang" directory.
# The directory contains only single-phone words, and a bigram language model that
# is built without smoothing, on top of single phones.  The point of no smoothing
# is to limit the number of transitions, so we can decode reasonably fast, and the
# graph won't blow up.  This is probably going to be most useful for things like
# language-id.


# We might later have options here; if not, I'llr emove this.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;


if [ $# != 3 ]; then
  echo "Usage: utils/make_phone_bigram_lang.sh [options] <lang-dir> <ali-dir> <output-lang-dir>"
  echo "e.g.: utils/make_phone_bigram_lang.sh data/lang exp/tri3b_ali data/lang_phone_bg"
  exit 1;
fi

lang=$1
alidir=$2
lang_out=$3

for f in $lang/phones.txt $alidir/ali.1.gz; do
  [ ! -f $f ] && echo "Expected file $f to exist" && exit 1;
done

mkdir -p $lang_out || exit 1;

grep -v '#' $lang/phones.txt >  $lang_out/phones.txt # no disambig symbols
      # needed; G and L . G will be deterministic.
cp $lang/topo $lang_out
rm -r $lang_out/phones 2>/dev/null
cp -r $lang/phones/ $lang_out/
rm $lang_out/phones/word_boundary.* 2>/dev/null # these would
  # no longer be valid.
# List of disambig symbols will be empty: not needed, since G.fst and L.fst * G.fst
# are determinizable without any.
echo -n > $lang_out/phones/disambig.txt
echo -n > $lang_out/phones/disambig.int
echo -n > $lang_out/phones/disambig.csl

# Let OOV symbol be the first phone.  This is arbitrary, it's just
# so that validate_lang.pl succeeds.  We should never actually use
# this.
oov_sym=$(tail -n +2 $lang_out/phones.txt | head -n 1 | awk '{print $1}')
oov_int=$(tail -n +2 $lang_out/phones.txt | head -n 1 | awk '{print $2}')
echo $oov_sym > $lang_out/oov.txt
echo $oov_int > $lang_out/oov.int


# Get phone-level transcripts of training data and create a
# language model.
ali-to-phones $alidir/final.mdl "ark:gunzip -c $alidir/ali.*.gz|" ark,t:- | \
  perl -e 'while(<>) {
    @A = split(" ", $_);
    shift @A; # Remove the utterance-id.
    foreach $p ( @A ) { $phones{$p} = 1; } # assoc. array of phones.
    unshift @A, "<s>";
    push @A, "</s>";
    for ($n = 0; $n+1 < @A; $n++) {
      $p = $A[$n]; $q = $A[$n+1];
      $count{$p,$q}++;
      $histcount{$p}++;
    }
  }
  @phones = keys %phones;
  unshift @phones, "<s>";
  # @phones is now all real phones, plus <s>.
  for ($n = 0; $n < @phones; $n++) {
    $phn2state{$phones[$n]} = $n;
  }
  foreach $p (@phones) {
    $src = $phn2state{$p};
    $hist = $histcount{$p};
    $hist > 0 || die;    
    foreach $q (@phones) {
      $c = $count{$p,$q};
      if (defined $c) {
        $cost = -log($c / $hist); # cost on FST arc.
        $dest = $phn2state{$q};
        print "$src $dest $q $cost\n";  # Note: q is actually numeric.
      }
    }
    $c = $count{$p,"</s>"};
    if (defined $c) {
      $cost = -log($c / $hist); # cost on FST arc.      
      print "$src $cost\n"; # final-prob.
    }
  } ' | fstcompile --acceptor=true > $lang_out/G.fst

# symbols for phones and words are the same.
# Neither has disambig symbols.
cp $lang_out/phones.txt $lang_out/words.txt
  
grep -v '<eps>' $lang_out/phones.txt | awk '{printf("0 0 %s %s\n", $2, $2);} END{print("0 0.0");}' | \
   fstcompile  > $lang_out/L.fst

# L and L_disambig are the same.
cp $lang_out/L.fst $lang_out/L_disambig.fst

utils/validate_lang.pl $lang_out || exit 1;
echo "$0: ignore warnings above from validate_lang.pl (these are expected)"

