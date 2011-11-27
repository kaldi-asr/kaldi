#!/bin/bash

# Remove excess utterances once they appear  more than a specified
# number of times with the same transcription, in a data set.
# E.g. useful for removing excess "uh-huh" from training.

if [ $# != 3 ]; then
  echo "Usage: remove_dup_utts.sh max-count src-data-dir dest-data-dir"
  exit 1;
fi

maxcount=$1
srcdir=$2
destdir=$3
mkdir -p $destdir

[ ! -f $srcdir/text ] && echo "Invalid input directory $srcdir" && exit 1;


cat $srcdir/text | \
  perl -e '
  $maxcount = shift @ARGV; 
  @all = ();
   $p1 = 103349; $p2 = 71147; $k = 0;
   sub random { # our own random number generator: predictable.
     $k = ($k + $p1) % $p2;
     return ($k / $p2);
  }
  while(<>) {
    push @all, $_;
    @A = split(" ", $_);
    shift @A;
    $text = join(" ", @A);
    $count{$text} ++;
  }
  foreach $line (@all) {
    @A = split(" ", $line);
    shift @A;
    $text = join(" ", @A);
    $n = $count{$text};
    if ($n < $maxcount || random() < ($maxcount / $n)) {
      print $line;
    }
  }'  $maxcount >$destdir/text 

echo "Reduced number of utterances from `cat $srcdir/text | wc -l` to `cat $destdir/text | wc -l`"

scripts/filter_scp.pl $destdir/text <$srcdir/feats.scp >$destdir/feats.scp
scripts/filter_scp.pl $destdir/text <$srcdir/utt2spk >$destdir/utt2spk
scripts/utt2spk_to_spk2utt.pl < $destdir/utt2spk > $destdir/spk2utt
[ -f $srcdir/wav.scp ] && scripts/filter_scp.pl $destdir/feats.scp <$srcdir/wav.scp >$destdir/wav.scp
[ -f $srcdir/spk2gender ] && scripts/filter_scp.pl $destdir/spk2utt <$srcdir/spk2gender >$destdir/spk2gender
