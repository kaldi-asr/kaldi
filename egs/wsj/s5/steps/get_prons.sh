#!/bin/bash
# Copyright  2014  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0

# Begin configuration section.  
cmd=run.pl
stage=1
lmwt=10
# End configuration options.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "usage: $0 <data-dir> <lang-dir> <dir>"
   echo "e.g.:  $0 data/train data/lang exp/tri3"
   echo "or:  $0 data/train data/lang exp/tri3/decode_dev"
   echo "This script writes files prons.*.gz in the directory provided, which must"
   echo "contain alignments (ali.*.gz) or lattices (lat.*.gz).  These files are as"
   echo "output by nbest-to-prons (see its usage message)."
   echo "Main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --lmwt <lm-weight>                               # scale for LM, only applicable"
   echo "                                                   # for lattice input (default: 10)"
   exit 1;
fi

data=$1
lang=$2
dir=$3

for f in $data/utt2spk $lang/words.txt $dir/num_jobs; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1;
done

nj=$(cat $dir/num_jobs) || exit 1;
sdata=$data/split$nj
oov=`cat $lang/oov.int` || exit 1;
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;


if [ -f $dir/final.mdl ]; then
  mdl=$dir/final.mdl
else
  if [ -f $dir/../final.mdl ]; then
    mdl=$dir/../final.mdl  # e.g. decoding directories.
  else
    echo "$0: expected $dir/final.mdl or $dir/../final.mdl to exist."
    exit 1;
  fi
fi

if [ -f $lang/phones/word_boundary.int ]; then
  align_words_cmd="lattice-align-words $lang/phones/word_boundary.int $mdl ark:- ark:-"
else
  if [ ! -f $lang/phones/align_lexicon.int ]; then
    echo "$0: expected either $lang/phones/word_boundary.int or $lang/phones/align_lexicon.int to exist."
    exit 1;
  fi
  align_words_cmd="lattice-align-words-lexicon $lang/phones/align_lexicon.int $mdl ark:- ark:-"
fi

if [ -f $dir/ali.1.gz ]; then
  echo "$0: $dir/ali.1.gz exists, so starting from alignments."
  
  if [ $stage -le 1 ]; then
    rm $dir/prons.*.gz 2>/dev/null
    $cmd JOB=1:$nj $dir/log/nbest_to_prons.JOB.log \
      linear-to-nbest "ark:gunzip -c $dir/ali.JOB.gz|" \
      "ark:sym2int.pl --map-oov $oov -f 2- $lang/words.txt <$sdata/JOB/text |" \
      "" "" ark:- \| $align_words_cmd \| \
      nbest-to-prons $mdl ark:- "|gzip -c >$dir/prons.JOB.gz" || exit 1;
  fi
else
  if [ ! -f $dir/lat.1.gz ]; then
    echo "$0: expected either $dir/ali.1.gz or $dir/lat.1.gz to exist."
    exit 1;
  fi
  echo "$0: $dir/lat.1.gz exists, so starting from lattices."

  if [ $stage -le 1 ]; then
    rm $dir/prons.*.gz 2>/dev/null
    $cmd JOB=1:$nj $dir/log/nbest_to_prons.JOB.log \
      lattice-1best --lm-scale=$lmwt "ark:gunzip -c $dir/lat.JOB.gz|" ark:- \| \
      $align_words_cmd \| \
      nbest-to-prons $mdl ark:- "|gzip -c >$dir/prons.JOB.gz" || exit 1;
  fi
fi


if [ $stage -le 2 ]; then
  gunzip -c $dir/prons.*.gz | \
    awk '{ $1=""; $2=""; $3=""; count[$0]++; } END{for (k in count) { print count[k], k; }}' > $dir/pron_counts.int || exit 1;
fi

if [ $stage -le 3 ]; then
  cat $dir/pron_counts.int | utils/int2sym.pl -f 2 $lang/words.txt | \
    utils/int2sym.pl -f 3- $lang/phones.txt | sort -nr > $dir/pron_counts.txt
fi

if [ $stage -le 4 ]; then
  if [ -f $lang/phones/word_boundary.int ]; then
    # remove the _B, _I, _S, _E markers from phones; this is often convenient
    # if we want to go back to a word-position-independent source lexicon.
    cat $dir/pron_counts.txt | perl -ane '@A = split(" ", $_);
     for ($n=2;$n<@A;$n++) { $A[$n] =~ s/_[BISE]$//; } print join(" ", @A) . "\n"; ' >$dir/pron_counts_nowb.txt
  fi
fi

echo "$0: done writing prons to $dir/prons.*.gz, counts in $dir/pron_counts.{int,txt}"
if [ -f $lang/phones/word_boundary.int ]; then
  echo "$0: ... and also in $dir/pron_counts_nowb.txt"
fi

exit 0;
