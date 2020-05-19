#!/usr/bin/env bash

. ./path.sh

format=pdf # pdf svg
output=

. utils/parse_options.sh

if [ $# != 3 ]; then
   echo "usage: $0 [--format pdf|svg] [--output <path-to-output>] <utt-id> <lattice-ark> <word-list>"
   echo "e.g.:  $0 utt-0001 \"test/lat.*.gz\" tri1/graph/words.txt"
   exit 1;
fi

uttid=$1
lat=$2
words=$3

tmpdir=$(mktemp -d); trap "rm -r $tmpdir" EXIT # cleanup

gunzip -c $lat | lattice-to-fst ark:- ark,scp:$tmpdir/fst.ark,$tmpdir/fst.scp || exit 1
! grep "^$uttid " $tmpdir/fst.scp && echo "ERROR : Missing utterance '$uttid' from gzipped lattice ark '$lat'" && exit 1
fstcopy "scp:grep '^$uttid ' $tmpdir/fst.scp |" "scp:echo $uttid $tmpdir/$uttid.fst |" || exit 1
fstdraw --portrait=true --osymbols=$words $tmpdir/$uttid.fst | dot -T${format} > $tmpdir/$uttid.${format}

if [ ! -z $output ]; then
  cp $tmpdir/$uttid.${format} $output
fi

[ $format == "pdf" ] && evince $tmpdir/$uttid.pdf
[ $format == "svg" ] && eog $tmpdir/$uttid.svg

exit 0
