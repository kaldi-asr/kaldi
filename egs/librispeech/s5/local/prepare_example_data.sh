#!/usr/bin/env bash

# Copyright 2015  Guoguo Chen
# Apache 2.0
#
# This script creates example data directory which will be used to illustrate
# how to use the models from openslr.org


num_examples=5

echo "$0 $@"  # Print the command line for logging
[ -f path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

if [ $# -ne 2 ]; then
   echo "Usage: $0 <original-data-dir> <example-data-dir>"
   echo " e.g.: $0 data/test_clean/ data/test_clean_example/"
   exit 1;
fi

srcdatadir=$1
datadir=$2

for f in spk2gender spk2utt text utt2spk wav.scp; do
  [ ! -f $srcdatadir/$f ] && echo "$0: no such file $srcdatadir/$f" && exit 1;
done

mkdir -p $datadir
mkdir -p $datadir/example_wav

head -5 $srcdatadir/text > $datadir/text

utils/filter_scp.pl $datadir/text $srcdatadir/wav.scp > $datadir/wav.scp.old

wav_list=`cat $datadir/wav.scp.old | awk 'BEGIN{ORS=" ";} {print $6;}'`
for f in $wav_list; do
  cp -f $f $datadir/example_wav/ || exit 1;
done

cat $datadir/wav.scp.old | perl -e '
  while (<STDIN>) {
    chomp;
    @col = split(" ", $_);
    @col == 7 || die "Bad line in wav file: $_\n";
    $col[5] =~ s"^.*/""g;
    $col[5] = "'$datadir'" . "/example_wav/" . $col[5];
    for ($x = 0; $x < 6; $x += 1) {
      print "$col[$x] ";
    }
    print "$col[6]\n";
  }' > $datadir/wav.scp
rm -rf $datadir/wav.scp.old

cp -f $srcdatadir/spk2gender $datadir/
cp -f $srcdatadir/spk2utt $datadir/
cp -f $srcdatadir/utt2spk $datadir/

utils/fix_data_dir.sh $datadir
