#!/bin/bash
# Copyright 2014  Johns Hopkins University (Author: Yenda Trmal)
# Apache 2.0

# Begin configuration section.  
nj=4
cmd=run.pl
acwt=0.1
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 5 ]; then
  echo "Usage $0 [options] <data-dir> <model-dir> <ali-dir> <decode-dir> <out-dir>"
  exit 1;
fi

set -u
set -e

data=$1; shift
modeldir=$1; shift
alidir=$1; shift
latdir=$1; shift
wdir=$1; shift

model=$modeldir/final.mdl
[ ! -f $model ] && echo "File $model does not exist!" && exit 1
phones=$data/phones.txt
[ ! -f $phones ] && echo "File $phones does not exist!" && exit 1

! ali_nj=`cat $alidir/num_jobs` && echo "Could not open the file $alidir/num_jobs" && exit 1
! lat_nj=`cat $latdir/num_jobs` && echo "Could not open the file $latdir/num_jobs" && exit 1
if [ $ali_nj -ne $lat_nj ] ; then
  echo "Alignments num_jobs and lattices num_jobs mismatch!"
  exit 1
fi
[ ! $nj -le $ali_nj ] && echo "Number of jobs is too high (max is $ali_nj)." && exit 1

mkdir -p $wdir/log

cat $data/phones.txt | sed "s/^\([^ _\t][^ _\t]*\)_[^ \t][^ \t]* /\1 /g" > $wdir/phones.txt

echo "Converting alignments to phone sequences..."
$cmd JOB=1:$nj $wdir/log/ali_to_phones.JOB.log \
  compute-wer --text --mode=all\
    ark:\<\( \
      ali-to-phones  $model ark:"gunzip -c $alidir/ali.JOB.gz|" ark,t:- \|\
      int2sym.pl -f 2- $wdir/phones.txt - \) \
    ark:\<\( \
      lattice-to-phone-lattice $model ark:"gunzip -c $latdir/lat.JOB.gz|"  ark:- \| \
      lattice-best-path --acoustic-scale=$acwt  ark:- ark,t:- ark:/dev/null \| \
      int2sym.pl -f 2- $wdir/phones.txt - \) \
    $wdir/confusions.JOB.txt

confusion_files=""
for i in `seq 1 $nj` ; do
  confusion_files="$confusion_files $wdir/confusions.$i.txt"
done

echo "Converting statistics..."
cat $confusion_files | sort | uniq -c | grep -v -E '<oov>|<sss>|<vns>|SIL' | \
  perl -ane '
    if ($F[1] eq "correct") {
      die "Unknown format " . join(" ", @F) . "\n" if ($#F != 2);
      print "$F[2] $F[2] $F[0]\n";
    } elsif ($F[1] eq "deletion" ) {
      die "Unknown format " . join(" ", @F) . "\n" if ($#F != 2);
      print "$F[2] <eps> $F[0]\n";
    } elsif ($F[1] eq "insertion") {
      die "Unknown format " . join(" ", @F) . "\n" if ($#F != 2);
      print "<eps> $F[2] $F[0]\n";
    } elsif ($F[1] eq "substitution") {
      die "Unknown format " . join(" ", @F) . "\n" if ($#F != 3);
      print "$F[2] $F[3] $F[0]\n";
    } else {
      die "Unknown line " . join(" ", @F). "\n";
    }' > $wdir/confusions.txt

exit 0
#-echo "Converting alignments to phone sequences..."
#-$cmd JOB=1:$nj $wdir/log/ali_to_phones.JOB.log \
#-  ali-to-phones  $model ark:"gunzip -c $alidir/ali.JOB.gz|" ark,t:- \|\
#-  int2sym.pl -f 2- $wdir/phones.txt - \> $wdir/ali.JOB.txt
#-
#-echo "Converting lattices to phone sequences..."
#-$cmd JOB=1:$nj $wdir/log/lat_to_phones.JOB.log \
#-  lattice-to-phone-lattice $model ark:"gunzip -c $latdir/lat.JOB.gz|"  ark:- \| \
#-  lattice-best-path --acoustic-scale=$acwt  ark:- ark,t:- ark:/dev/null \| \
#-  int2sym.pl -f 2- $wdir/phones.txt - \> $wdir/lat.JOB.txt

