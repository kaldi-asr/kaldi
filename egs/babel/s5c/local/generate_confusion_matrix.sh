#!/usr/bin/env bash
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
  echo " e.g.: local/prepare_confusions.sh --nj 32  exp/sgmm5/graph exp/sgmm5 exp/sgmm5_ali exp/sgmm5_denlats  exp/conf_matrix"
  echo ""
  echo "main options (for others, see top of script file)"
  echo "  --nj <nj>                                # number of parallel jobs"
  echo "  --cmd <cmd>                              # Command to run in parallel with"
  echo "  --acwt <value|default=0.1>               # Acoustic model weight. Value will be used for 1-best path decoding of the lattices"
  echo ""
  echo "Please note that the output confusion matrix will be phoneme-based"
  echo "and all the phone contexts  (singleton, intra, begin, end) or phoneme"
  echo "tags (such as tone or stress) will be collapsed into a single monophone"
  echo ""
  echo "The output format is line oriented."
  echo "Each line can have one of these four formats (A, B being different phones, <eps> special symbol"
  echo "  A A count        #Number of hits, i.e. correctly determined phones"
  echo "  A B count        #Number of substitutions of A with B   "
  echo "  A <eps> count    #Number of deletions"
  echo "  <eps> A count    #Number of insertions"
  exit 1;
fi

set -u
set -e
set -o pipefail

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
[ ! $nj -le $ali_nj ] && echo "Number of jobs is too high (max is $ali_nj)." && nj=$ali_nj

mkdir -p $wdir/log

cat $data/phones.txt | sed 's/_[B|E|I|S]//g' |\
  sed 's/_[%|"]//g' | sed 's/_[0-9]\+//g' > $wdir/phones.txt

echo "Converting alignments to phone sequences..."
$cmd JOB=1:$nj $wdir/log/ali_to_phones.JOB.log \
  align-text\
    ark:\<\( \
      ali-to-phones  $model ark:"gunzip -c $alidir/ali.JOB.gz|" ark,t:- \|\
      int2sym.pl -f 2- $wdir/phones.txt - \) \
    ark:\<\( \
      lattice-to-phone-lattice $model ark:"gunzip -c $latdir/lat.JOB.gz|"  ark:- \| \
      lattice-best-path --acoustic-scale=$acwt  ark:- ark,t:- ark:/dev/null \| \
      int2sym.pl -f 2- $wdir/phones.txt - \) \
    ark:$wdir/confusions.JOB.txt

confusion_files=""
for i in `seq 1 $nj` ; do
  confusion_files="$confusion_files $wdir/confusions.$i.txt"
done

echo "Converting statistics..."
cat $confusion_files | cut -f 2- -d ' ' | perl -ape 's/ *; */\n/g;'| sort | uniq -c | \
  grep -v -E '<oov>|<sss>|<vns>|SIL' | \
  perl -ane '
    die unless scalar @F == 3;
    print "$F[1] $F[2] $F[0]\n";
    ' > $wdir/confusions.txt

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

