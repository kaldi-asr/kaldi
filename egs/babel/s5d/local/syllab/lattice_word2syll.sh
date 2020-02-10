#!/usr/bin/env bash
# Copyright (c) 2016, Johns Hopkins University ( Yenda Trmal <jtrmal@gmail.com> )
# License: Apache 2.0

# Begin configuration section.
cmd=run.pl
acwt=0.1
beam=8
# End configuration section
echo $0 "$@"
. ./utils/parse_options.sh

set -e -o pipefail
set -o nounset                              # Treat unset variables as an error

data=$1; shift;
ilang=$1; shift;
olang=$1; shift;
input=$1; shift
output=$1; shift

nj=$(cat $input/num_jobs)

mkdir -p $output/log


if [ -f $olang/lex.words2syllabs.fst ] ; then
  fstinvert $olang/lex.words2syllabs.fst | fstreverse | \
    fstminimize --allow_nondet | fstreverse > $output/L.fst

  $cmd JOB=1:$nj $output/log/convert.JOB.log \
    lattice-push --push-strings ark:"gunzip -c $input/lat.JOB.gz|" ark:- \| \
      lattice-lmrescore --lm-scale=-1.0 ark:- "fstproject --project_output=true $ilang/G.fst|" ark:- \| \
      lattice-compose ark:- $output/L.fst  ark:- \| \
      lattice-determinize-pruned --beam=8 --acoustic-scale=0.1 ark:-  ark:- \| \
      lattice-minimize ark:- "ark:|gzip -c > $output/lat.JOB.gz"
      #lattice-minimize ark:- ark:- \| \
      #lattice-lmrescore --lm-scale=1.0 ark:- "fstproject --project_output=true $olang/G.fst|" "ark:|gzip -c > $output/lat.JOB.gz"
else
  #for phonemes.... (IIRC)
  fstreverse $olang/L.fst | fstminimize | fstreverse > $output/L.fst
  $cmd JOB=1:$nj $output/log/convert.JOB.log \
    lattice-push --push-strings ark:"gunzip -c $input/lat.JOB.gz|" ark:- \| \
      lattice-lmrescore --lm-scale=-1.0 ark:- "fstproject --project_output=true $ilang/G.fst|" ark:- \| \
      lattice-align-words $ilang/phones/word_boundary.int $input/../final.mdl ark:- ark:-  \| \
      lattice-to-phone-lattice --replace-words $input/../final.mdl ark:- ark:- \| \
      lattice-align-phones $input/../final.mdl  ark:- ark:- \| \
      lattice-compose ark:- $output/L.fst ark:- \|\
      lattice-determinize-pruned --beam=$beam --acoustic-scale=$acwt ark:-  ark:-\| \
      lattice-minimize ark:- "ark:|gzip -c > $output/lat.JOB.gz"
      #lattice-lmrescore --lm-scale=1.0 ark:- "fstproject --project_output=true $olang/G.fst|" ark:"|gzip -c > $output/lat.JOB.gz"
fi

  #lattice-1best ark:- ark:-| nbest-to-linear ark:- ark:/dev/null ark,t:- \
  #utils/int2sym.pl -f 2- $olang/words.txt | head
cp $input/num_jobs $output/num_jobs

