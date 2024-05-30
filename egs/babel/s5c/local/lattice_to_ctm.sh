#!/usr/bin/env bash
# Copyright Johns Hopkins University (Author: Daniel Povey) 2012.  Apache 2.0.

# begin configuration section.
cmd=run.pl
stage=0
decode_mbr=true
beam=5
word_ins_penalty=0.5
min_lmwt=7
max_lmwt=17
model=

#end configuration section.

#debugging stuff
echo $0 $@

[ -f ./path.sh ] && . ./path.sh
[ -f ./cmd.sh ]  && . ./cmd.sh
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: $0 [options] <dataDir> <langDir|graphDir> <decodeDir>" && exit;
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --stage (0|1)                 # (createCTM | filterCTM )."
  exit 1;
fi

data=$1
lang=$2 # Note: may be graph directory not lang directory, but has the necessary stuff copied.
dir=$3

if [ -z "$model" ] ; then
  model=`dirname $dir`/final.mdl # Relative path does not work in some cases
  #model=$dir/../final.mdl # assume model one level up from decoding dir.
  #[ ! -f $model ] && model=`(set +P; cd $dir/../; pwd)`/final.mdl
fi


for f in $lang/words.txt $model $data/segments $data/reco2file_and_channel $dir/lat.1.gz; do
  [ ! -f $f ] && echo "$0: expecting file $f to exist" && exit 1;
done

name=`basename $data`; # e.g. eval2000

mkdir -p $dir/scoring/log

if [ $stage -le 0 ]; then
  if [ ! -f $lang/phones/word_boundary.int ] ; then
    $cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/get_ctm.LMWT.log \
      set -e -o pipefail \; \
      mkdir -p $dir/score_LMWT/ '&&' \
      lattice-scale --inv-acoustic-scale=LMWT "ark:gunzip -c $dir/lat.*.gz|" ark:- \| \
      lattice-add-penalty --word-ins-penalty=$word_ins_penalty ark:- ark:- \| \
      lattice-prune --beam=$beam ark:- ark:- \| \
      lattice-align-words-lexicon $lang/phones/align_lexicon.int $model ark:- ark:- \| \
      lattice-to-ctm-conf --decode-mbr=$decode_mbr ark:- - \| \
      utils/int2sym.pl -f 5 $lang/words.txt  \| tee $dir/score_LMWT/$name.utt.ctm \| \
      utils/convert_ctm.pl $data/segments $data/reco2file_and_channel \
      '>' $dir/score_LMWT/$name.ctm || exit 1;
  else
    $cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/get_ctm.LMWT.log \
      set -e -o pipefail \; \
      mkdir -p $dir/score_LMWT/ '&&' \
      lattice-scale --inv-acoustic-scale=LMWT "ark:gunzip -c $dir/lat.*.gz|" ark:- \| \
      lattice-add-penalty --word-ins-penalty=$word_ins_penalty ark:- ark:- \| \
      lattice-prune --beam=$beam ark:- ark:- \| \
      lattice-align-words $lang/phones/word_boundary.int $model ark:- ark:- \| \
      lattice-to-ctm-conf --decode-mbr=$decode_mbr ark:- - \| \
      utils/int2sym.pl -f 5 $lang/words.txt  \| tee $dir/score_LMWT/$name.utt.ctm \| \
      utils/convert_ctm.pl $data/segments $data/reco2file_and_channel \
      '>' $dir/score_LMWT/$name.ctm || exit 1;
  fi
fi

if [ $stage -le 1 ]; then
  # Remove some stuff we don't want to score, from the ctm.
  for x in $dir/score_*/$name.ctm; do
    cp $x $x.bkup1;
    cat $x.bkup1 | grep -v -E '\[NOISE|LAUGHTER|VOCALIZED-NOISE\]' | \
      grep -v -E '<UNK>|%HESITATION|\(\(\)\)' | \
      grep -v -E '<eps>' | \
      grep -v -E '<noise>' | \
      grep -v -E '<silence>' | \
      grep -v -E '<hes>' | \
      grep -v -E '<unk>' | \
      grep -v -E '<v-noise>' | \
      perl -e '@list = (); %list = ();
      while(<>) {
        chomp;
        @col = split(" ", $_);
        push(@list, $_);
        $key = "$col[0]" . " $col[1]";
        $list{$key} = 1;
      }
      foreach(sort keys %list) {
        $key = $_;
        foreach(grep(/$key/, @list)) {
          print "$_\n";
        }
      }' > $x;
  done
fi


echo "Lattice2CTM finished on " `date`
exit 0
