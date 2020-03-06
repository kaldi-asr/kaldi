#!/usr/bin/env bash
# Copyright Johns Hopkins University (Author: Daniel Povey) 2012.  Apache 2.0.

# begin configuration section.
cmd=run.pl
stage=0
decode_mbr=true
beam=4  # Use a fairly narrow beam because lattice-align-words is slow-ish.
word_ins_penalty=0.5
min_lmwt=7
max_lmwt=17
cleanup=true
model=

#end configuration section.

#debugging stuff
echo $0 $@

[ -f ./path.sh ] && . ./path.sh
[ -f ./cmd.sh ]  && . ./cmd.sh
. parse_options.sh || exit 1;

if [ $# -ne 4 ]; then
  echo "Usage: $0 [options] <dataDir> <langDir|graphDir> <w2s-dir>  <decodeDir>" && exit;
  echo "This is as lattice_to_ctm.sh, but for syllable-based systems where we want to"
  echo "obtain word-level ctms.  Here, <w2s-dir> is a directory like data/local/w2s,"
  echo "as created by run-6-syllables.sh.  It contains:"
  echo "   G.fst, Ldet.fst, words.txt, word_align_lexicon.int"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --stage (0|1)                 # (createCTM | filterCTM )."
  exit 1;
fi

data=$1
lang=$2 # Note: may be graph directory not lang directory, but has the necessary stuff copied.
w2sdir=$3
dir=$4

if [ -z "$model" ] ; then
  model=`dirname $dir`/final.mdl # Relative path does not work in some cases
  #model=$dir/../final.mdl # assume model one level up from decoding dir.
  #[ ! -f $model ] && model=`(set +P; cd $dir/../; pwd)`/final.mdl
fi

for f in $lang/words.txt $lang/phones/word_boundary.int \
     $model $data/segments $data/reco2file_and_channel $dir/lat.1.gz \
      $w2sdir/{G.fst,Ldet.fst,words.txt,word_align_lexicon.int}; do
  [ ! -f $f ] && echo "$0: expecting file $f to exist" && exit 1;
done

name=`basename $data`; # e.g. eval2000

mkdir -p $dir/scoring/log

# we are counting the LM twice since we have both the original, syllable-level LM
# and the new, word-level one, so we scale by 0.5 to get a reasonably scaled
# LM cost.

if [ $stage -le 0 ]; then
  nj=`cat $dir/num_jobs` || exit 1;
  $cmd JOB=1:$nj $dir/scoring/log/get_word_lats.JOB.log \
    lattice-compose "ark:gunzip -c $dir/lat.JOB.gz|" $w2sdir/Ldet.fst ark:- \| \
    lattice-determinize ark:- ark:- \| \
    lattice-compose ark:- $w2sdir/G.fst ark:- \| \
    lattice-scale --lm-scale=0.5 ark:- "ark:|gzip -c >$dir/wlat.JOB.gz" || exit 1;
fi

if [ $stage -le 1 ]; then
  $cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/get_ctm.LMWT.log \
    mkdir -p $dir/score_LMWT/ '&&' \
    lattice-scale --inv-acoustic-scale=LMWT "ark:gunzip -c $dir/wlat.*.gz|" ark:- \| \
    lattice-add-penalty --word-ins-penalty=$word_ins_penalty ark:- ark:- \| \
    lattice-prune --beam=$beam ark:- ark:- \| \
    lattice-push ark:- ark:- \| \
    lattice-align-words-lexicon --max-expand=10 --output-if-empty=true $w2sdir/word_align_lexicon.int $model ark:- ark:- \| \
    lattice-to-ctm-conf --decode-mbr=$decode_mbr ark:- - \| \
    utils/int2sym.pl -f 5 $w2sdir/words.txt  \| \
    utils/convert_ctm.pl $data/segments $data/reco2file_and_channel \
    '>' $dir/score_LMWT/$name.ctm || exit 1;
fi

if [ $stage -le 2 ]; then
  # Remove some stuff we don't want to score, from the ctm.
  for x in $dir/score_*/$name.ctm; do
    cp $x $x.bkup1;
    cat $x.bkup1 | grep -v -E '\[NOISE|LAUGHTER|VOCALIZED-NOISE\]' | \
      grep -v -E '<UNK>|%HESITATION|\(\(\)\)' | \
      grep -v -E '<eps>' | \
      grep -v -E '<noise>' | \
      grep -v -E '<silence>' | \
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

$cleanup && rm $dir/wlat.*.gz

echo "Lattice2CTM finished on " `date`
exit 0
