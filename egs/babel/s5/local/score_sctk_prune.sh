#!/bin/bash
# Copyright Johns Hopkins University (Authors: Daniel Povey, Sanjeev Khudanpur) 2012-2013.  Apache 2.0.

# begin configuration section.
cmd=run.pl
stage=0
cer=0
decode_mbr=true
beam=5
word_ins_penalty=0
min_lmwt=7
max_lmwt=17
model=
#end configuration section.

[ -f ./path.sh ] && . ./path.sh
[ -f ./cmd.sh ]  && . ./cmd.sh
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: $0 [options] <dataDir> <langDir|graphDir> <decodeDir>" && exit;
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --stage (0|1|2)                 # (createCTM | filterCTM | runSclite)."
  echo "    --cer (0|1)                     # compute CER in addition to WER"
  exit 1;
fi

data=$1
lang=$2 # Note: may be graph directory not lang directory, but has the necessary stuff copied.
dir=$3

if [ -z "$model" ] ; then
  model=$dir/../final.mdl # assume model one level up from decoding dir.
fi


ScoringProgram=$KALDI_ROOT/tools/sctk/bin/sclite
[ ! -f $ScoringProgram ] && echo "Cannot find scoring program at $ScoringProgram" && exit 1;

for f in $data/char.stm $data/stm $data/glm $lang/words.txt $lang/phones/word_boundary.int \
     $model $data/segments $data/reco2file_and_channel $dir/lat.1.gz; do
  [ ! -f $f ] && echo "$0: expecting file $f to exist" && exit 1;
done

name=`basename $data`; # e.g. eval2000

mkdir -p $dir/scoring/log

if [ $stage -le 0 ]; then
  $cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/get_ctm.LMWT.log \
    mkdir -p $dir/score_LMWT/ '&&' \
    lattice-scale --inv-acoustic-scale=LMWT "ark:gunzip -c $dir/lat.*.gz|" ark:- \| \
    lattice-add-penalty --word-ins-penalty=$word_ins_penalty ark:- ark:- \| \
    lattice-prune --beam=$beam ark:- ark:- \| \
    lattice-align-words $lang/phones/word_boundary.int $model ark:- ark:- \| \
    lattice-to-ctm-conf --decode-mbr=$decode_mbr ark:- - \| \
    utils/int2sym.pl -f 5 $lang/words.txt  \| \
    utils/convert_ctm.pl $data/segments $data/reco2file_and_channel \
    '>' $dir/score_LMWT/$name.ctm || exit 1;
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
    cp $x $x.bkup2;
    y=${x%.ctm};
    cat $x.bkup2 | \
      perl -e '
      use Encode;
      while(<>) {
        chomp;
        @col = split(" ", $_);
        @col == 6 || die "Bad number of columns!";
        if ($col[4] =~ m/[\x80-\xff]{2}/) {
          $word = decode("UTF8", $col[4]);
          @char = split(//, $word);
          $start = $col[2];
          $dur = $col[3]/@char;
          $start -= $dur;
          foreach (@char) {
            $char = encode("UTF8", $_);
            $start += $dur;
            # printf "$col[0] $col[1] $start $dur $char\n"; 
            printf "%s %s %.2f %.2f %s %s\n", $col[0], $col[1], $start, $dur, $char, $col[5]; 
          }
        }
      }' > $y.char.ctm
    cp $y.char.ctm $y.char.ctm.bkup1
  done
fi

if [ $stage -le 2 ]; then
  $cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/score.LMWT.log \
    cp $data/stm $dir/score_LMWT/ '&&' cp $data/glm $dir/score_LMWT/ '&&'\
    $ScoringProgram -s -r $dir/score_LMWT/stm stm -h $dir/score_LMWT/${name}.ctm ctm -o all -o dtl;

  if [ $cer -eq 1 ]; then
    $cmd LMWT=$min_lmwt:$max_lmwt $dir/scoring/log/score.LMWT.char.log \
      cp $data/char.stm $dir/score_LMWT/'&&'\
      $ScoringProgram -s -r $dir/score_LMWT/char.stm stm -h $dir/score_LMWT/${name}.char.ctm ctm -o all -o dtl;
  fi
  
#  for x in $dir/score_*/*.ctm; do
#    mv $x.filt $x;
#    rm -f $x.filt*;
#  done

#  for x in $dir/score_*/*stm; do
#    mv $x.filt $x;
#    rm -f $x.filt*;
#  done
fi

echo "Finished scoring on" `date`
exit 0
