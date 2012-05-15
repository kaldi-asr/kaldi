#!/bin/bash



if [ $# != 3 ]; then
   echo "Word-align lattices (make the arcs sync up with words)"
   echo "Warning: this script (and the programs it call) make some assumptions"
   echo "about the phone set, e.g. that you used word-position-dependent phones."
   echo "Usage: scripts/walign_lats.sh <lang-dir> <input-decode-dir> <output-decode-dir>"
   exit 1;
fi

. path.sh || exit 1;

lang=$1
indir=$2
outdir=$3


mdl=`dirname $indir`/final.mdl

[ ! -f $mdl ] && echo "No such file $mdl" && exit 1;



# Get options strings for lattice-word-align programs.
# These specify which phone-ids are silence, which word-begin, etc.
mkdir -p $outdir

silphone=`grep -w -i sil $lang/phones.txt | awk '{print $2}'`
[ -z $silphone ] && echo "No silence phone-- weird name?" && exit 1;
opts="--silence-phones=$silphone";
# Following file has no form of silence.
cat $lang/phones.txt | grep -v -w '<eps>' | grep -v -w -E `sed "s/:/|/g" $lang/silphones.csl` \
  > $outdir/phones.txt.nosil

# Following file has just the noise phones, which may appear in the LM as <UNK> or noise --
# but not actual silence.
cat $lang/phones.txt | grep -v -w '<eps>' | grep -v -w -i sil | grep -w -E `sed "s/:/|/g" $lang/silphones.csl` \
  > $outdir/phones.txt.noise

opts="$opts --winternal-phones=`cat $outdir/phones.txt.nosil | grep -v _ | awk '{if (x == "") { x = $2; } else { x  = x ":" $2; }} END{print x;}'`"
opts="$opts --wbegin-phones=`cat $outdir/phones.txt.nosil | grep _B | awk '{if (x == "") { x = $2; } else { x  = x ":" $2; }} END{print x;}'`"
opts="$opts --wend-phones=`cat $outdir/phones.txt.nosil | grep _E | awk '{if (x == "") { x = $2; } else { x  = x ":" $2; }} END{print x;}'`"
opts="$opts --wbegin-and-end-phones=`cat $outdir/phones.txt.nosil | grep _S | awk '{if (x == "") { x = $2; } else { x  = x ":" $2; }} END{print x;}'`"
extra_wbegin_and_end_phones=`cat $outdir/phones.txt.noise| awk '{if (x == "") { x = $2; } else { x  = x ":" $2; }} END{print x;}'`
[ ! -z $extra_wbegin_and_end_phones ] && opts="$opts:$extra_wbegin_and_end_phones";  # Add things like SPN, NSN to list of
 # phones for --wbegin-and-end-phones.

if grep -Fw "!SIL" $lang/words.txt >/dev/null; then
  opts="$opts --silence-label=`grep -Fw "!SIL" $lang/words.txt | awk '{print $2}'`"
fi

for inlat in `echo $indir/lat.*.gz`; do
  outlat=`echo $inlat | sed "s|$indir|$outdir|"`
  lattice-word-align --test=true $opts $mdl "ark:gunzip -c $inlat|" "ark,t:|gzip -c >$outlat" || exit 1;
done
