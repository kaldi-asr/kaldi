#!/usr/bin/env bash

# E.g. of usage: ./convert_htk.sh ../.. /mnt/matylda5/jhu09/setup/CH1/English/exp/xwrd.R0_800_TB500/hmm84/MMF /mnt/matylda5/jhu09/setup/CH1/English/exp/xwrd.R0_800_TB500/hmm10_800_500/cluster.trees convert_dir

# This script takes as input the mmf and trees file and creates kaldi.tree and kaldi.mdl
# as output.

make_linear=no
cleanup=yes
for foo in 1 2 3; do
  if [ "$1" == "--linear-topology" ]; then
     make_linear=yes;
     shift;
  fi
  if [ "$1" == "--no-cleanup" ]; then
     cleanup=no;
     shift;
  fi
done

rootdir=$1 # e.g. /homes/eva/q/qpovey/UBM-ASR/branches/clean/
mmf=$2   #e.g. /mnt/matylda5/jhu09/setup/CH1/English/exp/xwrd.R0_800_TB500/hmm84/MMF
trees=$3 # e.g. /mnt/matylda5/jhu09/setup/CH1/English/exp/xwrd.R0_800_TB500/hmm10_800_500/cluster.trees
outdir=$4 # place to put tmp stuff + kaldi models.

# Convert relative to absolute pathnames in arguments.
D=`dirname "$rootdir"`
B=`basename "$rootdir"`
rootdir="`cd \"$D\" 2>/dev/null && pwd || echo \"$D\"`/$B"

D=`dirname "$trees"`
B=`basename "$trees"`
trees="`cd \"$D\" 2>/dev/null && pwd || echo \"$D\"`/$B"

D=`dirname "$mmf"`
B=`basename "$mmf"`
mmf="`cd \"$D\" 2>/dev/null && pwd || echo \"$D\"`/$B"

export PATH=$PATH:$rootdir/misc/htk_conversion/:$rootdir/src/gmmbin/

mkdir -p $outdir
cd $outdir

mmf2trans.pl < $mmf > trans1.txt
if [ $make_linear == "yes" ]; then
  make_linear.pl < trans1.txt > trans2.txt
else
  cp trans1.txt trans2.txt
fi

grep TRANSITION trans2.txt | awk '{print $2}' | sort > phonelist.txt

# Work out phones that appear in questions but not data (will remove them.)
grep QS $trees | \
 perl -ane ' foreach $a ( split(",", $_) ) { $a =~ m:\"(.+)\-\*: && print "$1\n"; $a =~ m:\"\*\+(.+)\": && print "$1\n"; } ' \
 | sort | uniq > phonelist_tree.txt

if diff phonelist.txt phonelist_tree.txt | grep -v a | grep -v '>'; then
   echo "Phones seen in tree but not model: error."
   exit 1;
fi

diff phonelist.txt phonelist_tree.txt | grep '>' | awk '{print $2}' > bad_phones.txt
# bad_phones.txt is phones that appear in the tree file but not the model file.
# Will just delete them from the tree

# Get initial tree.
parse_trees.pl  $trees  > trees1.txt

# Add stump "trees" for the context-independent phones.  This assumes they are macros-- otherwise, we have a problem.
get_ci_phone_trees.pl $mmf | cat trees1.txt -  > trees2.txt

# Remove "bad" phones (unseen phones) from the trees.
perl -e 'foreach $p (@ARGV) { $is_bad{$p} = 1; }; while(<STDIN>) { foreach $a(split(" ",$_)) 
  { if(!$is_bad{$a}) { print "$a "; } } print "\n"; } ' `cat bad_phones.txt` < trees2.txt > trees3.txt

# Add eps to every question that has "sil" in it (should produce better end-of-file behaviour.)
# have to treat the first symbol on each line specially (would be the phone whose tree it is).
cat trees3.txt | awk '{for(n=1;n<=NF;n++) { if(n>1 && $n == "sil") { printf("sil <eps> "); } else
   { printf("%s ", $n); } } printf("\n"); }' > trees4.txt

# Extract the GMMs for each state.
get_hmm_states.pl $mmf > states1.txt
cat states1.txt | grep STATE | awk '{print $2}' | sort > statelist.txt

# Make symbol-tables for the phones and states.
echo '<eps>' | cat - phonelist.txt | awk '{ printf("%s %d\n", $1, NR-1); }' > phones.txt
cat statelist.txt | awk '{ printf("%s %d\n", $1, NR-1); }' > statesyms.txt

integerize.pl phones.txt < trans2.txt > trans3.txt
# work out map from phone-id to number of emitting states for that phone.
grep TRANSITION trans3.txt | awk '{print $2 " " $3-2; }' > phone2len.txt

trans2topo.pl < trans3.txt > kaldi.topo

integerize.pl statesyms.txt < states1.txt > states2.txt

integerize.pl phones.txt statesyms.txt < trees4.txt > trees5.txt

# Sort the integers in the questions (the things between square brackets),
# the Kaldi code requires this.
cat trees5.txt | perl -ane ' chop; @A = (split("\\[",$_)); print shift @A; foreach $x (@A)
    { ($y,$z)=split("\\]",$x); 
   $y = join(" ", sort{$a <=> $b}(split(" ", $y))); print " [ $y ] $z "; } print "\n"; ' > trees6.txt

tree_convert.pl phone2len.txt trees6.txt > kaldi.tree

dim=`grep -w MEAN $mmf | head -1 | awk '{print $2}'` # probably 39.

convert_states.pl $dim < states2.txt > kaldi.am_gmm

gmm-init-trans --binary=false kaldi.topo kaldi.am_gmm kaldi.tree kaldi.mdl || exit 1;

# clean up:

if [ $cleanup == "yes" ]; then 
  mv phones.txt tmpa; 
  mv statesyms.txt tmpb;
  rm *.txt kaldi.am_gmm
  mv tmpa phones.txt
  mv tmpb statesyms.txt
fi

# now, kaldi.tree and kaldi.mdl are a Kaldi-format model; kaldi.topo may
# also be useful for some purposes.
