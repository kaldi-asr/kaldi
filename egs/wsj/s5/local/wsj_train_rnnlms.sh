#!/bin/bash 

# Copyright 2012  Johns Hopkins University (author: Daniel Povey)  Tony Robinson

# This script trains LMs on the WSJ LM-training data.
# It requires that you have already run wsj_extend_dict.sh,
# to get the larger-size dictionary including all of CMUdict
# plus any OOVs and possible acronyms that we could easily 
# derive pronunciations for.

# This script takes no command-line arguments but takes the --cmd option.

# Begin configuration section.
rand_seed=0
cmd=run.pl
nwords=10000 # This is how many words we're putting in the vocab of the RNNLM. 
hidden=30
class=200 # Num-classes... should be somewhat larger than sqrt of nwords.
direct=1000 # Number of weights that are used for "direct" connections, in millions.
rnnlm_ver=rnnlm-0.3e # version of RNNLM to use
threads=1 # for RNNLM-HS
bptt=2 # length of BPTT unfolding in RNNLM
bptt_block=20 # length of BPTT unfolding in RNNLM
# End configuration section.

[ -f ./path.sh ] && . ./path.sh
. utils/parse_options.sh

if [ $# != 1 ]; then
   echo "Usage: local/wsj_train_rnnlms.sh [options] <dest-dir>"
   echo "For options, see top of script file"
   exit 1;
fi

dir=$1
srcdir=data/local/dict_larger
mkdir -p $dir

export PATH=$KALDI_ROOT/tools/$rnnlm_ver:$PATH


( # First make sure the kaldi_lm toolkit is installed.
 # Note: this didn't work out of the box for me, I had to
 # change the g++ version to just "g++" (no cross-compilation
 # needed for me as I ran on a machine that had been setup
 # as 64 bit by default.
 cd $KALDI_ROOT/tools || exit 1;
 if [ -f $rnnlm_ver/rnnlm ]; then
   echo Not installing the rnnlm toolkit since it is already there.
 else
   if [ $rnnlm_ver == "rnnlm-hs-0.1b" ]; then
       extras/install_rnnlm_hs.sh
   else
       echo Downloading and installing the rnnlm tools
       # http://www.fit.vutbr.cz/~imikolov/rnnlm/$rnnlm_ver.tgz
       if [ ! -f $rnnlm_ver.tgz ]; then
	   wget http://www.fit.vutbr.cz/~imikolov/rnnlm/$rnnlm_ver.tgz || exit 1;
       fi
       mkdir $rnnlm_ver
       cd $rnnlm_ver
       tar -xvzf ../$rnnlm_ver.tgz || exit 1;
       make CC=g++ || exit 1;
       echo Done making the rnnlm tools
       fi
   fi
) || exit 1;


if [ ! -f $srcdir/cleaned.gz -o ! -f $srcdir/lexicon.txt ]; then
  echo "Expecting files $srcdir/cleaned.gz and $srcdir/wordlist.final to exist";
  echo "You need to run local/wsj_extend_dict.sh before running this script."
  exit 1;
fi

cat $srcdir/lexicon.txt | awk '{print $1}' | grep -v -w '!SIL' > $dir/wordlist.all

# Get training data with OOV words (w.r.t. our current vocab) replaced with <UNK>.
echo "Getting training data with OOV words replaced with <UNK> (train_nounk.gz)" 
gunzip -c $srcdir/cleaned.gz | awk -v w=$dir/wordlist.all \
  'BEGIN{while((getline<w)>0) v[$1]=1;}
  {for (i=1;i<=NF;i++) if ($i in v) printf $i" ";else printf "<UNK> ";print ""}'|sed 's/ $//g' \
  | gzip -c > $dir/all.gz

echo "Splitting data into train and validation sets."
heldout_sent=10000
gunzip -c $dir/all.gz | head -n $heldout_sent > $dir/valid.in # validation data
gunzip -c $dir/all.gz | tail -n +$heldout_sent | \
 perl -e ' use List::Util qw(shuffle); @A=<>; print join("", shuffle(@A)); ' \
  > $dir/train.in # training data


  # The rest will consist of a word-class represented by <RNN_UNK>, that
  # maps (with probabilities) to a whole class of words.

# Get unigram counts from our training data, and use this to select word-list
# for RNNLM training; e.g. 10k most frequent words.  Rest will go in a class
# that we (manually, at the shell level) assign probabilities for words that
# are in that class.  Note: this word-list doesn't need to include </s>; this
# automatically gets added inside the rnnlm program.
# Note: by concatenating with $dir/wordlist.all, we are doing add-one
# smoothing of the counts.

cat $dir/train.in $dir/wordlist.all | grep -v '</s>' | grep -v '<s>' | \
  awk '{ for(x=1;x<=NF;x++) count[$x]++; } END{for(w in count){print count[w], w;}}' | \
  sort -nr > $dir/unigram.counts

head -$nwords $dir/unigram.counts | awk '{print $2}' > $dir/wordlist.rnn

tail -n +$nwords $dir/unigram.counts > $dir/unk_class.counts

tot=`awk '{x=x+$1} END{print x}' $dir/unk_class.counts`
awk -v tot=$tot '{print $2, ($1*1.0/tot);}' <$dir/unk_class.counts  >$dir/unk.probs


for type in train valid; do
  cat $dir/$type.in | awk -v w=$dir/wordlist.rnn \
    'BEGIN{while((getline<w)>0) v[$1]=1;}
    {for (i=1;i<=NF;i++) if ($i in v) printf $i" ";else printf "<RNN_UNK> ";print ""}'|sed 's/ $//g' \
    > $dir/$type
done
rm $dir/train.in # no longer needed-- and big.

# Now randomize the order of the training data.
cat $dir/train | awk -v rand_seed=$rand_seed 'BEGIN{srand(rand_seed);} {printf("%f\t%s\n", rand(), $0);}' | \
 sort | cut -f 2 > $dir/foo
mv $dir/foo $dir/train

# OK we'll train the RNNLM on this data.

# todo: change 100 to 320.
# using 100 classes as square root of 10k.
echo "Training RNNLM (note: this uses a lot of memory! Run it on a big machine.)"
#time rnnlm -train $dir/train -valid $dir/valid -rnnlm $dir/100.rnnlm \
#  -hidden 100 -rand-seed 1 -debug 2 -class 100 -bptt 2 -bptt-block 20 \
#  -direct-order 4 -direct 1000 -binary >& $dir/rnnlm1.log &

$cmd $dir/rnnlm.log \
   $KALDI_ROOT/tools/$rnnlm_ver/rnnlm -threads $threads -independent -train $dir/train -valid $dir/valid \
   -rnnlm $dir/rnnlm -hidden $hidden -rand-seed 1 -debug 2 -class $class -bptt $bptt -bptt-block $bptt_block \
   -direct-order 4 -direct $direct -binary || exit 1;


# make it like a Kaldi table format, with fake utterance-ids.
cat $dir/valid.in | awk '{ printf("uttid-%d ", NR); print; }' > $dir/valid.with_ids

utils/rnnlm_compute_scores.sh --rnnlm_ver $rnnlm_ver $dir $dir/tmp.valid $dir/valid.with_ids \
  $dir/valid.scores
nw=`wc -w < $dir/valid.with_ids` # Note: valid.with_ids includes utterance-ids which
  # is one per word, to account for the </s> at the end of each sentence; this is the
  # correct number to normalize buy.
p=`awk -v nw=$nw '{x=x+$2} END{print exp(x/nw);}' <$dir/valid.scores` 
echo Perplexity is $p | tee $dir/perplexity.log

rm $dir/train $dir/all.gz

# This is a better setup, but takes a long time to train:
#echo "Training RNNLM (note: this uses a lot of memory! Run it on a big machine.)"
#time rnnlm -train $dir/train -valid $dir/valid -rnnlm $dir/320.rnnlm \
#  -hidden 320 -rand-seed 1 -debug 2 -class 300 -bptt 2 -bptt-block 20 \
#  -direct-order 4 -direct 2000 -binary
