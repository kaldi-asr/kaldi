#!/bin/bash

# This script trains LMs on the WSJ LM-training data.
# It requires that you have already run wsj_extend_dict.sh,
# to get the larger-size dictionary including all of CMUdict
# plus any OOVs and possible acronyms that we could easily 
# derive pronunciations for.


# This script takes no command-line arguments

dir=data/local/rnnlm
srcdir=data/local/dict_larger
mkdir -p $dir
export PATH=$PATH:`pwd`/../../../tools/rnnlm-0.3c
nwords=10000 # This is how many words we're putting in the vocab of the RNNLM. 

( # First make sure the kaldi_lm toolkit is installed.
 # Note: this didn't work out of the box for me, I had to
 # change the g++ version to just "g++" (no cross-compilation
 # needed for me as I ran on a machine that had been setup
 # as 64 bit by default.
 cd ../../../tools || exit 1;
 if [ -d rnnlm-0.3c ]; then
   echo Not installing the rnnlm toolkit since it is already there.
 else
   echo Downloading and installing the rnnlm tools
   # http://www.fit.vutbr.cz/~imikolov/rnnlm/rnnlm-0.3c.tgz
   if [ ! -f rnnlm-0.3c.tgz ]; then
     wget http://www.fit.vutbr.cz/~imikolov/rnnlm/rnnlm-0.3c.tgz || exit 1;
   fi
   tar -xvzf rnnlm-0.3c.tgz || exit 1;
   cd  rnnlm-0.3c
   make || exit 1;
   echo Done making the rnnlm tools
 fi
) || exit 1;


if [ ! -f $srcdir/cleaned.gz -o ! -f $srcdir/wordlist.final ]; then
  echo "Expecting files $srcdir/cleaned.gz and $srcdir/wordlist.final to exist";
  echo "You need to run local/wsj_extend_dict.sh before running this script."
  exit 1;
fi

# Get training data with OOV words (w.r.t. our current vocab) replaced with <UNK>.
echo "Getting training data with OOV words replaced with <UNK> (train_nounk.gz)" 
gunzip -c $srcdir/cleaned.gz | awk -v w=$srcdir/wordlist.final \
  'BEGIN{while((getline<w)>0) v[$1]=1;}
  {for (i=1;i<=NF;i++) if ($i in v) printf $i" ";else printf "<UNK> ";print ""}'|sed 's/ $//g' \
  | gzip -c > $dir/all.gz

echo "Splitting data into train and validation sets."
heldout_sent=10000
gunzip -c $dir/all.gz | head -n $heldout_sent > $dir/valid # validation data
gunzip -c $dir/all.gz | tail -n +$heldout_sent | \
 perl -e ' use List::Util qw(shuffle); @A=<>; print join("", shuffle(@A)); ' \
  > $dir/train # training data


  # The rest will consist of a word-class represented by <RNN_UNK>, that
  # maps (with probabilities) to a whole class of words.

# Get unigram counts from our training data, and use this to select word-list
# for RNNLM training; e.g. 10k most frequent words.  Rest will go in a class
# that we (manually, at the shell level) assign probabilities for words that
# are in that class.  Note: this word-list doesn't need to include </s>; this
# automatically gets added inside the rnnlm program.
# Note: by concatenating with $srcdir/wordlist.final, we are doing add-one
# smoothing of the counts.

cat $dir/train $srcdir/wordlist.final | grep -v '</s>' | grep -v '<s>' | \
  awk '{ for(x=1;x<=NF;x++) count[$x]++; } END{for(w in count){print count[w], w;}}' | \
  sort -nr > $dir/unigram.counts

head -$nwords $dir/unigram.counts | awk '{print $2}' > $dir/wordlist.$nwords

tail -n +$nwords $dir/unigram.counts > $dir/unk_class.counts

tot=`awk '{x=x+$1} END{print x}' $dir/unk_class.counts`
awk -v tot=$tot '{print $2, ($1*1.0/tot);}' <$dir/unk_class.counts  >$dir/unk_class.probs


for type in train valid; do
  cat $dir/$type | awk -v w=$dir/wordlist.$nwords \
    'BEGIN{while((getline<w)>0) v[$1]=1;}
    {for (i=1;i<=NF;i++) if ($i in v) printf $i" ";else printf "<RNN_UNK> ";print ""}'|sed 's/ $//g' \
    > $dir/$type.$nwords
done

# OK we'll train the RNNLM on this data.

# todo: change 100 to 320.
# using 100 classes as square root of 10k.
echo "Training RNNLM (note: this uses a lot of memory! Run it on a big machine.)"
#time rnnlm -train $dir/train.$nwords -valid $dir/valid.$nwords -rnnlm $dir/100.$nwords.rnnlm \
#  -hidden 100 -rand-seed 1 -debug 2 -class 100 -bptt 2 -bptt-block 20 \
#  -direct-order 4 -direct 1000 -binary >& $dir/rnnlm1.log &

# add -independent?
time rnnlm -train $dir/train.$nwords -valid $dir/valid.$nwords -rnnlm $dir/30.$nwords.rnnlm -hidden 30 -rand-seed 1 -debug 2 -class 200 -bptt 2 -bptt-block 20 -direct-order 4 -direct 1000 -binary &> $dir/rnnlm3log 

mkdir $dir/rnnlm.voc$nwords.hl30
(
 cd $dir/rnnlm.voc$nwords.hl30
 ln -s ../30.$nwords.rnnlm rnnlm
 cp ../wordlist.$nwords wordlist.rnn
 cp ../unk_class.probs unk.probs
)

# Now we want to evaluate the likelihood on the validation data.
# We use the script rnnlm_compute_scores.sh, which has to
# include the OOV probabilities.

# make it like a Kaldi table format, with fake utterance-ids.
cat $dir/valid | awk '{ printf("uttid-%d ", NR); print; }' > $dir/valid.with_ids

scripts/rnnlm_compute_scores.sh $dir/rnnlm.voc$nwords.hl30 $dir/tmp.valid $dir/valid.with_ids \
  $dir/valid.scores
nw=`wc -w < $dir/valid.with_ids` # Note: valid.with_ids includes utterance-ids which
  # is one per word, to account for the </s> at the end of each sentence; this is the
  # correct number to normalize by.
p=`awk -v nw=$nw '{x=x+$2} END{print exp(x/nw);}' <$dir/valid.scores` 
echo Perplexity over $nw words is $p | tee $dir/perplexity


# This is a better setup, but takes a long time to train:
#echo "Training RNNLM (note: this uses a lot of memory! Run it on a big machine.)"
#time rnnlm -train $dir/train -valid $dir/valid -rnnlm $dir/320.rnnlm \
#  -hidden 320 -rand-seed 1 -debug 2 -class 300 -bptt 2 -bptt-block 20 \
#  -direct-order 4 -direct 2000 -binary

