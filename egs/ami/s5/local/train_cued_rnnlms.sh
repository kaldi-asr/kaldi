#!/bin/bash

train_text=
nwords=10000
hidden=200
cachesize=20
crit=ce

rnnlm_ver=cuedrnnlm

bptt=5

#set -v
set -e

. path.sh
. cmd.sh

. utils/parse_options.sh

if [ $# != 1 ]; then
   echo "Usage: $0 [options] <dest-dir>"
   echo "For options, see top of script file"
   exit 1;
fi

dir=$1
srcdir=data/local/dict

mkdir -p $dir

$KALDI_ROOT/tools/extras/check_for_rnnlm.sh "$rnnlm_ver" || exit 1
export PATH=$KALDI_ROOT/tools/$rnnlm_ver:$PATH

if [  True ]; then
cat $srcdir/lexicon.txt | awk '{print $1}' | grep -v -w '!SIL' > $dir/wordlist.all

# Get training data with OOV words (w.r.t. our current vocab) replaced with <UNK>.
# TODO(hxu) now use UNK for debugging
cat $train_text | awk -v w=$dir/wordlist.all \
  'BEGIN{while((getline<w)>0) v[$1]=1;}
  {for (i=2;i<=NF;i++) if ($i in v) printf $i" ";else printf "[UNK] ";print ""}'|sed 's/ $//g' \
  | perl -e ' use List::Util qw(shuffle); @A=<>; print join("", shuffle(@A)); ' \
  | gzip -c > $dir/all.gz

echo "Splitting data into train and validation sets."
heldout_sent=10000
gunzip -c $dir/all.gz | head -n $heldout_sent > $dir/valid.in # validation data
gunzip -c $dir/all.gz | tail -n +$heldout_sent > $dir/train.in # training data


  # The rest will consist of a word-class represented by <RNN_UNK>, that
  # maps (with probabilities) to a whole class of words.

# Get unigram counts from our training data, and use this to select word-list
# for RNNLM training; e.g. 10k most frequent words.  Rest will go in a class
# that we (manually, at the shell level) assign probabilities for words that
# are in that class.  Note: this word-list doesn't need to include </s>; this
# automatically gets added inside the rnnlm program.
# Note: by concatenating with $dir/wordlist.all, we are doing add-one
# smoothing of the counts.

# get rid of this design - 
cat $dir/train.in $dir/wordlist.all | grep -v '</s>' | grep -v '<s>' | \
  awk '{ for(x=1;x<=NF;x++) count[$x]++; } END{for(w in count){print count[w], w;}}' | \
  sort -nr > $dir/unigram.counts

total_nwords=`wc -l $dir/unigram.counts | awk '{print $1}'`

#head -$nwords_input $dir/unigram.counts | awk '{print $2}' | tee $dir/wordlist.rnn.input | awk '{print NR-1, $1}' > $dir/wordlist.rnn.id.input
#head -$nwords_output $dir/unigram.counts | awk '{print $2}' | tee $dir/wordlist.rnn.output | awk '{print NR-1, $1}' > $dir/wordlist.rnn.id.output
head -$nwords $dir/unigram.counts | awk '{print $2}' | tee $dir/wordlist.rnn | awk '{print NR-1, $1}' > $dir/wordlist.rnn.id

tail -n +$nwords $dir/unigram.counts > $dir/unk_class.counts

tot=`awk '{x=x+$1} END{print x}' $dir/unk_class.counts`
awk -v tot=$tot '{print $2, ($1*1.0/tot);}' <$dir/unk_class.counts  >$dir/unk.probs
# TODO(hxu) using RNN_UNK for debugging

false && for type in train valid; do
  cat $dir/$type.in | awk -v w=$dir/wordlist.rnn \
    'BEGIN{while((getline<w)>0) v[$1]=1;}
    {for (i=1;i<=NF;i++) if ($i in v) printf $i" ";else printf "RNN_UNK ";print ""}'|sed 's/ $//g' \
    > $dir/$type
done
#
for type in train valid; do
  mv $dir/$type.in $dir/$type
done

#rm $dir/train.in # no longer needed-- and big. #TODO(hxu) keep this to debug

# Now randomize the order of the training data.
cat $dir/train | awk -v rand_seed=$rand_seed 'BEGIN{srand(rand_seed);} {printf("%f\t%s\n", rand(), $0);}' | \
 sort | cut -f 2 > $dir/foo
mv $dir/foo $dir/train

# OK we'll train the RNNLM on this data.

echo "Training CUED-RNNLM on GPU"
#time rnnlm -train $dir/train -valid $dir/valid -rnnlm $dir/100.rnnlm \
#  -hidden 100 -rand-seed 1 -debug 2 -class 100 -bptt 2 -bptt-block 20 \
#  -direct-order 4 -direct 1000 -binary >& $dir/rnnlm1.log &

layer_str=$[$nwords+2]:$hidden:$[$nwords+2]
bptt_delay=0

# TODO(hxu) fullvocsize

$cuda_mem_cmd $dir/rnnlm.log \
   steps/train_cued_rnnlm.sh -train -trainfile $dir/train \
   -validfile $dir/valid -minibatch 64 -layers $layer_str \
   -bptt $bptt -bptt-delay $bptt_delay -traincrit $crit -lrtune newbob \
   -inputwlist $dir/wordlist.rnn.id -outputwlist $dir/wordlist.rnn.id \
   -independent 1 -learnrate 1.0 \
   -fullvocsize $total_nwords \
   -writemodel $dir/rnnlm -randseed 1 -debug 2
fi

# make it like a Kaldi table format, with fake utterance-ids.
cat $dir/valid | awk '{ printf("uttid-%d ", NR); print; }' > $dir/valid.with_ids

utils/rnnlm_compute_scores.sh --rnnlm_ver $rnnlm_ver $dir $dir/tmp.valid $dir/valid.with_ids $dir/valid.scores

nw=`wc -w < $dir/valid.with_ids` # Note: valid.with_ids includes utterance-ids which
  # is one per word, to account for the </s> at the end of each sentence; this is the
  # correct number to normalize buy.
p=`awk -v nw=$nw '{x=x+$2} END{print exp(x/nw);}' <$dir/valid.scores` 
echo Perplexity is $p | tee $dir/perplexity.log

#rm $dir/train $dir/all.gz TODO(hxu) keep this to debug

# This is a better setup, but takes a long time to train:
#echo "Training RNNLM (note: this uses a lot of memory! Run it on a big machine.)"
#time rnnlm -train $dir/train -valid $dir/valid -rnnlm $dir/320.rnnlm \
#  -hidden 320 -rand-seed 1 -debug 2 -class 300 -bptt 2 -bptt-block 20 \
#  -direct-order 4 -direct 2000 -binary
