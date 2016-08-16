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
# TODO(hxu) will fix the cued-rnnlm <unk> bug and change this
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

for type in train valid; do
  mv $dir/$type.in $dir/$type
done

# Now randomize the order of the training data.
cat $dir/train | awk -v rand_seed=$rand_seed 'BEGIN{srand(rand_seed);} {printf("%f\t%s\n", rand(), $0);}' | \
 sort | cut -f 2 > $dir/foo
mv $dir/foo $dir/train

# OK we'll train the RNNLM on this data.

echo "Training CUED-RNNLM on GPU"

layer_str=$[$nwords+2]:$hidden:$[$nwords+2]
bptt_delay=0

echo $layer_str > $dir/layer_string
$cuda_mem_cmd $dir/rnnlm.log \
   steps/train_cued_rnnlm.sh -train -trainfile $dir/train \
   -validfile $dir/valid -minibatch 64 -layers $layer_str \
   -bptt $bptt -bptt-delay $bptt_delay -traincrit $crit -lrtune newbob \
   -inputwlist $dir/wordlist.rnn.id -outputwlist $dir/wordlist.rnn.id \
   -independent 1 -learnrate 1.0 \
   -fullvocsize $total_nwords \
   -writemodel $dir/rnnlm -randseed 1 -debug 2
fi

touch $dir/unk.probs  # dummy file, not used for cued-rnnlm

# make it like a Kaldi table format, with fake utterance-ids.
cat $dir/valid | awk '{ printf("uttid-%d ", NR); print; }' > $dir/valid.with_ids

utils/rnnlm_compute_scores.sh --rnnlm_ver $rnnlm_ver $dir $dir/tmp.valid $dir/valid.with_ids $dir/valid.scores

nw=`cat $dir/valid.with_ids | sed 's= =\n=g' | wc -l | awk '{print $1}'` # Note: valid.with_ids includes utterance-ids which
  # is one per word, to account for the </s> at the end of each sentence; this is the
  # correct number to normalize buy.
p=`awk -v nw=$nw '{x=x+$2} END{print exp(x/nw);}' <$dir/valid.scores` 
echo Perplexity is $p | tee $dir/perplexity.log

