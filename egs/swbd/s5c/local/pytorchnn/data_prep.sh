#!/usr/bin/env bash

# This script prepares the data directory for PyTorch based neural LM training.
# It prepares the following files in a output directory:
# 1. Vocabulary: $dir/words.txt copied from data/lang/words.txt.
# 2. Training and test data: $dir/{train/valid/test}.txt with each sentence per line.
#    Note: train.txt contains both training data of SWBD and Fisher. And the train
     # and dev datasets of SWBD are not the same as Kaldi RNNLM as we use
#    data/train_nodev/text as training data and data/train_dev/text as valid data.
#    While Kaldi RNNLM split data/train_nodev/text as train/dev for SWBD.
#    The test.txt can be any test set users are interested in, for example, eval2000.
#    We sorted utterances of each conversation of SWBD as we found it gives
#    better perplexities and WERs.


# Begin configuration section.
stage=0
train=data/train_nodev/text
valid=data/train_dev/text
test=data/eval2000/text
fisher=data/local/lm/fisher/text1.gz

. ./path.sh
. ./cmd.sh
. utils/parse_options.sh

set -e

if [ $# != 1 ]; then
   echo "Usage: $0 <dest-dir>"
   echo "For details of what the script does, see top of script file"
   exit 1;
fi

dir=$1 # data/pytorchnn/
mkdir -p $dir

for f in $train $valid $test $fisher; do
    [ ! -f $f ] && echo "$0: expected file $f to exist." && exit 1
done

# Sort and preprocess SWBD dataset
python3 local/pytorchnn/sort_by_start_time.py --infile $train --outfile $dir/swbd.train.sorted.txt
python3 local/pytorchnn/sort_by_start_time.py --infile $valid --outfile $dir/swbd.valid.sorted.txt
python3 local/pytorchnn/sort_by_start_time.py --infile $test --outfile $dir/swbd.test.sorted.txt
for data in train valid test; do
  cat $dir/swbd.${data}.sorted.txt | cut -d ' ' -f2- | tr 'A-Z' 'a-z' > $dir/$data.txt
  rm $dir/swbd.${data}.sorted.txt
done

# Process Fisher dataset 
mkdir -p $dir/config
cat > $dir/config/hesitation_mapping.txt <<EOF
hmm hum
mmm um
mm um
mhm um-hum
EOF
gunzip -c $fisher | awk 'NR==FNR{a[$1]=$2;next}{for (n=1;n<=NF;n++) if ($n in a) $n=a[$n];print $0}' \
  $dir/config/hesitation_mapping.txt - > $dir/fisher.txt

# Merge training data of SWBD and Fisher (ratio 3:1 to match Kaldi RNNLM's preprocessing)
cat $dir/train.txt $dir/fisher.txt $dir/train.txt $dir/train.txt > $dir/train_total.txt
rm $dir/train.txt
mv $dir/train_total.txt $dir/train.txt
rm $dir/fisher.txt

# Symbol for unknown words
echo "<unk>" >$dir/config/oov.txt
cp data/lang/words.txt $dir/
# Make sure words.txt contains the symbol for unknown words
if ! grep -w '<unk>' $dir/words.txt >/dev/null; then
  n=$(cat $dir/words.txt | wc -l)
  echo "<unk> $n" >> $dir/words.txt
fi

echo "Data preparation done."
