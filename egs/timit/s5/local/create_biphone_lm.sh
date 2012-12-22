# Copyright 2012  Navdeep Jaitly

# Is mostly a cut and paste operation, derived from 
# ../../../tools/kaldi_lm/train_lm.sh to create an lm for 
# biphone/bigram language models, which train_lm.sh does not
# deign to do.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.


# note: output is
# data/local/lm/3gram-mincount/lm_unpruned.gz 
# Expects train.gz, word_map in [argument 1 folder].
# Call from local/timit_train_lms.sh.

if [ $# != 1 ]; then
  echo "Usage: ../../local/create_biphone_lm.sh [lm folder]"
  echo "eg: ../../local/create_biphone_lm.sh data/local"
  exit 1; 
fi 


export PATH=$PATH:`pwd`/../../../tools/kaldi_lm
dir=$1

requirements="$dir/train.gz $dir/word_map"
for f in $requirements; do
  if [ ! -f $f ]; then
     echo "create_biphone_lm.sh: no such file $f"
     exit 1;
  fi
done

echo "Training biphone language model in folder $dir"
subdir=$dir/biphone
echo "Creating directory $subdir"
mkdir -p $subdir

# Clearly we don't have enough data to build a properly cross validated back-off model.
# In addition there is no need for a backoff model since we have all bigrams in the
# training data. However, taking out some of the data for validation set may remove
# some of the bigrams. This may seem like a bad thing, but could be a good thing if
# the resulting smoothing helps.

heldout_sent=300
write_arpa=1

if [ -s $subdir/ngrams.gz -a -s $subdir/heldout_ngrams.gz ]; then
  echo "Not creating raw N-gram counts ngrams.gz and heldout_ngrams.gz since they already exist in $subdir"
  echo "(remove them if you want them regenerated)"
else 
  echo Getting raw N-gram counts

  gunzip -c $dir/train.gz | tail -n +$heldout_sent | get_raw_ngrams 2 | sort | uniq -c |\
        uniq_to_ngrams | sort | gzip -c > $subdir/ngrams.gz    
  # Note: the Perl command below adds ":" before the count, which
  # is a marker that these N-grams are test N-grams.
  gunzip -c $dir/train.gz | head -n $heldout_sent | \
       get_raw_ngrams 2 | sort | uniq -c | uniq_to_ngrams | \
       perl -ane 's/(\S+)$/:$1/; print;' | sort | gzip -c > $subdir/heldout_ngrams.gz    
fi

cat > $subdir/config.0 <<EOF
D=0.4 tau=0.9 phi=2.0
D=0.6 tau=0.9 phi=2.0
D=0.8 tau=1.1 phi=2.0
EOF
cat > $subdir/config.diff_1 <<EOF
D=0 tau=1 phi=0
D=0 tau=1 phi=0
D=0 tau=1 phi=0
EOF
cat > $subdir/config.diff_2 <<EOF
D=0 tau=0 phi=0
D=0 tau=0 phi=0
D=1 tau=0 phi=0
EOF
cat > $subdir/config.diff_3 <<EOF
D=0 tau=0 phi=0
D=0 tau=0 phi=0
D=0 tau=1 phi=0
EOF
cat > $subdir/config.diff_4 <<EOF
D=0 tau=0 phi=0
D=0 tau=0 phi=0
D=0 tau=0 phi=1
EOF
cat > $subdir/config.diff_5 <<EOF
D=0 tau=0 phi=0
D=1 tau=0 phi=0
EOF
cat > $subdir/config.diff_6 <<EOF
D=0 tau=0 phi=0
D=0 tau=1 phi=0
EOF
cat > $subdir/config.diff_7 <<EOF
D=0 tau=0 phi=0
D=0 tau=0 phi=1
EOF
num_configs=7

awk '{print $2}' $dir/word_map > $dir/wordlist.mapped

# Define a subroutine
get_perplexity()  { # echoes the perplexity to stdout. uses current "$config" as config
  time gunzip -c $subdir/ngrams.gz | \
   discount_ngrams "$config" | sort | merge_ngrams | \
   interpolate_ngrams $dir/wordlist.mapped 0.5 | sort | \
   sort -m <(gunzip -c $subdir/heldout_ngrams.gz) - | compute_perplexity
}

mkdir -p $subdir/configs/ $subdir/perplexities/

if [ -f $subdir/config.$num_configs ]; then
  echo Not doing optimization of discounting parameters since
  echo file $subdir/config.$num_configs already exists
else
  for n in `seq 1 $num_configs`; do
    echo "Iteration $n/$num_configs of optimizing discounting parameters"
    for alpha in -0.25 0.0 0.35; do
      config=$subdir/configs/config.$n.$alpha
      # Note: if this ensure-nonnegative stuff gets active here it would cause
      # the optimization to give the wrong answer, but we've set up the config files
      # in such a way that this shouldn't happen.
      scale_configs.pl $subdir/config.$[$n-1] $subdir/config.diff_$n $alpha > $config
      get_perplexity > $subdir/perplexities/$n.$alpha &
    done
    wait
    optimize_alpha.pl -0.25 `cat $subdir/perplexities/$n.-0.25` \
                       0.0 `cat $subdir/perplexities/$n.0.0` \
                      0.35 `cat $subdir/perplexities/$n.0.35` > $subdir/perplexities/alpha.$n || exit 1;
    alpha=`cat $subdir/perplexities/alpha.$n`
    echo "Alpha value on iter $n is $alpha"
    scale_configs.pl $subdir/config.$[$n-1] $subdir/config.diff_$n $alpha > $subdir/config.$n
  done
fi
echo Final config is:
cat $subdir/config.$num_configs

# Create final LM as discounted (but not interpolated) N-grams:
if gunzip -c $subdir/ngrams_disc.gz >&/dev/null; then
  echo "Not creating discounted N-grams file $subdir/ngrams_disc.gz since it already exists"
else
  echo "Discounting N-grams."
  gunzip -c $subdir/ngrams.gz | \
   discount_ngrams $subdir/config.$num_configs | sort | merge_ngrams | \
   gzip -c > $subdir/ngrams_disc.gz
fi

echo "Computing final perplexity"
gunzip -c $subdir/ngrams_disc.gz | \
  interpolate_ngrams $dir/wordlist.mapped 0.5 | \
  sort | sort -m <(gunzip -c $subdir/heldout_ngrams.gz) - | \
  compute_perplexity 2>&1 | tee  $subdir/perplexity &


if [ $write_arpa == 1 ]; then
  echo "Building ARPA LM (perplexity computation is in background)"
  mkdir -p $subdir/tmpdir
  gunzip -c $subdir/ngrams_disc.gz | \
    interpolate_ngrams --arpa $dir/wordlist.mapped 0.5 | \
    sort | finalize_arpa.pl $subdir/tmpdir | \
    map_words_in_arpa.pl $dir/word_map | \
    gzip -c > $subdir/lm_unpruned.gz
fi

