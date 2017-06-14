#!/bin/bash

# This script was copied from ../10hSystem/local (Author: Guoguo Chen?)
# It will be modified to make it somewhat more reusable
# But the current goal is just to have some LM while training AMs, so that
# intermediate models may be evaluated ... (Sanejev Khudanpur)

# This script trains LMs on the WSJ LM-training data.
# It requires that you have already run wsj_extend_dict.sh,
# to get the larger-size dictionary including all of CMUdict
# plus any OOVs and possible acronyms that we could easily
# derive pronunciations for.

# This script takes as command-line arguments the relevant data/lang
# and data/train directory.  We just train an LM on that.
# the output is G.fst, in the lang directory.

# Begin configuration section.
LMtoCompileIntoFST=4gram-mincount/lm_pr5.0.gz  # This is the LM to compile into G.fst
# End configuration section.

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: $0 lang-dir data-dir temp-dir";
  echo "  --LMtoCompileIntoFST ([34]gram[-mincount]*/lm_(unpruned|pr[12345].0).gz"
  exit 1;
fi

lang=$1
data=$2
dir=$3

mkdir -p $dir

export PATH=$PATH:/export/babel/sanjeev/kaldi-trunk/tools/kaldi_lm

##################################################################
# This portion just makes sure the kaldi-lm tools are available
##################################################################

( # First make sure the kaldi_lm toolkit is installed.
 cd /export/babel/sanjeev/kaldi-trunk/tools || exit 1;
 if [ -d kaldi_lm ]; then
   echo Not installing the kaldi_lm toolkit since it is already there.
 else
   echo Downloading and installing the kaldi_lm tools
   if [ ! -f kaldi_lm.tar.gz ]; then
     wget http://merlin.fit.vutbr.cz/kaldi/kaldi_lm.tar.gz || exit 1;
   fi
   tar -xvzf kaldi_lm.tar.gz || exit 1;
   cd kaldi_lm
   make || exit 1;
   echo Done making the kaldi_lm tools
 fi
) || exit 1;

##################################################################
# This portion gets the wordlist from the dictionary prep step,
# and the training text from the acoustic-training data prep step.
#
# It then cerates a training text file, w/ OOVs replaced by <unk>
##################################################################

cat $lang/words.txt | grep -v -w '#0' | awk '{print $1;}' > $dir/wordlist || exit 1;

cat $data/text | awk '{for (n=2;n<NF;n++) printf("%s ", $n); printf "\n";}' | \
 gzip -c > $dir/train_in.gz || exit 1;

# Get training data with OOV words (w.r.t. our current vocab) replaced with <unk>.
echo "Getting training data with OOV words replaced with <unk> (train_nounk.gz)"
gunzip -c $dir/train_in.gz | awk -v w=$dir/wordlist \
  'BEGIN{while((getline<w)>0) v[$1]=1;}
  {for (i=1;i<=NF;i++) if ($i in v) printf $i" ";else printf "<unk> ";print ""}'|sed 's/ $//g' \
  | gzip -c > $dir/train_nounk.gz

##################################################################
# Next is a NEW mapping of running words into integer sequences.
# Shouldn't it be the mapping used when creating the lexicon FST?
# Anyway, this mapping is reverse-sorted w.r.t. unigram counts
##################################################################

gunzip -c $dir/train_nounk.gz | cat - $dir/wordlist | \
  awk '{ for(x=1;x<=NF;x++) count[$x]++; } END{for(w in count){print count[w], w;}}' | \
 sort -nr > $dir/unigram.counts

# Get "mapped" words-- a character encoding of the words that makes the common words very short.
cat $dir/unigram.counts  | awk '{print $2}' | get_word_map.pl "<s>" "</s>" "<unk>" > $dir/word_map

gunzip -c $dir/train_nounk.gz | awk -v wmap=$dir/word_map 'BEGIN{while((getline<wmap)>0)map[$1]=$2;}
  { for(n=1;n<=NF;n++) { printf map[$n]; if(n<NF){ printf " "; } else { print ""; }}}' | gzip -c >$dir/train.gz

# To save disk space, remove the un-mapped training data.  We could
# easily generate it again if needed.
rm $dir/train_nounk.gz


##################################################################
# At this point, we are ready to build a few different Kaldi LMs
##################################################################

echo "Training 3gram-mincount"
train_lm.sh --arpa --lmtype 3gram-mincount $dir

echo "Pruning by threshold 1.0"
prune_lm.sh --arpa 1.0 $dir/3gram-mincount

echo "Pruning by threshold 2.0"
prune_lm.sh --arpa 2.0 $dir/3gram-mincount

echo "Pruning by threshold 3.0"
prune_lm.sh --arpa 3.0 $dir/3gram-mincount

echo "Pruning by threshold 4.0"
prune_lm.sh --arpa 4.0 $dir/3gram-mincount

echo "Pruning by threshold 5.0"
prune_lm.sh --arpa 5.0 $dir/3gram-mincount

echo "Training 3gram"
train_lm.sh --arpa --lmtype 3gram $dir

echo "Pruning by threshold 1.0"
prune_lm.sh --arpa 1.0 $dir/3gram

echo "Pruning by threshold 2.0"
prune_lm.sh --arpa 2.0 $dir/3gram

echo "Pruning by threshold 3.0"
prune_lm.sh --arpa 3.0 $dir/3gram

echo "Pruning by threshold 4.0"
prune_lm.sh --arpa 4.0 $dir/3gram

echo "Pruning by threshold 5.0"
prune_lm.sh --arpa 5.0 $dir/3gram

echo "Training 4gram-mincount"
train_lm.sh --arpa --lmtype 4gram-mincount $dir

echo "Pruning by threshold 1.0"
prune_lm.sh --arpa 1.0 $dir/4gram-mincount

echo "Pruning by threshold 2.0"
prune_lm.sh --arpa 2.0 $dir/4gram-mincount

echo "Pruning by threshold 3.0"
prune_lm.sh --arpa 3.0 $dir/4gram-mincount

echo "Pruning by threshold 4.0"
prune_lm.sh --arpa 4.0 $dir/4gram-mincount

echo "Pruning by threshold 5.0"
prune_lm.sh --arpa 5.0 $dir/4gram-mincount

echo "Training 4gram"
train_lm.sh --arpa --lmtype 4gram $dir

echo "Pruning by threshold 1.0"
prune_lm.sh --arpa 1.0 $dir/4gram

echo "Pruning by threshold 2.0"
prune_lm.sh --arpa 2.0 $dir/4gram

echo "Pruning by threshold 3.0"
prune_lm.sh --arpa 3.0 $dir/4gram

echo "Pruning by threshold 4.0"
prune_lm.sh --arpa 4.0 $dir/4gram

echo "Pruning by threshold 5.0"
prune_lm.sh --arpa 5.0 $dir/4gram

##################################################################
# Choose one of the LMs above and compile it into an FST as below
# The default LM chosen to be the last pruned 4gram-mincount
#
# Note: One can cheat and provide an external ARPA LM here!!!
#       To do so, make sure that
#         -- its vocabulary is fully covered by $lang/words.txt,
#         -- it is gzipped and
#         -- it is placed in the $dir directory.
#       Then simply provide its name via --LMtoCompileIntoFST
#
##################################################################

phone_disambig_symbol=`grep \#0 $lang/phones.txt | awk '{print $2}'`
word_disambig_symbol=`grep \#0 $lang/words.txt | awk '{print $2}'`

if [ -r $dir/$LMtoCompileIntoFST ]; then
    gzipped_ARPA_LM="$dir/$LMtoCompileIntoFST"
else
    echo "$0 WARNING: Cannot read ARPA LM $dir/$LMtoCompileIntoFST"
    echo "        Trying $LMtoCompileIntoFST"
    if [ -r $LMtoCompileIntoFST ]; then
	gzipped_ARPA_LM="$LMtoCompileIntoFST"
    else
	echo "$0 ERROR: Cannot read ARPA LM"
	exit 1
    fi
fi

echo "Compiling $gzipped_ARPA_LM into $lang/G.fst"

. ./path.sh || exit 1;
gunzip -c $gzipped_ARPA_LM | \
    arpa2fst --disambig-symbol=#0 \
             --read-symbol-table=$lang/words.txt - $lang/G.fst || exit 1;
fstisstochastic $lang/G.fst

##################################################################
# Redo the FST step after reviewing perplexities reported by the
# various training/pruning commands executed above
##################################################################

exit 0
