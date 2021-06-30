#!/usr/bin/env bash
# Copyright 2021  Xiaomi Corporation (Author: Yongqing Wang)
#                 Seasalt AI, Inc (Author: Guoguo Chen)

# Prepares the dictionary and auto-generates pronunciations for the words that
# are in our training data but not in the CMUdict.

set -e -o pipefail

nj=4 # number of parallel Sequitur G2P jobs, we would like to use
stage=0
cmd=run.pl

. ./path.sh || exit 1;
. ./utils/parse_options.sh || exit 1;


if [ $# -ne 3 ]; then
  echo "Usage: $0 [options] <g2p-model> <train-dir> <dict-dir>"
  echo " e.g.: $0 g2p/g2p.model.4 data/train_combined data/local/dict"
  echo "Options:"
  echo "  --cmd <command>      # script to launch jobs with, default: run.pl"
  echo "  --nj <nj>            # number of jobs to run, default: 4."
  exit 1
fi

g2p_model=$1
train_dir=$2
dict_dir=$3

lexicon=$dict_dir/lexicon.txt
lexicon_nosilence=$dict_dir/lexicon_nosilence.txt
cmudict_dir=$dict_dir/cmudict
cmudict_plain=$dict_dir/cmudict.0.7a.plain

[ -d $dict_dir ] || mkdir -p $dict_dir || exit 1;

if [ $stage -le 0 ]; then
  echo "$0: Downloading and preparing CMUdict"
  if [ ! -s $cmudict_dir/cmudict.0.7a ]; then
    svn co -r 12440 https://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict \
      $cmudict_dir || exit 1;
  fi
  echo "$0: Removing the pronunciation variant markers ..."
  grep -v ';;;' $cmudict_dir/cmudict.0.7a | \
    perl -ane 'if(!m:^;;;:){ s:(\S+)\(\d+\) :$1 :; print; }' \
    > $cmudict_plain || exit 1;
fi

if [ $stage -le 1 ]; then
  # Check if Sequitur G2P is installed.
  echo "$0: Checking Sequitur installation."
  sequitur=`which g2p.py`
  if [ -z $sequitur ] || [ ! -x $sequitur ]; then
    if ! which swig >&/dev/null; then
      echo "$0: Please first install swig, and then install Sequitur by running"
      echo "$0: $KALDI_ROOT/tools/extra/install_sequitur.sh"
      exit 1;
    else
      echo "$0: Installing Sequitur $KALDI_ROOT/tools/extra/install_sequitur.sh"
      pushd $KALDI_ROOT/tools
      extras/install_sequitur.sh || exit 1;
      popd
      . ./path.sh
      sequitur=`which g2p.py`
    fi
  fi
  if [ -z $sequitur ] || [ ! -x $sequitur ]; then
    echo "$0: Still can't find Sequitur, check your path.sh"
    exit 1;
  fi
fi

if [ $stage -le 2 ]; then
  echo "$0: Applying Sequitur G2P to out-of-vocabulary words"
  # Sanity check.
  [ ! -f "$train_dir/text" ] && echo "$0: Expecting $train_dir/text" && exit 1;
  [ -f $dict_dir/words-raw.txt ] && rm $dict_dir/words-raw.txt

  # Variables.
  g2p_dir=$dict_dir/g2p
  words=$dict_dir/words.txt
  g2p_oov=$g2p_dir/oov.txt
  g2p_lexicon=$g2p_dir/lexicon.txt
  mkdir -p $g2p_dir || exit 1;

  # All words from training transcripts.
  cat $train_dir/text |\
    sed 's|\t| |g' | cut -d " " -f 2- | sed 's| |\n|g' \
    >> $dict_dir/words-raw.txt || exit 1;
  sort -u $dict_dir/words-raw.txt | sed '/^$/d' > $words || exit 1;

  # Out-of-vocabulary words.
  awk 'NR==FNR{a[$1] = 1; next} !($1 in a)' $cmudict_plain $words |\
    sort > $g2p_oov || exit 1;

  # Spliting OOV words.
  g2p_oov_splits=$(for n in `seq $nj`; do echo $g2p_oov.$n; done)
  utils/split_scp.pl $g2p_oov $g2p_oov_splits || exit 1;

  echo "$0: Generating pronunciations for words in $g2p_oov.* ..."
  mkdir -p $g2p_dir/log
  $cmd JOB=1:$nj $g2p_dir/log/g2p.JOB.log \
    local/g2p.sh $g2p_model $g2p_oov.JOB $g2p_lexicon.JOB || exit 1;

  # Sanity check.
  g2p_oov_size=$(wc -l <$g2p_oov)
  g2p_lexicon_size=$(wc -l < <(cat $g2p_lexicon.*))
  if [ "$g2p_oov_size" -ne "$g2p_lexicon_size" ]; then
    echo "$0: Failed to generate pronunciations for all oov words. Input"
    echo "$0: $g2p_oov_size words v.s. output $g2p_lexicon_size entries."
    exit 1;
  fi
  sort <(cat $g2p_oov.*) > $dict_dir/g2p_oov.txt || exit 1;
  sort <(cat $g2p_lexicon.*) > $dict_dir/g2p_lexicon.txt || exit 1;
  echo "$0: $g2p_lexicon_size lexicon entries generated."
fi

if [ $stage -le 3 ]; then
  echo "$0: Combining CMUdict the G2P lexicon"
  cat $cmudict_plain $dict_dir/g2p_lexicon.txt |\
    sort > $lexicon_nosilence || exit 1;
  echo "$0: Combined lexicon saved to $lexicon_nosilence"
fi

if [ $stage -le 4 ]; then
  echo "$0: Creating lexicon related files."
  # Sanity check.
  if [ ! -f $lexicon_nosilence ] || [ ! -s $lexicon_nosilence ]; then
    echo "$0: $lexicon_nosilence is empty"
    exit 1;
  fi

  # Creating related files.
  silence_phones=$dict_dir/silence_phones.txt
  optional_silence=$dict_dir/optional_silence.txt
  nonsil_phones=$dict_dir/nonsilence_phones.txt
  extra_questions=$dict_dir/extra_questions.txt
  (echo SIL; echo SPN;) > $silence_phones || exit 1
  echo SIL > $optional_silence || exit 1
  awk '{for (i=2; i<=NF; ++i) { print $i; gsub(/[0-9]/, "", $i); print $i}}' \
    $lexicon_nosilence |\
    sort -u | perl -e '
      while (<>) {
        chop;
        m:^([^\d]+)(\d*)$: || die "Bad phone $_";
        $phones_of{$1} .= "$_ ";
      }
      foreach $list (values %phones_of) {
        print $list . "\n";
      }' | sort > $nonsil_phones || exit 1;
  cat $silence_phones | \
    awk '{printf("%s ", $1);} END{printf "\n";}' > $extra_questions || exit 1;
  cat $nonsil_phones | perl -e '
    while (<>) {
      foreach $p (split(" ", $_)) {
        $p =~ m:^([^\d]+)(\d*)$: || die "Bad phone $_";
        $q{$2} .= "$p ";
      }
    }
    foreach $l (values %q) {
      print "$l\n";
    }' >> $extra_questions || exit 1;
  echo "$0: $(wc -l <$silence_phones) silence phones in $silence_phones"
  echo "$0: $(wc -l <$optional_silence) optional silence in $optional_silence"
  echo "$0: $(wc -l <$nonsil_phones) non-silence phones in $nonsil_phones"
  echo "$0: $(wc -l <$extra_questions) extra questions in $extra_questions"

  # Creating lexicon.txt
  (echo '!SIL SIL'; echo '<SPOKEN_NOISE> SPN'; echo '<UNK> SPN'; ) |\
    cat - $lexicon_nosilence | sort | uniq > $lexicon || exit 1;
  echo "$0: Lexicon generated $lexicon"
fi

echo "$0: Done"
