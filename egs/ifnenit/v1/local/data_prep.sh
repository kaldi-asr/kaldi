#!/bin/bash
source cmd.sh
source path.sh

# To be run from one directory above this script.

# Note: when creating your own data preparation scripts, it's a good idea
# to make sure that the speaker id (if present) is a prefix of the utterance
# id, that the output scp file is sorted on utterance id, and that the 
# transcription file is exactly the same length as the scp file and is also
# sorted on utterance id (missing transcriptions should be removed from the
# scp file using e.g. scripts/filter_scp.pl)

# oldLC should be some utf8.*
oldLC=en_US.utf8

database='/export/b01/babak/IFN-ENIT/ifnenit_v2.0p1e/data'

# source ./path.sh

mkdir -p data

export LC_ALL=$oldLC

#if [ -d "blblbdalsdld" ]
#then


for set in 'train' 'test'
do
  ## Clean up
  if [[ -f tmp.unsorted ]]
  then
    rm tmp.unsorted
  fi
  if [ -d "data/$set" ]; then
    rm -r data/$set
  fi

  ## Gather transcriptions
  mkdir data/$set
  cat data/text.$set > tmp.unsorted
  # done
  export LC_ALL=C
  cat tmp.unsorted | sort -k1 > tmp.sorted
  export LC_ALL=$oldLC
  cat tmp.sorted | cut -d' ' -f1 > data/$set/uttids
  cat tmp.sorted | cut -d' ' -f2- | python3 local/remove_diacritics.py | python3 local/replace_arabic_punctuation.py | python3 local/replace_brackets.py | tr '+' '\\' | tr '=' '\\' | sed 's/\xA0/X/g' | sed 's/\x00\xA0/X/g' | sed 's/\xC2\xA0/X/g' | sed 's/\s\+/ /g' | sed 's/ \+$//' | sed 's/^ \+$//' | paste -d' ' data/$set/uttids - > data/$set/text
  rm tmp.unsorted tmp.sorted

  local/process_data.py $database data/$set --dataset $set --model_type word || exit 1
  sort data/$set/images.scp -o data/$set/images.scp
  sort data/$set/utt2spk -o data/$set/utt2spk

  export LC_ALL=C
  ./utils/utt2spk_to_spk2utt.pl data/$set/utt2spk > data/$set/spk2utt
  export LC_ALL=$oldLC

  mkdir -p data/{train,test}/data

  local/make_feature_vect.py data/$set --scale-size 40 | \
    copy-feats --compress=true --compression-method=7 \
    ark:- ark,scp:data/$set/data/images.ark,data/$set/feats.scp || exit 1

  steps/compute_cmvn_stats.sh data/$set || exit 1;


done



if [ -d "data/local" ]; then
  rm -r data/local
fi

## Determine phoneme set
mkdir -p data/local/lm
cat data/train/text | cut -d' ' -f2- | tr ' ' "\n" | sort -u > data/local/lm/train.vocab
cat data/local/lm/train.vocab | python3 local/get_atb_pronun.py > data/train/words2latin
cat data/train/text | cut -d' ' -f2- | python3 local/rollout_pronuns.py data/train/words2latin | cut -d' ' -f2- | tr ' ' "\n" | sort | uniq -c | awk '{if ($1 > 50 || length($2) == 3) print $2}' | fgrep -v '~A' > data/local/phonemeset

## Lexicon and word/phoneme lists
mkdir -p data/lang/
mkdir -p data/local/dict
echo '<unk>' > data/lang/oov.txt
cat data/train/words2latin | python3 local/map_to_rareA.py data/local/phonemeset > data/local/dict/lexicon.txt
echo "<unk> rareA" >> data/local/dict/lexicon.txt
echo "!SIL sil" >> data/local/dict/lexicon.txt
export LC_ALL=C
cat data/local/phonemeset | fgrep -v '.A' | fgrep -v ',A' | fgrep -v 'conn' | fgrep -v 'sil' | sort > data/local/dict/nonsilence_phones.txt
export LC_ALL=$oldLC
echo ',A' > data/local/dict/silence_phones.txt
echo '.A' >> data/local/dict/silence_phones.txt
echo 'conn' >> data/local/dict/silence_phones.txt
echo 'rareA' >> data/local/dict/silence_phones.txt
echo 'sil' >> data/local/dict/silence_phones.txt
echo 'sil' > data/local/dict/optional_silence.txt
# config folder
cat config/extra_questions.org.txt| python3 local/reduce-to-vocabulary.py data/local/dict/nonsilence_phones.txt | sort -u | fgrep ' ' > data/local/dict/extra_questions.txt

export LC_ALL=C
utils/prepare_lang.sh --num-sil-states 3 --num-nonsil-states 4 --position-dependent-phones false data/local/dict "<unk>" data/local/lang data/lang_pregdl_nolm
export LC_ALL=$oldLC


cut -d ' ' -f 2- data/train/text > data/local/lm/train.lines
cat data/test/text | awk '{ for(i=2;i<=NF;i++) print $i;}' | sort -u >test_words.txt
cat data/train/text | awk '{ for(i=2;i<=NF;i++) print $i;}' | sort -u >train_words.txt
utils/filter_scp.pl --exclude train_words.txt test_words.txt > diff.txt
cat diff.txt >> data/local/lm/train.lines

## Create LM (mix 1-gram LM on training transcriptions with decoding LM)
/export/b01/babak/srilm/bin/i686-m64/ngram-count -text data/local/lm/train.lines -map-unk "<unk>" -order 1 -lm data/local/lm/train.lm
gzip data/local/lm/train.lm
# Following command needs srilm/bin and srilm/bin/i686-m64/ in PATH
export LC_ALL=C
utils/format_lm_sri.sh --srilm-opts "-order 1" data/lang_pregdl_nolm data/local/lm/train.lm.gz data/local/dict/lexicon.txt data/lang_pregdl
export LC_ALL=$oldLC


mv data data.noligatures
mkdir -p data
cp -r data.noligatures/test data
cp -r data.noligatures/train data

mkdir -p data/lang/
mkdir -p data/local/dict
echo '<unk>' > data/lang/oov.txt
cp data.noligatures/local/dict/silence_phones.txt data/local/dict
cp data.noligatures/local/dict/optional_silence.txt data/local/dict
cp data.noligatures/local/dict/extra_questions.txt data/local/dict
export LC_ALL=en_US.utf8
cat data.noligatures/local/dict/lexicon.txt |  sed 's/\s\+la[BM]\{1\}\s\+conn\s\+a[meha]\{1\}E/ laLE/g' | python3 local/add_ligature_variants.py config/ligatures > data/local/dict/lexicon.txt
export LC_ALL=C
cat data/local/dict/lexicon.txt| cut -d' ' -f2- | tr ' ' "\n" | sort -u > data/local/phonemeset
cat data/local/phonemeset | fgrep -v 'rare' | fgrep -v '.A' | fgrep -v ',A' | fgrep -v 'conn' | fgrep -v 'sil' | sort > data/local/dict/nonsilence_phones.txt
utils/prepare_lang.sh --num-sil-states 3 --num-nonsil-states 4 --position-dependent-phones false data/local/dict "<unk>" data/local/lang data/lang_nolm
utils/format_lm_sri.sh --srilm-opts "-order 1" data/lang_nolm data.noligatures/local/lm/train.lm.gz data/local/dict/lexicon.txt data/lang




