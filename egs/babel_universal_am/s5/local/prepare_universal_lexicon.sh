#!/bin/bash

gales_path=/export/MStuDy/Matthew/BABEL/babel-kaldi/egs/babel/PHONEMIC_SYSTEMS/GALES_SETUP
LANG="101"

. ./utils/parse_options.sh

if [ $# -eq 0 ]; then
  echo "Usage: ./local_/prepare_universal_lexicon.sh <ilex> <odict>"
  exit 1
fi

ilex=$1
odict=$2

if [ -f ${gales_path}/TONE_MAPS/${LANG} ]; then
  tonemap=${gales_path}/TONE_MAPS/${LANG}
fi

if [ -f ${gales_path}/DIPHTHONG_MAPS/${LANG} ]; then
  diphthongmap=${gales_path}/DIPHTHONG_MAPS/${LANG}
else
  echo "The file ${gales_path}/DIPHTHONG_MAPS/${LANG} is required."
  exit 1
fi

mkdir -p $odict 
python ./local/prepare_universal_lexicon.py $odict/lexicon.txt $ilex \
  ${gales_path}/DIPHTHONG_MAPS/${LANG} #${gales_path}/TONE_MAPS/${LANG}


echo -e "<silence> SIL\n<unk> <oov>\n<noise> <sss>\n<v-noise> <vns>" > ${odict}/silence_lexicon.txt
#grep "<hes>" ${odict}/lexicon.txt >  ${odict}/extra_lexicon.txt

#./local/prepare_unicode_lexicon.py --silence-lexicon ${odict}/silence_lexicon.txt \
#                                   --extra-lexicon ${odict}/extra_lexicon.txt \
#                                   $odict/lexicon.txt $odict

./local/prepare_unicode_lexicon.py --silence-lexicon ${odict}/silence_lexicon.txt \
                                   $odict/lexicon.txt $odict


exit




if [ $# -eq 0 ]; then
  echo "Usage: ./local/prepare_universal_lexicon.sh <odir> <lexicon> [<diphthong> [<tones>]]"
  exit 1
fi

odir=$1
lexicon=$2

mkdir -p $odir







