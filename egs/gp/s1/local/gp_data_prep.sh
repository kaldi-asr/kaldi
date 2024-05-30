#!/bin/bash -u

# Copyright 2012  Arnab Ghoshal

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

set -o errexit

function error_exit () {
  echo -e "$@" >&2; exit 1;
}

function read_dirname () {
  local dir_name=`expr "X$1" : '[^=]*=\(.*\)'`;
  [ -d "$dir_name" ] || error_exit "Argument '$dir_name' not a directory";
  local retval=`cd $dir_name 2>/dev/null && pwd || exit 1`
  echo $retval
}

PROG=`basename $0`;
usage="Usage: $PROG <arguments>\n
Prepare train, dev, eval file lists for a language.\n\n
Required arguments:\n
  --config-dir=DIR\tDirecory containing the necessary config files\n
  --corpus-dir=DIR\tDirectory for the GlobalPhone corpus\n
  --lm-dir=DIR\t\tDirectory containing language models\n
  --work-dir=DIR\t\tWorking directory\n
";

if [ $# -lt 4 ]; then
  error_exit $usage;
fi

while [ $# -gt 0 ];
do
  case "$1" in
  --help) echo -e $usage; exit 0 ;;
  --config-dir=*)
  CONFDIR=`read_dirname $1`; shift ;;
  --corpus-dir=*)
  GPDIR=`read_dirname $1`; shift ;;
  --lm-dir=*)
  LMDIR=`read_dirname $1`; shift ;;
  --work-dir=*)
  WDIR=`read_dirname $1`; shift ;;
  *)  echo "Unknown argument: $1, exiting"; echo -e $usage; exit 1 ;;
  esac
done

# (1) check if the config files are in place:
cd $CONFDIR
[ -f dev_spk.list ] || error_exit "$PROG: Dev-set speaker list not found.";
[ -f eval_spk.list ] || error_exit "$PROG: Eval-set speaker list not found.";
[ -f lang_codes.txt ] || error_exit "$PROG: Mapping for language name to 2-letter code not found.";

cd $WDIR
[ -f path.sh ] && . ./path.sh  # Sets the PATH to contain necessary executables

# (2) get the various file lists (for audio, transcription, etc.) for the
# specified language.
for LCODE in GE PO SP SW; do
  mkdir -p data/$LCODE
  gp_prep_flists.sh --corpus-dir=$GPDIR --dev-spk=$CONFDIR/dev_spk.list \
    --eval-spk=$CONFDIR/eval_spk.list --lang-map=$CONFDIR/lang_codes.txt \
    --work-dir=data $LCODE 2>data/$LCODE/prep_flists.log & 
  # Running these in parallel since this does audio conversion (to figure out
  # which files cannot be processed) and takes some time to run. 
done
wait;

# (3) Normalize the dictionary and transcripts.
for LCODE in GE PO SP SW; do
  full_name=`awk '/'$LCODE'/ {print $2}' $CONFDIR/lang_codes.txt`;
  gp_norm_dict_${LCODE}.pl -i $GPDIR/Dictionaries/${LCODE}/${full_name}-GPDict.txt | sort -u > data/$LCODE/local/lexicon_nosil_${LCODE}.txt
  (echo -e '!SIL\tSIL\n<UNK>\tSPN';) \
    | cat - data/$LCODE/local/lexicon_nosil_${LCODE}.txt \
    > data/$LCODE/local/lexicon_${LCODE}.txt;
  
  # add disambig symbols to the lexicon:
  ndisambig=`add_lex_disambig.pl data/$LCODE/local/lexicon_${LCODE}.txt data/$LCODE/local/lexicon_disambig_${LCODE}.txt`
  ndisambig=$[$ndisambig+1];  # add one disambig symbol for silence
  echo $ndisambig > data/$LCODE/local/lex_ndisambig

  # Get the list of phones and map them to integers (adding silence and spoken
  # nosie to the list).
  cut -f2 data/$LCODE/local/lexicon_nosil_${LCODE}.txt | sed -e "s?_.*??g" \
    | tr ' ' '\n' | sort -u \
    | awk 'BEGIN{ print "<eps> 0"; print "SIL 1"; print "SPN 2"; N=3; } 
           { printf("%s %d\n", $1, N++); }' > data/$LCODE/local/phones.txt
  # If using word-boundary markers on phones, use this in the awk command above
           # { printf("%s_WB %d\n", $1, N++); }
  # If using position markers on phones, use these in the awk command above
           # { printf("%s_B %d\n", $1, N++); }
           # { printf("%s_E %d\n", $1, N++); }
           # { printf("%s_S %d\n", $1, N++); }

  # Get the list of words:
  cut -f1 data/$LCODE/local/lexicon_${LCODE}.txt | sort -u \
    | awk 'BEGIN{print "<eps> 0";} {printf("%s %d\n", $1, NR);} 
           END{printf("#0 %d\n", NR+1);}' > data/$LCODE/local/words.txt

  for x in train dev eval; do
    gp_norm_trans_${LCODE}.pl -i data/$LCODE/local/${x}_${LCODE}.trans \
      > data/$LCODE/local/${x}_${LCODE}.trans2;
  done

done

# (4) Normalize the LMs - this is very Edinburgh-specific since we have some 
# LMs that came with the GlobalPhone corpus.
gp_prep_lms_edin.sh --lm-dir=$LMDIR --work-dir=$WDIR

echo "Finished data preparation."
