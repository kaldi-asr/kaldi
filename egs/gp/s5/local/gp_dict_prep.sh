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
  printf "$@\n" >&2; exit 1;
}

function read_dirname () {
  [ -d "$1" ] || error_exit "Argument '$1' not a directory";
  local retval=`cd $1 2>/dev/null && pwd || exit 1`
  echo $retval
}

. ./path.sh    # Sets the PATH to contain necessary executables

# Begin configuration section.
config_dir=conf    # if true, use SRILM to change the LM vocab
map_dir=
# end configuration sections

help_message="Usage: "`basename $0`" [options] GP-dir LC [LC ... ]
where GP-dir is the directory containing the GlobalPhone corpus, and 
LC is a 2-letter code for GlobalPhone languages (e.g. RU for Russian).\n
options: 
  --help                # print this message and exit
  --config-dir DIR      # directory to find config files (default: $config_dir)
  --map-dir DIR         # directory to find phone mappings (default: '$map_dir')
";

. utils/parse_options.sh

if [ $# -lt 2 ]; then
  printf "$help_message\n"; exit 1;
fi

GPDIR=`read_dirname $1`; shift;
LANGUAGES=
while [ $# -gt 0 ]; do
  case "$1" in
  ??) LANGUAGES=$LANGUAGES" $1"; shift ;;
  *)  echo "Unknown argument: $1, exiting"; error_exit "$help_message" ;;
  esac
done

# (1) check if the config files are in place:
pushd $config_dir > /dev/null
[ -f dev_spk.list ] || error_exit "$PROG: Dev-set speaker list not found.";
[ -f eval_spk.list ] || error_exit "$PROG: Eval-set speaker list not found.";
[ -f lang_codes.txt ] || error_exit "$PROG: Mapping for language name to 2-letter code not found.";

popd > /dev/null
[ -f path.sh ] && . path.sh  # Sets the PATH to contain necessary executables

# (1) Normalize the dictionary
for L in $LANGUAGES; do
  printf "Language - ${L}: preparing pronunciation lexicon ... "
  mkdir -p data/$L/local/dict
  full_name=`awk '/'$L'/ {print $2}' $config_dir/lang_codes.txt`;
  pron_lex=$GPDIR/Dictionaries/${L}/${full_name}-GPDict.txt
  if [ ! -f "$pron_lex" ]; then
    pron_lex=$GPDIR/Dictionaries/${L}/${full_name}GP.dict  # Polish & Bulgarian
    [ -f "$pron_lex" ] || { echo "Error: no dictionary found for $L"; exit 1; }
  fi

  if [ ! -z "$map_dir" ]; then  # map the phones to a different phoneset
    if [ -f "$map_dir/$full_name" ]; then  # found the mapping file
      local/gp_norm_dict_${L}.pl -i "$pron_lex" -m "$map_dir/$full_name" \
	| sort -u > data/$L/local/dict/lexicon_nosil.txt
    else
      echo "No phone mapping '$map_dir/$full_name': keeping original phoneset";
      local/gp_norm_dict_${L}.pl -i "$pron_lex" | sort -u \
	> data/$L/local/dict/lexicon_nosil.txt
    fi
  else
    local/gp_norm_dict_${L}.pl -i "$pron_lex" | sort -u \
      > data/$L/local/dict/lexicon_nosil.txt
  fi

  (printf '!SIL\tsil\n<unk>\tspn\n';) \
    | cat - data/$L/local/dict/lexicon_nosil.txt \
    > data/$L/local/dict/lexicon.txt;
  echo "Done"

  printf "Language - ${L}: extracting phone lists ... "
  # silence phones, one per line.
  { echo sil; echo spn; } > data/$L/local/dict/silence_phones.txt
  echo sil > data/$L/local/dict/optional_silence.txt
  cut -f2- data/$L/local/dict/lexicon_nosil.txt | tr ' ' '\n' | sort -u \
    > data/$L/local/dict/nonsilence_phones.txt
  # Ask questions about the entire set of 'silence' and 'non-silence' phones. 
  # These augment the questions obtained automatically by clustering. 
  ( tr '\n' ' ' < data/$L/local/dict/silence_phones.txt; echo;
    tr '\n' ' ' < data/$L/local/dict/nonsilence_phones.txt; echo;
    ) > data/$L/local/dict/extra_questions.txt
  echo "Done"
done

echo "Finished dictionary preparation."
