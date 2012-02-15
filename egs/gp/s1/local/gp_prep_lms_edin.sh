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
  --lm-dir=DIR\t\tDirectory containing language models\n
  --work-dir=DIR\t\tWorking directory\n
";

if [ $# -lt 2 ]; then
  error_exit $usage;
fi

while [ $# -gt 0 ];
do
  case "$1" in
  --help) echo -e $usage; exit 0 ;;
  --lm-dir=*)
  LMDIR=`read_dirname $1`; shift ;;
  --work-dir=*)
  WDIR=`read_dirname $1`; shift ;;
  *)  echo "Unknown argument: $1, exiting"; echo -e $usage; exit 1 ;;
  esac
done

cd $WDIR;
tmpdir=$(mktemp -d);
trap 'rm -rf "$tmpdir"' EXIT

# German - 17K
(
  { gzip -dc $LMDIR/ge/DE17k-tag.arpabo.3.gz | sed -e 's?DE=??g' \
      | gp_norm_lm.pl -i - | gzip -c > data/GE/local/lm_GE17k_tg.arpa.gz;
    subset_lm.pl -i data/GE/local/lm_GE17k_tg.arpa.gz -n 2 \
      -o data/GE/local/lm_GE17k_bg.arpa.gz;
    prune-lm --threshold=1e-7 $LMDIR/ge/DE17k-tag.arpabo.3.gz $tmpdir/lm_GE17.3
    sed -e 's?DE=??g' $tmpdir/lm_GE17.3 | gp_norm_lm.pl -i - | gzip -c \
      > data/GE/local/lm_GE17k_tg_pr.arpa.gz 
  } >& data/GE/prep_lms.log

# German - 60K 
  { gzip -dc $LMDIR/ge/DE60k.arpabo.3.gz | sed -e 's?DE=??g' \
      | gp_norm_lm.pl -i - | gzip -c > data/GE/local/lm_GE60k_tg.arpa.gz;
    subset_lm.pl -i data/GE/local/lm_GE60k_tg.arpa.gz -n 2 \
	-o data/GE/local/lm_GE60k_bg.arpa.gz;
    prune-lm --threshold=1e-7 $LMDIR/ge/DE60k.arpabo.3.gz $tmpdir/lm_GE60.3
    sed -e 's?DE=??g' $tmpdir/lm_GE60.3 | gp_norm_lm.pl -i - | gzip -c \
      > data/GE/local/lm_GE60k_tg_pr.arpa.gz
  } >> data/GE/prep_lms.log 2>&1
) &

# Portuguese - 60K
( gzip -dc $LMDIR/po/PO60k.arpabo.3.gz | gp_norm_lm.pl -i - \
    | gzip -c > data/PO/local/lm_PO60k_tg.arpa.gz
  subset_lm.pl -i data/PO/local/lm_PO60k_tg.arpa.gz -n 2 \
    -o data/PO/local/lm_PO60k_bg.arpa.gz
  prune-lm --threshold=1e-7 $LMDIR/po/PO60k.arpabo.3.gz $tmpdir/lm_PO60.3
  gp_norm_lm.pl -i $tmpdir/lm_PO60.3 | gzip -c \
    > data/PO/local/lm_PO60k_tg_pr.arpa.gz
) >& data/PO/prep_lms.log &

# Spanish - 23K
( gzip -dc $LMDIR/sp/SP23k-tag.arpabo.3.gz | sed -e 's?SP=??g' \
    | gp_norm_lm.pl -i - | gzip -c > data/SP/local/lm_SP23k_tg.arpa.gz
  subset_lm.pl -i data/SP/local/lm_SP23k_tg.arpa.gz -n 2 \
    -o data/SP/local/lm_SP23k_bg.arpa.gz
  prune-lm --threshold=1e-7 $LMDIR/sp/SP23k-tag.arpabo.3.gz $tmpdir/lm_SP23.3
  sed -e 's?SP=??g' $tmpdir/lm_SP23.3 | gp_norm_lm.pl -i - | gzip -c \
   > data/SP/local/lm_SP23k_tg_pr.arpa.gz
) >& data/SP/prep_lms.log &

# Swedish - 24K
( gzip -dc $LMDIR/sw/SW24k.arpabo.3.gz | gp_norm_lm.pl -i - \
    | gzip -c > data/SW/local/lm_SW24k_tg.arpa.gz
  subset_lm.pl -i data/SW/local/lm_SW24k_tg.arpa.gz -n 2 \
    -o data/SW/local/lm_SW24k_bg.arpa.gz
  prune-lm --threshold=1e-7 $LMDIR/sw/SW24k.arpabo.3.gz $tmpdir/lm_SW24.3
  gp_norm_lm.pl -i $tmpdir/lm_SW24.3 | gzip -c \
   > data/SW/local/lm_SW24k_tg_pr.arpa.gz
) >& data/SW/prep_lms.log &

wait;

echo "Finished preparing language models."
