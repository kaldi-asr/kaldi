#!/usr/bin/env bash
# Copyright (c) 2016, Johns Hopkins University ( Yenda Trmal <jtrmal@gmail.com> )
# License: Apache 2.0

# Begin configuration section.
# End configuration section
set -e -o pipefail
set -o nounset                              # Treat unset variables as an error

. ./cmd.sh;

devset=dev10h.pem
evalset=eval.seg
cmd="$decode_cmd"


rootdir=exp/nnet3/lstm_bidirectional_sp/decode_shadow.seg
combinedir=exp/combine/lstm_bidirectional_sp/shadow.seg

[ ! -d data/shadow.seg/compounds/$devset ] && \
  echo >&2 "data/shadow.seg/compounds/$devset does not exist!"  && exit 1
[ ! -d data/shadow.seg/compounds/$evalset ] && \
  echo >&2 "data/shadow.seg/compounds/$evalset does not exist!"  && exit 1

for decode in $rootdir/{,phones,syllabs}; do
  [ ! -d $decode ] && \
    echo >&2 "$decode does not exist!"  && exit 1
  local/nist_eval/filter_data.sh   \
    data/shadow.seg ${devset}  $decode
  local/nist_eval/filter_data.sh --ntrue-from ${devset}  \
    data/shadow.seg ${evalset} $decode
done



for kwset in data/shadow.seg/compounds/$devset/kwset_* ; do
  kwsetdir=$(basename $kwset)
  kwsetid=${kwsetdir#*_}

  echo "Processing kwset id=$kwsetid"
  local/search/combine.sh --extraid "$kwsetid"  --cmd "$cmd" \
    data/shadow.seg/compounds/${devset}/  data/langp_test \
    $rootdir/{,syllabs/,phones/}${devset}/${kwsetdir}  $combinedir/${devset}

  local/search/combine_special.sh --extraid "$kwsetid"  --cmd "$cmd" \
    data/shadow.seg/compounds/${evalset}/  data/langp_test \
    $combinedir/${devset}/${kwsetdir}/  \
    $rootdir/{,syllabs/,phones/}${evalset}/${kwsetdir}  $combinedir/${evalset}
done




