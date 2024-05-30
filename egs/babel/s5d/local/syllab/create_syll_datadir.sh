#!/usr/bin/env bash
# Copyright (c) 2015, Johns Hopkins University ( Yenda Trmal <jtrmal@gmail.com> )
# License: Apache 2.0

help_message="Converts normal (with word level transcriptions) into syllabic\nExpects 4 parameters:\n"
# Begin configuration section.
boost_sil=1.0
cmd=run.pl
nj=4
# End configuration section
. ./utils/parse_options.sh

set -e -o pipefail
set -o nounset                              # Treat unset variables as an error

. ./cmd.sh
. ./path.sh

if [ $# -ne 4 ] ; then
  echo "$#"
  echo -e "$help_message"
  return 1;
fi

input=$1
word_lang=$2
syll_lang=$3
output=$4

[ ! -f exp/tri5/final.mdl  ] && \
  echo "File exp/tri5/final.mdl must exist" && exit 1;

[ ! -d $input/split$nj ] && utils/split_data.sh $input $nj

utils/copy_data_dir.sh $input $output
touch $output/.plp.done
touch $output/.done

if [ -f $input/text ] ; then
  steps/align_fmllr.sh \
      --boost-silence $boost_sil --nj $nj --cmd "$cmd" \
      $input $word_lang exp/tri5 exp/tri5_ali/align_$(basename $input)

  local/syllab/ali_to_syllabs.sh \
    --cmd "$cmd" \
    $input $syll_lang exp/tri5_ali/align_$(basename $input) \
    exp/tri5_ali_syll/align_$(basename $output)

  cp exp/tri5_ali_syll/align_$(basename $output)/text $output/text
fi

exit 0



