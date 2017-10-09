#!/bin/bash -u

# Copyright 2015 (c) Johns Hopkins University (Jan Trmal <jtrmal@gmail.com>)

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

[ -f ./path.sh ] && . ./path.sh

uconv=`command -v uconv 2>/dev/null` \
  || { echo  >&2 "uconv not found on PATH. You will have to install ICU4C"; exit 1; }

sph2pipe=`command -v sph2pipe 2>/dev/null` \
  || { echo  >&2 "sph2pipe not found on PATH. Did you run make in the $KALDI_ROOT/tools directory?"; exit 1; }

srilm=`command -v ngram 2>/dev/null` \
  || { echo  >&2 "srilm not found on PATH. Please use the script $KALDI_ROOT/tools/extras/install_srilm.sh"; exit 1; }

sox=`command -v sox 2>/dev/null` \
  || { echo  >&2 "sox not found on PATH. Please install it manually (you will need version 14.4.0 and higher)."; exit 1; }

# If sox is found on path, check if the version is correct
if [ ! -z "$sox" ]; then
  sox_version=`$sox --version 2>&1| head -1 | sed -e 's?.*: ??' -e 's?.* ??'`
  if [[ ! $sox_version =~ v14.4.* ]]; then
    echo "Unsupported sox version $sox_version found on path. You will need version v14.4.0 and higher."
    exit 1
  fi
fi

exit  0


