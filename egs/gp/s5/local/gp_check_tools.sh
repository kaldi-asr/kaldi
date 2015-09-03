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

usage="Usage: "`basename $0`" <working-dir> <path.sh>"

if [ $# -lt 2 ]; then
  echo -e $usage; exit 1;
fi
WDIR=$1
path_file=$2

[ -f $path_file ] && . $path_file \
  || { echo "Need the path file to modify if shorten or sox are installed."; }

install_shorten=false
install_sox=false

shorten=`which shorten 2>/dev/null` \
  || { echo "shorten not found on PATH: installing"; install_shorten=true; }
sox=`which sox 2>/dev/null` \
  || { echo "sox not found on PATH: installing"; install_sox=true; }

# If shorten is found on path, check if the version is correct
if [ ! -z "$shorten" ]; then
  shorten_version=`$shorten -h 2>&1| head -1 | sed -e 's?.*version ??' -e 's?:.*??'`
  if [ $shorten_version != "3.6.1" ]; then
    echo "Unsupported shorten version $shorten_version found on path. Installing 3.6.1"
    install_shorten=true
  else
    echo "Using shorten (v$shorten_version) from $shorten"
  fi
fi

# If sox is found on path, check if the version is correct
if [ ! -z "$sox" ]; then
  sox_version=`$sox -h 2>&1| head -1 | sed -e 's?.*: ??' -e 's?.* ??'`
  if [ $sox_version != "v14.3.2" ]; then
    echo "Unsupported sox version $sox_version found on path. Installing 14.3.2"
    install_sox=true
  else
    echo "Using sox ($sox_version) from $sox"
  fi
fi

b=`basename $path_file .sh`
d=`dirname $path_file`
tmp_file=`mktemp`
trap 'rm -f "$tmp_file"' EXIT

if $install_shorten; then
  local/gp_install.sh --install-shorten $install_shorten $WDIR || exit 1
  cp $path_file $d/old-${b}.sh
  sed -e "s?^SHORTEN_BIN=?SHORTEN_BIN=$WDIR/tools/shorten-3.6.1/bin?" \
    $d/old-${b}.sh > $tmp_file
  echo 'export PATH=$PATH:$SHORTEN_BIN' >> $tmp_file
else
  cp $path_file $tmp_file
fi

if $install_sox; then
  local/gp_install.sh --install-sox $install_sox $WDIR || exit 1
  cp $path_file $d/old-${b}.sh
  sed -e "s?^SOX_BIN=?SOX_BIN=$WDIR/tools/sox-14.3.2/bin?" $tmp_file > $path_file
  echo 'export PATH=$PATH:$SOX_BIN' >> $path_file
fi
