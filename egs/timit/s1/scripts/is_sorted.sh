#!/bin/bash
# Copyright 2010-2011 Microsoft Corporation

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


# Usage: is_sorted.sh [script-file]
# This script returns 0 (success) if the script file argument [or standard input]
# is sorted and 1 otherwise.

export LC_ALL=C

if [ $# == 0 ]; then
  scp=-
fi
if [ $# == 1 ]; then
  scp=$1
fi
if [ $# -gt 1 -o "$1" == "--help" -o "$1" == "-h" ]; then
  echo "Usage: is_sorted.sh [script-file]"
  exit 1
fi

cat $scp > /tmp/tmp1.$$
sort /tmp/tmp1.$$ > /tmp/tmp2.$$
cmp /tmp/tmp1.$$ /tmp/tmp2.$$ >/dev/null
ret=$?
rm /tmp/tmp1.$$  /tmp/tmp2.$$
if [ $ret == 0 ]; then
   exit 0;
else
  echo "is_sorted.sh: script file $scp is not sorted";
  exit 1;
fi
