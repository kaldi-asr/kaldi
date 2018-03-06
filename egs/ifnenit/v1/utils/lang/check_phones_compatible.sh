#!/bin/bash
# Copyright 2016 Hang Lyu

# Licensed udner the Apache License, Version 2.0 (the "Lincense");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OF IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

# This script exits with status zero if the phone symbols tables are the same
# except for possible differences in disambiguation symbols (meaning that all
# symbols except those beginning with a # are mapped to the same values).
# Otherwise it prints a warning and exits with status 1.
# For the sake of compatibility with other scripts that did not write the 
# phones.txt to model directories, this script exits silently with status 0 
# if one of the phone symbol tables does not exist.
# For the sake of compatibility with other scripts that did not write the 
# phones.txt to model directories, this script exits silently with status 0 
# if one of the phone symbol tables does not exist.

. utils/parse_options.sh || exit 1;

if [ $# -ne 2 ]; then
  echo "Usage: utils/lang/check_phones_compatible.sh <phones-symbol-table1> <phones-symbol-table2>"
  echo "e.g.: utils/lang/check_phones_compatible.sh data/lang/phones.txt exp/tri3/phones.txt"
  exit 1;
fi

table_first=$1
table_second=$2

# check the files exist or not 
if [ ! -f $table_first ]; then
  if [ ! -f $table_second ]; then
    echo "$0: Error! Both of the two phones-symbol tables are absent."
    echo "Please check your command"
    exit 1;
  else
    #The phones-symbol-table1 is absent. The model directory maybe created by old script.
    #For back compatibility, this script exits silently with status 0.
    exit 0;
  fi
elif [ ! -f $table_second ]; then
  #The phones-symbol-table2 is absent. The model directory maybe created by old script.
  #For back compatibility, this script exits silently with status 0.
  exit 0;
fi

#Check the two tables are same or not (except for possible difference in disambiguation symbols).
if ! cmp -s <(grep -v "^#" $table_first) <(grep -v "^#" $table_second); then
  echo "$0: phone symbol tables $table_first and $table_second are not compatible."
  exit 1;
fi

exit 0;
