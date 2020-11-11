#!/usr/bin/env bash
# Copyright (c) 2013, Ondrej Platek, Ufal MFF UK <oplatek@ufal.mff.cuni.cz>
#
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
# limitations under the License. #

EXP=$1

# make sure that the directories exists
conflict=""
for d in $@ ; do
    if [ -d $d ] || [ -f $d ] ; then
        conflict="$conflict $d"
    fi
done

if [[ ! -z "$conflict" ]] ; then
    echo "Running new experiment will create following directories."
    echo "Some of them already exists!"
    echo ""
    echo "Existing directories:"
    for d in $conflict ; do 
        echo "   $d"
    done
    read -p "Should I delete the conflicting directories NOW y/n?"
    case $REPLY in
        [Yy]* ) echo "Deleting $conflict directories"; rm -rf $conflict;;
        * ) echo 'Keeping conflicting directories and exiting ...'; exit 1;;
    esac
fi

for d in $@ ; do
    mkdir -p $d
done

# Save the variables set up 
(set -o posix ; set ) > $EXP/experiment_bash_vars.log
# git log -1 > $EXP/alex_gitlog.log
# git diff > $EXP/alex_gitdiff.log
