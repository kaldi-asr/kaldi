#!/bin/bash

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

tgt=$1; shift
exp=$1; shift
lang=$1; shift

mkdir -p $tgt

echo "--- Exporting models to $tgt ..."

# See local/save_check.sh  which saves the settings at the beginning for details
# cp -f $exp/alex_gitlog.log $exp/alex_gitdiff.log $exp/experiment_bash_vars.log $tgt
cp -f $exp/experiment_bash_vars.log $tgt

# Store also the results
cp -f $exp/results.log $tgt/results.log


cp -f common/mfcc.conf $tgt 

cp -f $exp/tri2a/final.mdl $tgt/tri2a.mdl
cp -f $exp/tri2a/tree $tgt/tri2a.tree

cp -f $exp/tri2b/final.mdl $tgt/tri2b.mdl
cp -f $exp/tri2b/tree $tgt/tri2b.tree
cp -f $exp/tri2b/final.mat $tgt/tri2b.mat

cp -f $exp/tri2b_mmi_b*/final.mdl $tgt/tri2b_bmmi.mdl
cp -f $exp/tri2b/tree $tgt/tri2b_bmmi.tree
cp -f $exp/tri2b_mmi_b*/final.mat $tgt/tri2b_bmmi.mat

cp -f $lang/phones.txt $lang/phones/silence.csl $tgt


# FIXME do I need splice_opts for something?
