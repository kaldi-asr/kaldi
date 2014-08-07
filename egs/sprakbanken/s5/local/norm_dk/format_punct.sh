#!/bin/bash

# Copyright 2014 Andreas Kirkedal

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


#tmp1=`pwd`/anot.tmp

cat $1 | perl -pe 's/[\.|\!|\?]$/ \./g' | \
perl -pe 's/[\.|\!|\?] (_N[LS]_)$/ \. \1/g' | \
perl -pe 's/([:])([ |\n])/ \1\2/g' | \
perl -C -pe 's/["\t´\(\)\[\]\{\}]/ /g' | \
perl -C -pe "s/’/'/g" | \
perl -C -pe 's/[-,-] / /g' | \
perl -C -pe 's/([a-zæøå])-([A-ZÆØÅ])/\1 \2/g' | \
perl -C -pe 's/([A-ZÆØÅ])-([a-zæøå])/\1 \2/g' | \
perl -C -pe 's/[ |\.]-/ /g' | \
perl -C -pe 's/[_\.\+][_\.\+]+/ /g' | \
perl -pe 's/»|«|̣̣©|“|”|;//g' | \
perl -pe 's/([0-3][0-9][0-1][0-9][0-9][0-9])-([0-9][0-9][0-9][0-9])/\1_\2/g' | \
perl -pe 's/([0-2][0-9])[\.|\\]([0-5][0-9])/\1 \2/g' | \
perl -pe 's/([%])/ PROCENT /g' | \
perl -pe 's/([+])/ PLUS /g' | \
perl -pe 's/^- //g' | \
perl -pe 's/^ *| *$//g' | \

tr -s ' '
