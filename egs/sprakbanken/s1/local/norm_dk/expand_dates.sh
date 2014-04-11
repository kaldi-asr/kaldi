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


cat $1 | perl -pe 's/([0-3][0-9]\.)([0-1][0-9]\.)([1-2][0|9][0-9][0-9])([.| ])/\1 \2 \3\4/g' | \
perl -pe 's/den ([0-3]?[0-9])\/([0-1]?[0-9]) /den \1\. \2\. /g' | \
perl -pe 's/den ([0-3][0-9]\.)([0-1]?[0-9]\.) /den \1 \2 /g' | \
perl -pe 's/per ([0-3]?[0-9])\/([0-1]?[0-9]) /per \1\. \2\. /g' | \
perl -pe 's/([0-3]?[0-9])\/([0-1]?[0-9]) kl/\1\. \2\. kl/g' | \
perl -pe 's/([0-3]?[0-9])\/([0-1]?[0-9]) ([0-2][0-9])([.| ])/\1\. \2\. 20\3\4/g' | \
perl -pe 's/([0-3]?[0-9])\/([0-1]?[0-9]) ([3-9][0-9])([.| ])/\1\. \2\. 19\3\4/g'          


