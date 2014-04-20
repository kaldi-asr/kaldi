#!/bin/bash

# Copyright 2014 Mirsk Digital ApS  (Author: Andreas Kirkedal)

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


perl -pe 's/([\n ])\.([ \n])/\1PUNKTUM\2/g' | \
perl -pe 's/([\n ])\:([ \n])/\1KOLON\2/g' | \
perl -pe 's/([\n ])\;([ \n])/\1SEMIKOLON\2/g' | \
perl -pe 's/([\n ])_NL_([ \n])/\1NY LINJE\2/g' | \
perl -pe 's/([\n ])_NS_([ \n])/\1NYT AFSNIT\2/g' | \

tr -s ' '