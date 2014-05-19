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


cat $1 | perl -pe 's/([P|p])\.?[t|T]\./\1atienten/g' | \
perl -pe 's/([A|a])fd\./\1fdeling/g' | \
perl -pe 's/ve\./venstre/g' | \
perl -pe 's/hø\./højre/g' | \
perl -pe 's/ m.?h.?p\. / med henblik på /g' | \
perl -pe 's/mdr\./måneder/g' | \
perl -pe 's/ p ?n / p\.n\. /g' | \
perl -pe 's/konf\./konference/g' | \
perl -pe 's/i a ?d ?p/i a\.d\.p\./g' | \
perl -pe 's/([T|t])abl\./\1ablet/g' | \
perl -pe 's/dagl\./daglig/g' | \
perl -pe 's/nat\./naturlige/g' | \
perl -pe 's/([Aa])\.[Ss]\.[Aa]\./\1namnese som anført/g' | \
perl -pe 's/\/([al|nra])/ \/ \1/g' | \
perl -pe 's/ Hb / hæmoglobin /g' | \
perl -pe 's/([P|p])\.?g\.?a\./\1å grund af/g' | \
perl -pe 's/([R|r])\.?p\./\1ecipe/g' | \
perl -pe 's/[R|r]tg\./\1øntgen/g' | \
perl -pe 's/([S|s])v\.t\./\1varende til/g' | \
perl -pe 's/([S|s]t\.p\.) et c./\1_et_c\./g' | \
perl -pe 's/([S|s]tet) p et c/\1 p_et_c\./g' | \
perl -pe 's/ BS([\.|:|,| ])/Blodsukker\1/g' | \
perl -pe 's/ UE([\.|:| ])/Underekstremitet\1/g' | \
perl -pe 's/ AF([\.|:|,])/ Atrieflagren\1/g' | \
perl -pe 's/ AS([\.|:|,| ])/ Aortastenose\1/g' | \
perl -pe 's/ UL([\.|:|,| ])/ Ultralyd\1/g' | \
perl -pe 's/ BT([\.|:|,| ])/Blodtryk\1/g' | \
perl -pe 's/([S|s])at\./\1aturation/g' | \
perl -pe 's/([E|e])\.?v\.?t\.?/\1ventuelt/g' | \
perl -pe 's/([S|s])t\. ?([p|c]\.)/\1tet\. \2/g' | \
perl -pe 's/([S|s]tet) ?([p|c])/\1 \2\./g' | \
perl -pe 's/ ([D|d])\. / \1en /g' | \
perl -pe 's/([V|v])esk?\. ?resp\./\1esikulær respiration/g'
