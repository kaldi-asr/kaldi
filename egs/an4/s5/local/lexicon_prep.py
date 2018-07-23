#!/usr/bin/env python

# Copyright 2016  Allen Guo

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

import os
import re
import sys

if len(sys.argv) != 2:
    print ('Usage: python lang_prep.py [an4_root]')
    sys.exit(1)
an4_root = sys.argv[1]

with open(os.path.join(an4_root, 'etc', 'an4.dic')) as dic_f, \
     open(os.path.join('data', 'local', 'dict', 'lexicon.txt'), 'w') as lexicon_f:

    lexicon_f.truncate()
    
    for line in dic_f.readlines():
        line = line.strip()
        if not line:
            continue        
        line = re.sub(r'(\(\d+\))?\s+', ' ', line)
        lexicon_f.write(line + '\n')

