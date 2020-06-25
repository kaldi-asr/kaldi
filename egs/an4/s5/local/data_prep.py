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

from __future__ import print_function
import os
import re
import sys

if len(sys.argv) != 3:
    print ('Usage: python data_prep.py [an4_root] [sph2pipe]')
    sys.exit(1)
an4_root = sys.argv[1]
sph2pipe = sys.argv[2]

sph_dir = {
    'train': 'an4_clstk', 
    'test': 'an4test_clstk'
}

for x in ['train', 'test']:
    with open(os.path.join(an4_root, 'etc', 'an4_' + x + '.transcription')) as transcript_f, \
         open(os.path.join('data', x, 'text'), 'w') as text_f, \
         open(os.path.join('data', x, 'wav.scp'), 'w') as wav_scp_f, \
         open(os.path.join('data', x, 'utt2spk'), 'w') as utt2spk_f:

        text_f.truncate()
        wav_scp_f.truncate()
        utt2spk_f.truncate()

        lines = sorted(transcript_f.readlines(), key=lambda s: s.split(' ')[0])
        for line in lines:
            line = line.strip()
            if not line:
                continue
            words = re.search(r'^(.*) \(', line).group(1)
            if words[:4] == '<s> ':
                words = words[4:]
            if words[-5:] == ' </s>':
                words = words[:-5]
            source = re.search(r'\((.*)\)', line).group(1)
            pre, mid, last = source.split('-')
            utt_id = '-'.join([mid, pre, last])

            text_f.write(utt_id + ' ' + words + '\n')
            wav_scp_f.write(utt_id + ' ' + sph2pipe + ' -f wav -p -c 1 ' + \
                os.path.join(an4_root, 'wav', sph_dir[x], mid, source + '.sph') + ' |\n')
            utt2spk_f.write(utt_id + ' ' + mid + '\n')
