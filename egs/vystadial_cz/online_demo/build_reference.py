#!/usr/bin/env python
# encoding: utf-8
from __future__ import unicode_literals
from __future__ import print_function

import glob
import sys
import os
import codecs

def build_reference(wav_scp, ref_path):
    print(wav_scp, ref_path)
    with codecs.open(ref_path, 'w', 'utf-8') as w:
        with codecs.open(wav_scp, 'r', 'utf-8') as scp:
            for line in scp:
                name, wavpath = line.strip().split(' ', 1)
                with codecs.open(wavpath + '.trn', 'r', 'utf-8') as trn:
                    trans = trn.read().strip()
                    w.write(u'%s %s\n' % (name, trans))


if __name__ == '__main__':
    usage = '''
    Usage: python %(exec)s (audio_directory|in.scp) decode_directory

    Where directory contains files "*.scp" and
    audio files "*.wav" and their transcriptions "*.wav.trn".
    The "*.scp" files contains of list wav names and their path.

    The %(exec)s looks for "*.scp" files builds a reference from "*.wav.trn"
    '''
    usage_args = {'exec': sys.argv[0]}

    if len(sys.argv) != 3:
        print("Wrong number of arguments", file=sys.stderr)
        print(usage % {'exec': sys.argv[0]}, file=sys.stderr)
        sys.exit(1)

    if sys.argv[1].endswith('scp'):
        scps = [sys.argv[1]]
    else:
        scps = glob.glob(os.path.join(sys.argv[1], '*.scp'))
    target_dir = sys.argv[2]
    if not len(scps):
        print("No '*.scp' files found", file=sys.stderr)
        print(usage % {'exec': sys.argv[0]}, file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(target_dir):
        print("No '*.scp' files found", file=sys.stderr)
        print(usage % {'exec': sys.argv[0]}, file=sys.stderr)
        sys.exit(1)

    refers = [os.path.join(target_dir, os.path.basename(scp) + '.tra') for scp in scps]
    for scp, refer in zip(scps, refers):
        build_reference(scp, refer)
