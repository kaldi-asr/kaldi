#!/usr/bin/env python3

# Copyright  2020  Yuri Khokhlov, Ivan Medennikov (STC-innovations Ltd)
# Apache 2.0.

"""This script transforms phone-indices in alignment to 0(silence phones), 1(speech phones), 2(spn phones)"""

import os
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Usage: conv_ali_to_vad_012.py 1:2:3:4:5 6:7:8:9:10 <in-text-phone-ali> <out-text-vad-ali>')
    parser.add_argument('silence_phones', type=str)
    parser.add_argument('spn_phones', type=str)
    parser.add_argument('phone_ali', type=str)
    parser.add_argument('vad_ali', type=str)
    args = parser.parse_args()

    print('Options:')
    print('  Silence phones (colon-separated list): {}'.format(args.silence_phones))
    print('  Spoken-noise phones (colon-separated list): {}'.format(args.spn_phones))
    print('  Input phone ali in text format: {}'.format(args.phone_ali))
    print('  Output vad ali in text format: {}'.format(args.vad_ali))

    silence_set = set(args.silence_phones.split(':'))
    print("sil phones: ")
    print(args.silence_phones.split(':'))
    spn_set = set(args.spn_phones.split(':'))
    print("spn phones: ")
    print(args.spn_phones.split(':'))

    assert os.path.exists(args.phone_ali), 'File does not exist {}'.format(args.phone_ali)
    parent = os.path.dirname(os.path.abspath(args.vad_ali))
    if not os.path.exists(parent):
        os.makedirs(parent)

    print('Starting to convert')
    count = 0
    with open(args.phone_ali) as ali_file:
        with open(args.vad_ali, 'wt') as vad_file:
            for line in ali_file:
                line = line.strip()
                if len(line) == 0:
                    continue
                parts = line.split(' ')
                parts = list(filter(None, parts))
                assert len(parts) > 1, 'Empty alignment in line {}'.format(line)
                vad_file.write('{}'.format(parts[0]))
                phones = parts[1:]
                for phone in phones:
                    if phone in silence_set:
                        vad_file.write(' 0')
                    elif phone in spn_set:
                        vad_file.write(' 2')
                    else: 
                        vad_file.write(' 1')
                vad_file.write('\n')
                count += 1
            vad_file.close()
        ali_file.close()
    print('Converted alignments for {} utterances'.format(count))

