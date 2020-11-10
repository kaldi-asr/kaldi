#!/usr/bin/env python3
# Copyright 2020 Xuechen LIU
# Apache 2.0.

import os
import re
import sys

MAX_MICID = 15

def _split_utt_from_wav(wav_name, mic_id, mode):
    wav_pathname = os.path.splitext(wav_name)[0]
    if mode == 'sc':
        return re.sub(r'{}', str(mic_id).zfill(2), wav_pathname)
    else:
        return wav_pathname

def main():
    input_trial = sys.argv[1]
    wav_scp = sys.argv[2]
    output_trial = sys.argv[3]
    
    mode = sys.argv[4]
    if mode not in ['sc', 'mc']:
        sys.exit('please only set mode to be sc or mc.')
    
    in_file = open(input_trial, 'r')
    out_file = open(output_trial, 'w')
    wavscp = open(wav_scp, 'r')

    # obviously not optimal, but file.read() does not work?
    utt_list = [line.split('/')[-1].rstrip() for line in wavscp]

    num_trials = 0
    print('generating trial for {0}'.format(mode))
    for line in in_file.readlines():
        model, test, decision = line.split()
        for i in range(1, MAX_MICID+1):
            wtest = _split_utt_from_wav(test, i, 'sc')
            wmodel = _split_utt_from_wav(model, i, mode)
            # we need to check if files exist
            model_wav = '{0}.wav'.format(wmodel)
            test_wav = '{0}.wav'.format(wtest)
            out_file.write("{0} {1} {2}\n".format(wmodel.replace('_', '-'),
                    wtest.replace('_', '-'), decision))
            num_trials += 1
    print('{0} trials generated for mode {1}.'.format(num_trials, mode))

    in_file.close()
    out_file.close()
    wavscp.close()


if __name__ == "__main__":
    main()
