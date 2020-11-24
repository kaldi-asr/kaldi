#!/usr/bin/env python3
# coding: utf-8
import sys
import re


def utterance_id(wav_id, spk, start_t, end_t):
    return '{spk_id:0>6s}_{start_t:0>7s}_{end_t:0>7s}_{wav_id}'.format(wav_id=wav_id,
                                                                     spk_id=spk,
                                                                     start_t=start_t,
                                                                     end_t=end_t)


def read_transcript(wav_id, f):
    transcript = list()
    for line in f:
        line = line.strip()
        try:
            start_t, end_t, spk, text = re.split(r'\s+', line, 3)
        except:
            start_t, end_t, spk = re.split(r'\s+', line, 2)
            text = ''

        if spk.startswith('background'):
            continue
            #spk = 'BKGR'

        if spk[-1] == ':':
            spk = spk[:-1]

        start_t = str(int(float(start_t) * 100))
        end_t = str(int(float(end_t) * 100))
        utt = utterance_id(wav_id, spk, start_t, end_t)
        transcript.append((utt, wav_id, start_t, end_t, spk, text))
    return transcript


def main(list_filename, debug=False):
    files = list()
    with open(list_filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            wav_id, text = line.split()
            files.append((wav_id, text))

    transcripts = dict()
    for wav_id, tsv in files:
        with open(tsv, 'r', encoding='utf-8') as f:
            transcript = read_transcript(wav_id, f)
            transcripts[wav_id] = transcript

    for wav_id in sorted(transcripts.keys()):
        for line in transcripts[wav_id]:
            print('\t'.join(line))


if __name__ == '__main__':
    main(sys.argv[1])
