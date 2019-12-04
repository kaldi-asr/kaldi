#!/usr/bin/env python3
# Copyright   2019   Ashish Arora
# Apache 2.0.

import sys, io
import string
infile = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
output = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
spkorder_writer = open(sys.argv[1],'w', encoding='utf8')
total_words={}
total_errors={}
spk_order={}
total_errors_arrayid={}
total_words_arrayid={}

output.write('WER for each recording: \n')
for line in infile:
    toks = line.strip().split()
    recordingid = toks[1]
    total_words[recordingid] = toks[-5][:-1]
    total_errors[recordingid] = toks[-4][:-1]
    spk_order[recordingid] = toks[6][1] + '_' + toks[7][0] + '_' + toks[8][0] + '_' + toks[9][0]
    arrayid=recordingid.strip().split('_')[1]
    if arrayid not in total_errors_arrayid:
        total_errors_arrayid[arrayid]=0
        total_words_arrayid[arrayid]=0
    total_errors_arrayid[arrayid]+=int(total_errors[recordingid])
    total_words_arrayid[arrayid]+=int(total_words[recordingid])
    wer = float(total_errors[recordingid])/float(total_words[recordingid])*100
    utt = "{0} {1} {2} {3} {4:5.2f}".format(recordingid, spk_order[recordingid], total_words[recordingid], total_errors[recordingid], wer)
    output.write(utt + '\n')
    spkorder_writer.write(recordingid + ':' + str(spk_order[recordingid]) + '\n')


output.write('WER for each array: \n')
for arrayid in sorted(total_errors_arrayid):
    wer = float(total_errors_arrayid[arrayid])/float(total_words_arrayid[arrayid])*100
    utt = "{0} {1} {2} {3:5.2f}".format(arrayid, total_words_arrayid[arrayid], total_errors_arrayid[arrayid], wer)
    output.write(utt + '\n')

