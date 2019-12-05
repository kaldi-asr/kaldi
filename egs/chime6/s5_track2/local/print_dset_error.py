#!/usr/bin/env python3
# Copyright   2019   Ashish Arora
# Apache 2.0.

import sys, io
import string
infile = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
output = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
spkorder_writer = open(sys.argv[1],'w', encoding='utf8')
array_id_error_dict={}
output.write('WER for each recording: \n')
for line in infile:
    toks = line.strip().split()
    recordingid = toks[1]
    total_words = toks[-5][:-1]
    total_errors = toks[-4][:-1]
    total_ins = toks[-3][:-1]
    total_del = toks[-2][:-1]
    total_sub = toks[-1]
    spk_order = toks[6][1] + '_' + toks[7][0] + '_' + toks[8][0] + '_' + toks[9][0]
    spkorder_writer.write(recordingid + ':' + spk_order + '\n')
    wer = float(total_errors)/float(total_words)*100
    recoid_spkorder = "{0} {1}".format(recordingid, spk_order)
    wer_detail = "WER {0:5.2f} [ {1} / {2}, {3} ins, {4} del, {5} sub ]".format(wer, total_words, total_errors, total_ins, total_del, total_sub)
    utt = recoid_spkorder + ' ' + wer_detail
    output.write(utt + '\n')
    arrayid=recordingid.strip().split('_')[1]
    if arrayid not in array_id_error_dict:
        array_id_error_dict[arrayid]=[0]*5
    array_id_error_dict[arrayid][0]+=int(total_words)
    array_id_error_dict[arrayid][1]+=int(total_errors)
    array_id_error_dict[arrayid][2]+=int(total_ins)
    array_id_error_dict[arrayid][3]+=int(total_del)
    array_id_error_dict[arrayid][4]+=int(total_sub)


output.write('WER for each array: \n')
for arrayid in sorted(array_id_error_dict):
    wer = float(array_id_error_dict[arrayid][1])/float(array_id_error_dict[arrayid][0])*100
    wer_detail = "WER {0:5.2f} [ {1} / {2}, {3} ins, {4} del, {5} sub ]".format(wer, array_id_error_dict[arrayid][0], array_id_error_dict[arrayid][1], array_id_error_dict[arrayid][2], array_id_error_dict[arrayid][3], array_id_error_dict[arrayid][4])
    output.write(wer_detail + '\n')

