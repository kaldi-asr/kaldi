#!/usr/bin/env python3
# 2020 Dongji Gao

import sys

text_file = sys.argv[1]
segments = sys.argv[2]
stm = sys.argv[3]

channel = "1"

# segments:  <utterance-id> <recording-id> <segment-begin> <segment-end>
# super_id = recording_id
# this script is creating an stm file using kaldi segments and text file
with open(segments, 'r') as seg:
    seg_dict = dict()
    for line in seg.readlines():
        utt_id, super_id, start_time, end_time = line.strip().split()
        seg_dict[utt_id] = (super_id, start_time, end_time)


with open(text_file, 'r') as tf:
    with open(stm, 'w') as stm:
        super_list = []
        prev_super_id = ""
        for line in tf.readlines():
            line_list = line.strip().split()
            utt_id = line_list[0]
            text = " ".join(line_list[1:])
            super_id, start_time, end_time = seg_dict[utt_id]

            if super_id != prev_super_id:
                super_list_sorted = sorted(super_list, key=lambda x: float(x[3]))
                for element in super_list_sorted:
                   stm.write("{}\n".format(" ".join(element))) 
                prev_super_id = super_id
                super_list = [(super_id, channel, utt_id, start_time, end_time, text)]
            else:
                super_list.append((super_id, channel, utt_id, start_time, end_time, text))

        super_list_sorted = sorted(super_list, key=lambda x: float(x[3]))
        for element in super_list_sorted:
           stm.write("{}\n".format(" ".join(element))) 
