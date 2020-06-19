#!/usr/bin/env python

import os
import sys
from pathlib import Path

asr_result = sys.argv[1]
mt_dir = sys.argv[2]
file = sys.argv[3]
output_dir = sys.argv[4]

def get_asr_dict(file):
    asr_dict = dict()
    with open(file, 'r') as f:
        for line in f.readlines():
            line_split = line.split()
            utt_id = line_split[0]
            sentence = " ".join(line_split[1:])
            assert(utt_id not in asr_dict)
            asr_dict[utt_id] = sentence

    return asr_dict

def get_utt_id(line_list):
    start_time = "".join(line_list[2].split('.'))
    start_digit = len(line_list[2].split('.')[1])
    start_dif = 3 - start_digit

    end_time = "".join(line_list[3].split('.'))
    end_digit = len(line_list[3].split('.')[1])
    end_dif = 3 - end_digit

    if start_time == '00':
        start_time = '0'
    else:
        start_time += "0" * start_dif
    end_time += "0" * end_dif

    utt_id = line_list[0] + "_" + start_time + "_" + end_time
    return utt_id

def write_result(asr_dict, file, mt_dir, output_dir):
    # create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # read each file
    with open(file, 'r') as f:
        for line in f.readlines():
            file_name = line.split()[0]
            output_file_name = file_name.split('.')[0]
            with open(mt_dir+"/"+file_name, 'r') as input_file:
                with open(output_dir+"/"+output_file_name+".txt", 'w') as output_file:
                    for line in input_file.readlines():
                        line_split = line.split()
                        utt_id = get_utt_id(line_split)
                        if utt_id not in asr_dict:
                            print(utt_id)
                        else:
                            output_file.write(utt_id + " " + asr_dict[utt_id] + "\n")

def main():
    asr_dict = get_asr_dict(asr_result)
    write_result(asr_dict, file, mt_dir, output_dir)

if __name__ == "__main__":
    main()
