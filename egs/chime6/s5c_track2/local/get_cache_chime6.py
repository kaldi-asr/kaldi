#!/usr/bin/env python3

# Copyright   2020   Prisyach Tatyana (STC-innovations Ltd)
# Apache 2.0.

import json
import argparse
import sys
import re

def get_args():
    parser = argparse.ArgumentParser(
        """This script creates chime6.json from ts-vad segments for dev and eval in the format required by pb_chime5
        e.g. {} segments \\
                dev \\
                CHiME6/audio/dev \\
                pb_chime5/cache/chime6.json""".format(sys.argv[0]))

    parser.add_argument("segments", help="""ts-vad segments""")
    parser.add_argument("dset_type", help="""dataset name (dev or eval)""")
    parser.add_argument("dir_chime6", help="""chime6 data directory to dev or eval""")
    parser.add_argument("json", help="""path to chime6.json""")
    args = parser.parse_args()
    return args

def create_main_fields(dset_type):
    if dset_type == "dev":
        to_json = { "alias": {"dev":["S02","S09"]}, "datasets": {"S02": {}, "S09": {}}}
    elif dset_type == "eval":
        to_json = { "alias": {"eval":["S01","S21"]}, "datasets": {"S01": {}, "S21": {}}}
    return to_json

def create_utt_field(dset_type, utt_name, ses, spk, time_start, time_end, fd, dir_chime6):
    if dset_type == "dev":
        if ses == "S02":
            dic_spk = ["P05", "P06", "P07", "P08"]
            kinects = ["U01", "U02", "U03", "U04", "U05", "U06"]
        elif ses == "S09":
            dic_spk = ["P25", "P26", "P27", "P28"]
            kinects = ["U01", "U02", "U03", "U04", "U06"]
    elif dset_type == "eval":
        if ses == "S01":
            dic_spk = ["P01", "P02", "P03", "P04"]
            kinects = ["U01", "U02", "U04", "U05", "U06"]
        elif ses == "S21":
            dic_spk = ["P45", "P46", "P47", "P48"]
            kinects = ["U01", "U02", "U03", "U04", "U05", "U06"]

    start = int(time_start * fd)
    end = int(time_end * fd)
    to_json = {"audio_path": {"observation": " "}, "end": end, "gender": " ", "location": " ", "notes": [], "num_samples": " ", "reference_array": " ", "session_id": " ", "speaker_id": " ", "start": start, "transcription": " "}
    to_json["num_samples"] = end - start
    to_json["session_id"] = ses
    to_json["speaker_id"] = dic_spk[spk-1]
    
    channels = 4
    to_json["audio_path"]["observation"] = {}
    for kinect in kinects:
        kinect_ch = []
        for ch in range(channels):
            kinect_ch.append(dir_chime6 + "/" + ses + "_" + kinect + "." + "CH" + str(ch+1) + ".wav")
        to_json["audio_path"]["observation"][kinect] = kinect_ch
    return to_json

def main():
    args = get_args()

    fd = 16000

    print("dset_type=", args.dset_type)
    json_chime6 = create_main_fields(args.dset_type)

    utt_list = open(args.segments).readlines()

    f_json = open(args.json, 'w')

    for utt in utt_list:
        utt_name, wav, time_start, time_end = utt.split(None)
        ses, kinect, spk, start, end = re.split('_|-', utt_name)
        utt_field = create_utt_field(args.dset_type, utt_name, ses, int(spk), float(time_start), float(time_end), fd, args.dir_chime6)
        json_chime6["datasets"][ses][utt_name] = utt_field

    f_json.write(json.dumps(json_chime6, indent=4))

if __name__ == '__main__':
    main()

