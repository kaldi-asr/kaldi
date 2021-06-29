#!/usr/bin/env python
# Apache 2.0.
#
# This script converts an RTTM with speaker info into CHiME-6 JSON annotation
# Usage: convert_rttm_to_json.py exp/seg/rttm exp/seg/chime6.json"""

import json
import sys
from datetime import timedelta
import argparse

sys.path.insert(0, 'steps')
import libs.common as common_lib


def get_args():
    parser = argparse.ArgumentParser(
        description="""This script converts an RTTM with
        speaker info into Chime6 json annotation""")

    parser.add_argument("rttm_file", type=str, help="""Input RTTM file.
        The format of the RTTM file is <type> <file-id> <channel-id> <begin-time>
        <end-time> <NA> <NA> <speaker> <conf>""")
    
    parser.add_argument("json_file", type=str, help="Output json file")
    args = parser.parse_args()

    return args

def load_rttm(file):
    recoid_dict = {}
    with common_lib.smart_open(file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            sessionid_arrayid = parts[1]
            sessionid = sessionid_arrayid.split('_')[0]
            reference = sessionid_arrayid.split('_')[-1]
            start_time = float(parts[3])
            start_time_td = timedelta(seconds=start_time)

            time = str(start_time_td).split(':')
            hrs, mins, secs = time[0], time[1], float(time[2])
            secs1 = "{0:.2f}".format(secs)
            start_time_str = str(hrs) + ':' + str(mins) + ':' + str(secs1)

            end_time = start_time + float(parts[4])
            end_time_td = str(timedelta(seconds=end_time))

            time = str(end_time_td).split(':')
            hrs, mins, secs = time[0], time[1], float(time[2])
            secs1 = "{0:.2f}".format(secs)
            end_time_str = str(hrs) + ':' + str(mins) + ':' + str(secs1)

            spkr = parts[7]
            st = int(start_time * 100)
            end = int(end_time * 100)
            utt = "{0}-{1:06d}-{2:06d}".format(spkr, st, end)
            recoid_dict[utt] = (end_time_str, start_time_str, "NA", spkr, reference, "NA", sessionid)
    return recoid_dict


def main():
    args = get_args()
    recoid_dict = load_rttm(args.rttm_file)
    output = []
    with open(args.json_file, 'w') as json_file:
        for record_id in sorted(recoid_dict.keys()):
            utt_dict = {
                "end_time": recoid_dict[record_id][0],
                "start_time": recoid_dict[record_id][1],
                "words": recoid_dict[record_id][2],
                "speaker": recoid_dict[record_id][3],
                "ref": recoid_dict[record_id][4],
                "location": recoid_dict[record_id][5],
                "session_id": recoid_dict[record_id][6]
            }
            output.append(utt_dict)
        json.dump(output, json_file, indent=4, separators=(',', ':'))


if __name__ == '__main__':
    main()
