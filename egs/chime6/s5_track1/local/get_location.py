#!/usr/bin/env python3
import json
from datetime import timedelta
from glob import glob
import sys, io


output = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
location_dict = {}

json_file_location= sys.argv[1] + '/*.json'
json_files = glob(json_file_location)

for file in json_files:
    with open(file, 'r') as f:
        session_dict = json.load(f)

    for uttid in session_dict:
        try:
            ref=uttid['ref']
            speaker_id = uttid['speaker']
            location = uttid['location']
            session_id=uttid['session_id']
            words = uttid['words']

            end_time_hh=str(uttid['end_time'][speaker_id])
            time = end_time_hh.strip().split(':')
            hrs, mins, secs = float(time[0]), float(time[1]), float(time[2])
            end_time = timedelta(hours=hrs, minutes=mins, seconds=secs).total_seconds()
            end_time = '{0:7.2f}'.format(end_time)
            end_time = "".join(end_time.strip().split('.'))
            end_time = int(end_time)
            end_time = str(end_time).zfill(7)

            start_time_hh = str(uttid['start_time'][speaker_id])
            time = start_time_hh.strip().split(':')
            hrs, mins, secs = float(time[0]), float(time[1]), float(time[2])
            start_time = timedelta(hours=hrs, minutes=mins, seconds=secs).total_seconds()
            start_time = '{0:7.2f}'.format(start_time)
            start_time = "".join(start_time.strip().split('.'))
            start_time = int(start_time)
            start_time =str(start_time).zfill(7)

            utt = "{0}_{1}-{2}-{3}".format(speaker_id, session_id, start_time, end_time)
            location_dict[utt]=(location)
        except:
            continue

for key in sorted(location_dict.keys()):
    utt= "{0} {1}".format(key, location_dict[key])
    output.write(utt+ '\n')

