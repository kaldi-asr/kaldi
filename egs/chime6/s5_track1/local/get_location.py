#!/usr/bin/env python3
# Copyright Ashish Arora
# Apache 2.0
# This script create a utterance and location mapping file
# It is used in score_for_submit script to get locationwise WER.
# for GSS enhancement

import json
from datetime import timedelta
from glob import glob
import sys, io
from decimal import Decimal

SAMPLE_RATE = 16000

def to_samples(time: str):
    "mapping time in string to int, as mapped in pb_chime5"
    "see https://github.com/fgnt/pb_chime5/blob/master/pb_chime5/database/chime5/get_speaker_activity.py"
    hours, minutes, seconds = [t for t in time.split(':')]
    hours = int(hours)
    minutes = int(minutes)
    seconds = Decimal(seconds)

    seconds_samples = seconds * SAMPLE_RATE
    samples = (
        hours * 3600 * SAMPLE_RATE
        + minutes * 60 * SAMPLE_RATE
        + seconds_samples
    )
    return int(samples)


def main():
    output = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    json_file_location= sys.argv[1] + '/*.json'
    json_files = glob(json_file_location)

    json_file_location= sys.argv[1] + '/*.json'
    json_files = glob(json_file_location)
    location_dict = {}
    json_file_location= sys.argv[1] + '/*.json'
    json_files = glob(json_file_location)
    location_dict = {}
    for file in json_files:
        with open(file, 'r') as f:
            session_dict = json.load(f)

        for uttid in session_dict:
            try:
                ref=uttid['ref']
                speaker_id = uttid['speaker']
                location = uttid['location']
                location=location.upper()
                session_id=uttid['session_id']
                words = uttid['words']
                end_sample=to_samples(str(uttid['end_time']))
                start_sample=to_samples(str(uttid['start_time']))
                start_sample_str = str(int(start_sample * 100 / SAMPLE_RATE)).zfill(7)
                end_sample_str = str(int(end_sample * 100 / SAMPLE_RATE)).zfill(7)
                utt = "{0}_{1}-{2}-{3}".format(speaker_id, session_id, start_sample_str, end_sample_str)
                location_dict[utt]=(location)
            except:
                continue

    for key in sorted(location_dict.keys()):
        utt= "{0} {1}".format(key, location_dict[key])
        output.write(utt+ '\n')

if __name__ == '__main__':
    main()
