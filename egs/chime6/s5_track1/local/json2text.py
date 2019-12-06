#!/usr/bin/env python3

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import json
import argparse
import logging
import sys


def hms_to_seconds(hms):
    hour = hms.split(':')[0]
    minute = hms.split(':')[1]
    second = hms.split(':')[2].split('.')[0]

    # .xx (10 ms order)
    ms10 = hms.split(':')[2].split('.')[1]

    # total seconds
    seconds = int(hour) * 3600 + int(minute) * 60 + int(second)

    return '{:07d}'.format(int(str(seconds) + ms10))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json', type=str, help='JSON transcription file')
    parser.add_argument('--mictype', type=str,
                        choices=['ref', 'worn', 'gss', 'u01', 'u02', 'u03', 'u04', 'u05', 'u06'],
                        help='Type of microphones')
    args = parser.parse_args()

    # logging info
    log_format = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s:%(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)

    logging.debug("reading %s", args.json)
    with open(args.json, 'rt', encoding="utf-8") as f:
        j = json.load(f)

    for x in j:
        if '[redacted]' not in x['words']:
            session_id = x['session_id']
            speaker_id = x['speaker']
            if args.mictype == 'ref':
                mictype = x['ref']
            elif args.mictype == 'worn' or args.mictype == 'gss':
                mictype = 'original'
            else:
                mictype = args.mictype.upper() # convert from u01 to U01

            # add location tag for scoring (only for dev and eval sets)
            if 'location' in x.keys():
                location = x['location'].upper()
            else:
                location = 'NOLOCATION'

            # remove mic dependency for CHiME-6
            start_time = x['start_time']
            end_time = x['end_time']

            # remove meta chars and convert to lower
            words = x['words'].replace('"', '')\
                              .replace('.', '')\
                              .replace('?', '')\
                              .replace(',', '')\
                              .replace(':', '')\
                              .replace(';', '')\
                              .replace('!', '').lower()

            # remove multiple spaces
            words = " ".join(words.split())

            # convert to seconds, e.g., 1:10:05.55 -> 3600 + 600 + 5.55 = 4205.55
            start_time = hms_to_seconds(start_time)
            end_time = hms_to_seconds(end_time)

            uttid = speaker_id + '_' + session_id
            if not args.mictype in ['worn', 'gss']:
                uttid += '_' + mictype

            if args.mictype == 'gss':
                uttid += '-' + start_time + '-' + end_time
            else:
                uttid += '_' + location + '-' + start_time + '-' + end_time

            # In several utterances, there are inconsistency in the time stamp
            # (the end time is earlier than the start time)
            # We just ignored such utterances.
            if end_time > start_time:
                sys.stdout.buffer.write((uttid + ' ' + words + '\n').encode("utf-8"))
