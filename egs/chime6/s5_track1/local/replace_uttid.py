#!/usr/bin/env python3
# Copyright Ashish Arora
# Apache 2.0
# This script is used in score_for_submit. It adds locationid to the utteranceid,
# using uttid_location file, for locationwise scoring.

import sys, io
output = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def load_uttid_location(f):
    locations = {}
    for line in f:
        parts=line.strip().split(' ')
        uttid, loc = parts[0], parts[1]
        locations[uttid] = loc
    return locations

locations = load_uttid_location(open(sys.argv[1],'r', encoding='utf8'))

for line in open(sys.argv[2],'r', encoding='utf8'):
    uttid, res = line.split(None, 1)
    try:
        location = locations[uttid]
        location_uttid = location +'_'+ str(uttid)
        output.write(location_uttid + ' ' + res)
    except KeyError as e:
            raise Exception("Could not find utteranceid in "
                            "uttid_location file"
                            "({0})\n".format(str(e)))
