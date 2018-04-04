#!/usr/bin/env python

# Copyright 2012 Vassil Panayotov
# Apache 2.0

"""
Takes a "PROMPTS" file with lines like:
1snoke-20120412-hge/mfc/a0405 IT SEEMED THE ORDAINED ORDER OF THINGS THAT DOGS SHOULD WORK

, an ID prefix and a list of audio file names (e.g. for above example the list will contain "a0405").
It checks if the prompts file have transcription for all audio files in the list and
if this is the case produces a transcript line for each file in the format:
prefix_a0405 IT SEEMED THE ORDAINED ORDER OF THINGS THAT DOGS SHOULD WORK
"""

import sys

def err(msg):
    print >> sys.stderr, msg

if len(sys.argv) < 3:
    err("Usage: %s <prompts-file> <id-prefix> <utt-id1> <utt-id2> ... " % sys.argv[0])
    sys.exit(1)

#err(str(sys.argv))
id_prefix = sys.argv[2]
utt_ids = sys.argv[3:]
utt2trans = dict()
unnorm_utt = set() 
for l in file(sys.argv[1]):
    u, trans = l.split(None, 1)
    u = u.strip().split('/')[-1]
    trans = trans.strip().replace("-", " ")
    if not trans.isupper() or \
       not trans.strip().replace(' ', '').replace("'", "").isalpha():
        err("The transcript for '%s'(user '%s') is not properly normalized - skipped!"
            % (u, id_prefix))
        err(trans)
        unnorm_utt.add(u)
        continue
    utt2trans[u] = trans

for uid in utt_ids:
    if uid in unnorm_utt:
        continue # avoid double reporting the same problem
    if not uid in utt2trans:
        err("No transcript found for %s_%s" % (id_prefix, uid))
        continue
    print "%s-%s %s" % (id_prefix, uid, utt2trans[uid])

