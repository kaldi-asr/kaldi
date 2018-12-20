#!/usr/bin/env python

from __future__ import print_function
import argparse, sys,os
from collections import defaultdict
parser = argparse.ArgumentParser(description="""
Combine consecutive utterances into fake speaker ids for a kind of
poor man's segmentation.  Reads old utt2spk from standard input,
outputs new utt2spk to standard output.""")
parser.add_argument("--utts-per-spk-max", type = int, required = True,
                    help="Maximum number of utterances allowed per speaker")
parser.add_argument("--seconds-per-spk-max", type = float, required = True,
                    help="""Maximum duration in seconds allowed per speaker.
                         If this option is >0, --utt2dur option must be provided.""")
parser.add_argument("--utt2dur", type = str,
                    help="""Filename of input 'utt2dur' file (needed only if
                    --seconds-per-spk-max is provided)""")
parser.add_argument("--respect-speaker-info", type = str, default = 'true',
                    choices = ['true', 'false'],
                    help="""If true, the output speakers will be split from "
                    "existing speakers.""")

args = parser.parse_args()

utt2spk = dict()
# an undefined spk2utt entry will default to an empty list.
spk2utt = defaultdict(lambda: [])

while True:
    line = sys.stdin.readline()
    if line == '':
        break;
    a = line.split()
    if len(a) != 2:
        sys.exit("modify_speaker_info.py: bad utt2spk line from standard input (expected two fields): " +
                 line)
    [ utt, spk ] = a
    utt2spk[utt] = spk
    spk2utt[spk].append(utt)

if args.seconds_per_spk_max > 0:
    utt2dur = dict()
    try:
        f = open(args.utt2dur)
        while True:
            line = f.readline()
            if line == '':
                break
            a = line.split()
            if len(a) != 2:
                sys.exit("modify_speaker_info.py: bad utt2dur line from standard input (expected two fields): " +
                         line)
            [ utt, dur ] = a
            utt2dur[utt] = float(dur)
        for utt in utt2spk:
            if not utt in utt2dur:
                sys.exit("modify_speaker_info.py: utterance {0} not in utt2dur file {1}".format(
                        utt, args.utt2dur))
    except Exception as e:
        sys.exit("modify_speaker_info.py: problem reading utt2dur info: " + str(e))

# splits a list of utts into a list of lists, based on constraints from the
# command line args.  Note: the last list will tend to be shorter than the others,
# we make no attempt to fix this.
def SplitIntoGroups(uttlist):
    ans = [] # list of lists.
    cur_uttlist = []
    cur_dur = 0.0
    for utt in uttlist:
        if ((args.utts_per_spk_max > 0 and len(cur_uttlist) == args.utts_per_spk_max) or
            (args.seconds_per_spk_max > 0 and len(cur_uttlist) > 0 and
             cur_dur + utt2dur[utt] > args.seconds_per_spk_max)):
            ans.append(cur_uttlist)
            cur_uttlist = []
            cur_dur = 0.0
        cur_uttlist.append(utt)
        if args.seconds_per_spk_max > 0:
            cur_dur += utt2dur[utt]
    if len(cur_uttlist) > 0:
        ans.append(cur_uttlist)
    return ans


# This function will return '%01d' if d < 10, '%02d' if d < 100, and so on.
# It's for printf printing of numbers in such a way that sorted order will be
# correct.
def GetFormatString(d):
    ans = 1
    while (d >= 10):
        d //= 10  # integer division
        ans += 1
    # e.g. we might return the string '%01d' or '%02d'
    return '%0{0}d'.format(ans)


if args.respect_speaker_info == 'true':
    for spk in sorted(spk2utt.keys()):
        uttlists = SplitIntoGroups(spk2utt[spk])
        format_string = '%s-' + GetFormatString(len(uttlists))
        for i in range(len(uttlists)):
            # the following might look like: '%s-%02d'.format('john_smith' 9 + 1),
            # giving 'john_smith-10'.
            this_spk = format_string % (spk, i + 1)
            for utt in uttlists[i]:
                print(utt, this_spk)
else:
    uttlists = SplitIntoGroups(sorted(utt2spk.keys()))
    format_string = 'speaker-' + GetFormatString(len(uttlists))
    for i in range(len(uttlists)):
        # the following might look like: 'speaker-%04d'.format(105 + 1),
        # giving 'speaker-0106'.
        this_spk = format_string % (i + 1)
        for utt in uttlists[i]:
            print(utt, this_spk)

