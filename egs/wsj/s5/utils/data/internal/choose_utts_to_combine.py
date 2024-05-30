#!/usr/bin/env python

# Copyright 2016  Vijayaditya Peddinti
#           2016  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

from __future__ import print_function
import argparse
from random import randint
import sys
import os
from collections import defaultdict


parser = argparse.ArgumentParser(description="""
This script, called from data/utils/combine_short_segments.sh, chooses consecutive
utterances to concatenate that will satisfy the minimum segment length.  It uses the
--spk2utt file to ensure that utterances from the same speaker are preferentially
combined (as far as possible while respecting the minimum segment length).
If it has to combine utterances across different speakers in order to satisfy the
duration constraint, it will assign the combined utterances to the speaker which
contributed the most to the duration of the combined utterances.


The utt2uts output of this program is a map from new
utterance-id to a list of old utterance-ids, so for example if the inputs were
utt1, utt2 and utt3, and utterances 2 and 3 were combined, the output might look
like:
utt1 utt1
utt2-combine2 utt2 utt3
The utt2spk output of this program assigns utterances to the speakers of the input;
in the (hopefully rare) case where utterances were combined across speakers, it
will assign the utterance to whichever of the original speakers contributed the most
to the grouped utterance.
""")


parser.add_argument("--min-duration", type = float, default = 1.55,
                    help="Minimum utterance duration")
parser.add_argument("--merge-within-speakers-only", type = str, default = 'false',
                    choices = ['true', 'false'],
                    help="If true, utterances are only combined from the same speaker."
                    "It may be useful for the speaker recognition task."
                    "If false, utterances are preferentially combined from the same speaker,"
                    "and then combined across different speakers.")
parser.add_argument("spk2utt_in", type = str, metavar = "<spk2utt-in>",
                    help="Filename of [input] speaker to utterance map needed "
                    "because this script tries to merge utterances from the "
                    "same speaker as much as possible, and also needs to produce"
                    "an output utt2spk map.")
parser.add_argument("utt2dur_in", type = str, metavar = "<utt2dur-in>",
                    help="Filename of [input] utterance-to-duration map, with lines like 'utt1 1.23'.")
parser.add_argument("utt2utts_out", type = str, metavar = "<utt2utts-out>",
                    help="Filename of [output] new-utterance-to-old-utterances map, with lines "
                    "like 'utt1 utt1' or 'utt2-comb2 utt2 utt3'")
parser.add_argument("utt2spk_out", type = str, metavar = "<utt2spk-out>",
                    help="Filename of [output] utt2spk map, which maps new utterances to original "
                    "speakers.  If utterances were combined across speakers, we map the new "
                    "utterance to the speaker that contributed the most to them.")
parser.add_argument("utt2dur_out", type = str, metavar = "<utt2spk-out>",
                    help="Filename of [output] utt2dur map, which is just the summations of "
                    "the durations of the source utterances.")


args = parser.parse_args()


# This LessThan is designed to be impervious to roundoff effects in cases where
# numbers are really always separated by a distance >> 1.0e-05.  It will return
# false if x and y are almost identical, differing only by roundoff effects.
def LessThan(x, y):
    return x < y - 1.0e-5


# This function implements the core of the utterance-combination code.
# The input 'durations' is a list of durations, which must all be
# >=0.0  This function tries to combine consecutive indexes
# into groups such that for each group, the total duration is at
# least 'min_duration'.  It returns a list of (start,end) indexes.
# For example, CombineList(0.1, [5.0,6.0,7.0]) would return
# [ (0,1), (1,2), (2,3) ] because no combination is necessary; each
# returned pair represents a singleton group.
# Or CombineList(1.0, [0.5, 0.6, 0.7]) would return
# [ (0,3) ].
# Or CombineList(1.0, [0.5, 0.6, 1.7]) would return
# [ (0,2), (2,3) ].
# Note: if sum(durations) < min_duration, this function will
# return everything in one group but of course the sum of durations
# will be less than the total.
def CombineList(min_duration, durations):
    assert min_duration >= 0.0 and min(durations) > 0.0

    num_utts = len(durations)

    # for each utterance-index i, group_start[i] gives us the
    # start-index of the group of utterances of which it's currently
    # a member.
    group_start = list(range(num_utts))
    # if utterance-index i currently corresponds to the start of a group
    # of utterances, then group_durations[i] is the total duration of
    # that utterance-group, otherwise undefined.
    group_durations = list(durations)
    # if utterance-index i currently corresponds to the start of a group
    # of utterances, then group_end[i] is the end-index (i.e. last index plus one
    # of that utterance-group, otherwise undefined.
    group_end = [ x + 1 for x in range(num_utts) ]

    queue = [ i for i in range(num_utts) if LessThan(group_durations[i], min_duration) ]

    while len(queue) > 0:
        i = queue.pop()
        if group_start[i] != i or not LessThan(group_durations[i], min_duration):
            # this group no longer exists or already has at least the minimum duration.
            continue
        this_dur = group_durations[i]
        # left_dur is the duration of the group to the left of this group,
        # or 0.0 if there is no such group.
        left_dur = group_durations[group_start[i-1]] if i > 0 else 0.0
        # right_dur is the duration of the group to the right of this group,
        # or 0.0 if there is no such group.
        right_dur = group_durations[group_end[i]] if group_end[i] < num_utts else 0.0


        if left_dur == 0.0 and right_dur == 0.0:
            # there is only one group.  Nothing more to merge; break
            assert group_start[i] == 0 and group_end[i] == num_utts
            break
        # work out whether to combine left or right,
        # by means of the combine_left variable [ True or False ]
        if left_dur == 0.0:
            combine_left = False
        elif right_dur == 0.0:
            combine_left = True
        elif LessThan(left_dur + this_dur, min_duration):
            # combining left would still be below the minimum duration->
            # combine right... if it's above the min duration then good;
            # otherwise it still doesn't really matter so we might as well
            # pick one.
            combine_left = False
        elif LessThan(right_dur + this_dur, min_duration):
            # combining right would still be below the minimum duration,
            # and combining left would be >= the min duration (else we wouldn't
            # have reached this line) -> combine left.
            combine_left = True
        elif LessThan(left_dur, right_dur):
            # if we reached here then combining either way would take us >= the
            # minimum duration; but if left_dur < right_dur then we combine left
            # because that would give us more evenly sized segments.
            combine_left = True
        else:
            # if we reached here then combining either way would take us >= the
            # minimum duration; but  left_dur >= right_dur, so we combine right
            # because that would give us more evenly sized segments.
            combine_left = False

        if combine_left:
            assert left_dur != 0.0
            new_group_start = group_start[i-1]
            group_end[new_group_start] = group_end[i]
            for j in range(group_start[i], group_end[i]):
                group_start[j] = new_group_start
                group_durations[new_group_start] += durations[j]
            # note: there is no need to add group_durations[new_group_start] to
            # the queue even if it is still below the minimum length, because it
            # would have previously had to have been below the minimum length,
            # therefore it would already be in the queue.
        else:
            assert right_dur != 0.0
            # group start doesn't change, group end changes.
            old_group_end = group_end[i]
            new_group_end = group_end[old_group_end]
            group_end[i] = new_group_end
            for j in range(old_group_end, new_group_end):
                group_durations[i] += durations[j]
                group_start[j] = i
            if LessThan(group_durations[i], min_duration):
                # the group starting at i is still below the minimum length, so
                # we need to put it back on the queue.
                queue.append(i)

    ans = []
    cur_group_start = 0
    while cur_group_start < num_utts:
        ans.append( (cur_group_start, group_end[cur_group_start]) )
        cur_group_start = group_end[cur_group_start]
    return ans

def SelfTest():
    assert CombineList(0.1, [5.0, 6.0, 7.0]) == [ (0,1), (1,2), (2,3) ]
    assert CombineList(0.5, [0.1, 6.0, 7.0]) == [ (0,2), (2,3) ]
    assert CombineList(0.5, [6.0, 7.0, 0.1]) == [ (0,1), (1,3) ]
    # in the two examples below, it combines with the shorter one if both would
    # be above min-dur.
    assert CombineList(0.5, [6.0, 0.1, 7.0]) == [ (0,2), (2,3) ]
    assert CombineList(0.5, [7.0, 0.1, 6.0]) == [ (0,1), (1,3) ]
    # in the example below, it combines with whichever one would
    # take it above the min-dur, if there is only one such.
    # note, it tests the 0.1 first as the queue is popped from the end.
    assert CombineList(1.0, [1.0, 0.5, 0.1, 6.0]) == [ (0,2), (2,4) ]

    for x in range(100):
        min_duration = 0.05
        num_utts = randint(1, 15)
        durations = []
        for i in range(num_utts):
            durations.append(0.01 * randint(1, 10))
        ranges = CombineList(min_duration, durations)
        if len(ranges) > 1:  # check that each range's duration is >= min_duration
            for j in range(len(ranges)):
                (start, end) = ranges[j]
                this_dur = sum([ durations[k] for k in range(start, end) ])
                assert not LessThan(this_dur, min_duration)

        # check that the list returned is not affected by very tiny differences
        # in the inputs.
        durations2 = list(durations)
        for i in range(len(durations2)):
            durations2[i] += 1.0e-07 * randint(-5, 5)
        ranges2 = CombineList(min_duration, durations2)
        assert ranges2 == ranges

# This function figures out the grouping of utterances.
# The input is:
# 'min_duration' which is the minimum utterance length in seconds.
# 'merge_within_speakers_only' which is a ['true', 'false'] choice.
# If true, then utterances are only combined if they belong to the same speaker.
# 'spk2utt' which is a list of pairs (speaker-id, [list-of-utterances])
# 'utt2dur' which is a dict from utterance-id to duration (as a float)
# It returns a lists of lists of utterances; each list corresponds to
# a group, e.g.
# [ ['utt1'], ['utt2', 'utt3'] ]
def GetUtteranceGroups(min_duration, merge_within_speakers_only, spk2utt, utt2dur):
    # utt_groups will be a list of lists of utterance-ids formed from the
    # first pass of combination.
    utt_groups = []
    # group_durations will be the durations of the corresponding elements of
    # 'utt_groups'.
    group_durations = []

    # This block calls CombineList for the utterances of each speaker
    # separately, in the 'first pass' of combination.
    for i in range(len(spk2utt)):
        (spk, utts) = spk2utt[i]
        durations = [] # durations for this group of utts.
        for utt in utts:
            try:
                durations.append(utt2dur[utt])
            except:
                sys.exit("choose_utts_to_combine.py: no duration available "
                         "in utt2dur file {0} for utterance {1}".format(
                        args.utt2dur_in, utt))
        ranges = CombineList(min_duration, durations)
        for start, end in ranges:  # each element of 'ranges' is a 2-tuple (start, end)
            utt_groups.append( [ utts[i] for i in range(start, end) ])
            group_durations.append(sum([ durations[i] for i in range(start, end) ]))

    old_dur_sum = sum(utt2dur.values())
    new_dur_sum = sum(group_durations)
    if abs(old_dur_sum - new_dur_sum) > 0.0001 * old_dur_sum:
        print("choose_utts_to_combine.py: large difference in total "
              "durations: {0} vs {1} ".format(old_dur_sum, new_dur_sum),
              file = sys.stderr)

    # Now we combine the groups obtained above, in case we had situations where
    # the combination of all the utterances of one speaker were still below
    # the minimum duration.
    if merge_within_speakers_only == 'true':
      return utt_groups
    else:
      new_utt_groups = []
      ranges = CombineList(min_duration, group_durations)
      for start, end in ranges:
          # the following code is destructive of 'utt_groups' but it doesn't
          # matter.
          this_group = utt_groups[start]
          for i in range(start + 1, end):
              this_group += utt_groups[i]
          new_utt_groups.append(this_group)
      print("choose_utts_to_combine.py: combined {0} utterances to {1} utterances "
            "while respecting speaker boundaries, and then to {2} utterances "
            "with merging across speaker boundaries.".format(
              len(utt2dur), len(utt_groups), len(new_utt_groups)),
            file = sys.stderr)
      return new_utt_groups


SelfTest()

if args.min_duration < 0.0:
    print("choose_utts_to_combine.py: bad minium duration {0}".format(
            args.min_duration))

# spk2utt is a list of 2-tuples (speaker-id, [list-of-utterances])
spk2utt = []
# utt2spk is a dict from speaker-id to utternace-id.
utt2spk = dict()
try:
    f = open(args.spk2utt_in)
except:
    sys.exit("choose_utts_to_combine.py: error opening --spk2utt={0}".format(args.spk2utt_in))
while True:
    line = f.readline()
    if line == '':
        break
    a = line.split()
    if len(a) < 2:
        sys.exit("choose_utts_to_combine.py: bad line in spk2utt file: " + line)
    spk = a[0]
    utts = a[1:]
    spk2utt.append((spk, utts))
    for utt in utts:
        if utt in utt2spk:
            sys.exit("choose_utts_to_combine.py: utterance {0} is listed more than once"
                     "in the spk2utt file {1}".format(utt, args.spk2utt_in))
        utt2spk[utt] = spk
f.close()

# utt2dur is a dict from utterance-id (as a string) to duration in seconds (as a float)
utt2dur = dict()
try:
    f = open(args.utt2dur_in)
except:
    sys.exit("choose_utts_to_combine.py: error opening utt2dur file {0}".format(args.utt2dur_in))
while True:
    line = f.readline()
    if line == '':
        break
    try:
        [ utt, dur ] = line.split()
        dur = float(dur)
        utt2dur[utt] = dur
    except:
        sys.exit("choose_utts_to_combine.py: bad line in utt2dur file {0}: {1}".format(
                args.utt2dur_in, line))


utt_groups = GetUtteranceGroups(args.min_duration, args.merge_within_speakers_only, spk2utt, utt2dur)

# set utt_group names to an array like [ 'utt1', 'utt2-comb2', 'utt4', ... ]
utt_group_names = [ group[0] if len(group)==1 else "{0}-comb{1}".format(group[0], len(group))
                    for group in utt_groups ]


# write the utt2utts file.
try:
    with open(args.utt2utts_out, 'w') as f:
        for i in range(len(utt_groups)):
            print(utt_group_names[i], ' '.join(utt_groups[i]), file = f)
except Exception as e:
    sys.exit("choose_utts_to_combine.py: exception writing to "
             "<utt2utts-out>={0}: {1}".format(args.utt2utts_out, str(e)))

# write the utt2spk file.
try:
    with open(args.utt2spk_out, 'w') as f:
        for i in range(len(utt_groups)):
            utt_group = utt_groups[i]
            spk_list = [ utt2spk[utt] for utt in utt_group ]
            if spk_list == [ spk_list[0] ] * len(utt_group):
                spk = spk_list[0]
            else:
                spk2dur = defaultdict(float)
                # spk2dur is a map from the speaker-id to the duration within this
                # utt, that it comprises.
                for utt in utt_group:
                    spk2dur[utt2spk[utt]] += utt2dur[utt]
                # the following code, which picks the speaker that contributed
                # the most to the duration of this utterance, is a little
                # complex because we want to break ties in a deterministic way
                # picking the earlier spaker in case of a tied duration.
                longest_spk_dur = -1.0
                spk = None
                for this_spk in sorted(spk2dur.keys()):
                    if LessThan(longest_spk_dur, spk2dur[this_spk]):
                        longest_spk_dur = spk2dur[this_spk]
                        spk = this_spk
                assert spk != None
            new_utt = utt_group_names[i]
            print(new_utt, spk, file = f)
except Exception as e:
    sys.exit("choose_utts_to_combine.py: exception writing to "
             "<utt2spk-out>={0}: {1}".format(args.utt2spk_out, str(e)))

# write the utt2dur file.
try:
    with open(args.utt2dur_out, 'w') as f:
        for i in range(len(utt_groups)):
            utt_name = utt_group_names[i]
            duration = sum([ utt2dur[utt] for utt in utt_groups[i]])
            print(utt_name, duration, file = f)
except Exception as e:
    sys.exit("choose_utts_to_combine.py: exception writing to "
             "<utt2dur-out>={0}: {1}".format(args.utt2dur_out, str(e)))

