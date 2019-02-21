#!/usr/bin/env python


# Copyright 2016 Johns Hopkins University (author: Daniel Povey)
# Apache 2.0.

from __future__ import print_function
from __future__ import division
import argparse
import sys, os
from collections import defaultdict


parser = argparse.ArgumentParser(description="This script reads stats created in analyze_lats.sh "
                                 "to print information about lattice depths broken down per phone. "
                                 "The normal output of this script is written to the standard output "
                                 "and is human readable (on crashes, we'll print an error to stderr.")

parser.add_argument("--frequency-cutoff-percentage", type = float,
                    default = 0.5, help="Cutoff, expressed as a percentage "
                    "(between 0 and 100), of frequency at which we print stats "
                    "for a phone.")

parser.add_argument("lang",
                    help="Language directory, e.g. data/lang.")

args = parser.parse_args()

# set up phone_int2text to map from phone to printed form.
phone_int2text = {}
try:
    f = open(args.lang + "/phones.txt", "r");
    for line in f.readlines():
        [ word, number] = line.split()
        phone_int2text[int(number)] = word
    f.close()
except:
    sys.exit("analyze_lattice_depth_stats.py: error opening or reading {0}/phones.txt".format(
            args.lang))
# this is a special case... for begin- and end-of-sentence stats,
# we group all nonsilence phones together.
phone_int2text[0] = 'nonsilence'

# populate the set and 'nonsilence', which will contain the integer phone-ids of
# nonsilence phones (and disambig phones, which won't matter).
nonsilence = set(phone_int2text.keys())
nonsilence.remove(0)
try:
    # open lang/phones/silence.csl-- while there are many ways of obtaining the
    # silence/nonsilence phones, we read this because it's present in graph
    # directories as well as lang directories.
    filename = "{0}/phones/silence.csl".format(args.lang)
    f = open(filename, "r")
    line = f.readline()
    for silence_phone in line.split(":"):
        nonsilence.remove(int(silence_phone))
    f.close()
except Exception as e:
    sys.exit("analyze_lattice_depth_stats.py: error processing {0}/phones/silence.csl: {1}".format(
            args.lang, str(e)))

# phone_depth_counts is a dict of dicts.
# for each integer phone-id 'phone',
# phone_depth_counts[phone] is a map from depth to count (of frames on which
# that was the 1-best phone in the alignment, and the lattice depth
# had that value).  So we'd access it as
# count = phone_depth_counts[phone][depth].

phone_depth_counts = dict()

# note: -1 is for all phones put in one bucket.
for p in [ -1 ] + list(phone_int2text.keys()):
    phone_depth_counts[p] = defaultdict(int)

total_frames = 0

while True:
    line = sys.stdin.readline()
    if line == '':
        break
    a = line.split()
    if len(a) != 3:
        sys.exit("analyze_lattice_depth_stats.py: reading stdin, could not interpret line: " + line)
    try:
        phone, depth, count = [ int(x) for x in a ]

        phone_depth_counts[phone][depth] += count
        total_frames += count
        if phone in nonsilence:
            nonsilence_phone = 0
            phone_depth_counts[nonsilence_phone][depth] += count
        universal_phone = -1
        phone_depth_counts[universal_phone][depth] += count
    except Exception as e:
        sys.exit("analyze_lattice_depth_stats.py: unexpected phone {0} "
                 "seen (lang directory mismatch?): line is {1}, error is {2}".format(phone, line, str(e)))

if total_frames == 0:
    sys.exit("analyze_lattice_depth_stats.py: read no input")


# If depth_to_count is a map from depth-in-frames to count,
# return the depth-in-frames that equals the (fraction * 100)'th
# percentile of the distribution.
def GetPercentile(depth_to_count, fraction):
    this_total_frames = sum(depth_to_count.values())
    if this_total_frames == 0:
        return 0
    else:
        items = sorted(depth_to_count.items())
        count_cutoff = int(fraction * this_total_frames)
        cur_count_total = 0
        for depth,count in items:
            assert count >= 0
            cur_count_total += count
            if cur_count_total >= count_cutoff:
                return depth
        assert false # we shouldn't reach here.

def GetMean(depth_to_count):
    this_total_frames = sum(depth_to_count.values())
    if this_total_frames == 0:
        return 0.0
    this_total_depth = sum([ float(l * c) for l,c in depth_to_count.items() ])
    return this_total_depth / this_total_frames


print("The total amount of data analyzed assuming 100 frames per second "
      "is {0} hours".format("%.1f" % (total_frames / 360000.0)))

# the next block prints lines like (to give some examples):
# Nonsilence phones as a group account for 74.4% of phone occurrences, with lattice depth (10,50,90-percentile)=(1,2,7) and mean=3.1
# Phone SIL accounts for 25.5% of phone occurrences, with lattice depth (10,50,90-percentile)=(1,1,4) and mean=2.5
# Phone Z_E accounts for 2.5% of phone occurrences, with lattice depth (10,50,90-percentile)=(1,2,6) and mean=2.9
# ...


# sort the phones in decreasing order of count.
for phone,depths in sorted(phone_depth_counts.items(), key = lambda x : -sum(x[1].values())):

    frequency_percentage = sum(depths.values()) * 100.0 / total_frames
    if frequency_percentage < args.frequency_cutoff_percentage:
        continue


    depth_percentile_10 = GetPercentile(depths, 0.1)
    depth_percentile_50 = GetPercentile(depths, 0.5)
    depth_percentile_90 = GetPercentile(depths, 0.9)
    depth_mean = GetMean(depths)

    if phone > 0:
        try:
            phone_text = phone_int2text[phone]
        except:
            sys.exit("analyze_lattice_depth_stats.py: phone {0} is not covered on phones.txt "
                     "(lang/alignment mismatch?)".format(phone))
        preamble = "Phone {phone_text} accounts for {percent}% of frames, with".format(
            phone_text = phone_text, percent = "%.1f" % frequency_percentage)
    elif phone == 0:
        preamble = "Nonsilence phones as a group account for {percent}% of frames, with".format(
            percent = "%.1f" % frequency_percentage)
    else:
        assert phone == -1
        preamble = "Overall,";

    print("{preamble} lattice depth (10,50,90-percentile)=({p10},{p50},{p90}) and mean={mean}".format(
            preamble = preamble,
            p10 = depth_percentile_10,
            p50 = depth_percentile_50,
            p90 = depth_percentile_90,
            mean = "%.1f" % depth_mean))
