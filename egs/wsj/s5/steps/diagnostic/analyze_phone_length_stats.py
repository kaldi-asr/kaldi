#!/usr/bin/env python


# Copyright 2016 Johns Hopkins University (author: Daniel Povey)
# Apache 2.0.

from __future__ import print_function
import argparse
import sys, os
from collections import defaultdict


parser = argparse.ArgumentParser(description="This script reads stats created in analyze_alignments.sh "
                                 "to print information about phone lengths in alignments.  It's principally "
                                 "useful in order to see whether there is a reasonable amount of silence "
                                 "at the beginning and ends of segments.  The normal output of this script "
                                 "is written to the standard output and is human readable (on crashes, "
                                 "we'll print an error to stderr.")

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
    sys.exit("analyze_phone_length_stats.py: error opening or reading {0}/phones.txt".format(
            args.lang))
# this is a special case... for begin- and end-of-sentence stats,
# we group all nonsilence phones together.
phone_int2text[0] = 'nonsilence'


# populate the set 'nonsilence', which will contain the integer phone-ids of
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
    f.close()
    for silence_phone in line.split(":"):
        nonsilence.remove(int(silence_phone))
except Exception as e:
    sys.exit("analyze_phone_length_stats.py: error processing {0}/phones/silence.csl: {1}".format(
            args.lang, str(e)))


# phone_length is a dict of dicts of dicts;
# phone_lengths[boundary_type] for boundary_type in [ 'begin', 'end', 'all' ] is
# a dict indexed by phone, containing dicts from length to a count of occurrences.
# Phones are ints and lengths are integers representing numbers of frames.
# So: count == phone_lengths[boundary_type][phone][length].
# note: for the 'begin' and 'end' boundary-types, we group all nonsilence phones
# into phone-id zero.
phone_lengths = dict()
for boundary_type in [ 'begin', 'end', 'all' ]:
    phone_lengths[boundary_type] = dict()
    for p in phone_int2text.keys():
        phone_lengths[boundary_type][p] = defaultdict(int)

# total_phones is a dict from boundary_type to total count [of phone occurrences]
total_phones = defaultdict(int)
# total_frames is a dict from boundary_type to total number of frames.
total_frames = defaultdict(int)
# total_frames is a dict from num-frames to count of num-utterances with that
# num-frames.

while True:
    line = sys.stdin.readline()
    if line == '':
        break
    a = line.split()
    if len(a) != 4:
        sys.exit("analyze_phone_length_stats.py: reading stdin, could not interpret line: " + line)
    try:
        count, boundary_type, phone, length = a
        total_phones[boundary_type] += int(count)
        total_frames[boundary_type] += int(count) * int(length)
        phone_lengths[boundary_type][int(phone)][int(length)] += int(count)
        if int(phone) in nonsilence:
            nonsilence_phone = 0
            phone_lengths[boundary_type][nonsilence_phone][int(length)] += int(count)
    except Exception as e:
        sys.exit("analyze_phone_length_stats.py: unexpected phone {0} "
                 "seen (lang directory mismatch?): {1}".format(phone, str(e)))

if len(phone_lengths) == 0:
    sys.exit("analyze_phone_length_stats.py: read no input")

# work out the optional-silence phone
try:
    f = open(args.lang + "/phones/optional_silence.int", "r")
    optional_silence_phone = int(f.readline())
    optional_silence_phone_text = phone_int2text[optional_silence_phone]
    f.close()
    if optional_silence_phone in nonsilence:
        print("analyze_phone_length_stats.py: was expecting the optional-silence phone to "
              "be a member of the silence phones, it is not.  This script won't work correctly.")
except:
    largest_count = 0
    optional_silence_phone = 1
    for p in phone_int2text.keys():
        if p > 0 and not p in nonsilence:
            this_count = sum([ l * c for l,c in phone_lengths['all'][p].items() ])
            if this_count > largest_count:
                largest_count = this_count
                optional_silence_phone = p
    optional_silence_phone_text = phone_int2text[optional_silence_phone]
    print("analyze_phone_length_stats.py: could not get optional-silence phone from "
          "{0}/phones/optional_silence.int, guessing that it's {1} from the stats. ".format(
            args.lang, optional_silence_phone_text))



# If length_to_count is a map from length-in-frames to count,
# return the length-in-frames that equals the (fraction * 100)'th
# percentile of the distribution.
def GetPercentile(length_to_count, fraction):
    total_phones = sum(length_to_count.values())
    if total_phones == 0:
        return 0
    else:
        items = sorted(length_to_count.items())
        count_cutoff = int(fraction * total_phones)
        cur_count_total = 0
        for length,count in items:
            assert count >= 0
            cur_count_total += count
            if cur_count_total >= count_cutoff:
                return length
        assert false # we shouldn't reach here.

def GetMean(length_to_count):
    total_phones = sum(length_to_count.values())
    if total_phones == 0:
        return 0.0
    total_frames = sum([ float(l * c) for l,c in length_to_count.items() ])
    return total_frames / total_phones


# Analyze frequency, median and mean of optional-silence at beginning and end of utterances.
# The next block will print something like
#  "At utterance begin, SIL is seen 15.0% of the time; when seen, duration (median, mean) is (5, 7.6) frames."
#  "At utterance end, SIL is seen 14.6% of the time; when seen, duration (median, mean) is (4, 6.1) frames."


# This block will print warnings if silence is seen less than 80% of the time at utterance
# beginning and end.
for boundary_type in 'begin', 'end':
    phone_to_lengths = phone_lengths[boundary_type]
    num_utterances = total_phones[boundary_type]
    assert num_utterances > 0
    opt_sil_lengths = phone_to_lengths[optional_silence_phone]
    frequency_percentage = sum(opt_sil_lengths.values()) * 100.0 / num_utterances
    # The reason for this warning is that the tradition in speech recognition is
    # to supply a little silence at the beginning and end of utterances... up to
    # maybe half a second.  If your database is not like this, you should know;
    # you may want to mess with the segmentation to add more silence.
    if frequency_percentage < 80.0:
        print("analyze_phone_length_stats.py: WARNING: optional-silence {0} is seen only {1}% "
              "of the time at utterance {2}.  This may not be optimal.".format(
                optional_silence_phone_text, frequency_percentage, boundary_type))



# this will control a sentence that we print..
boundary_to_text = { }
boundary_to_text['begin'] = 'At utterance begin'
boundary_to_text['end'] = 'At utterance end'
boundary_to_text['all'] = 'Overall'

# the next block prints lines like (to give some examples):
# At utterance begin, SIL accounts for 98.4% of phone occurrences, with duration (median, mean, 95-percentile) is (57,59.9,113) frames.
# ...
# At utterance end, nonsilence accounts for 4.2% of phone occurrences, with duration (median, mean, 95-percentile) is (13,13.3,22) frames.
# ...
# Overall, R_I accounts for 3.2% of phone occurrences, with duration (median, mean, 95-percentile) is (6,6.9,12) frames.

for boundary_type in 'begin', 'end', 'all':
    phone_to_lengths = phone_lengths[boundary_type]
    tot_num_phones = total_phones[boundary_type]
    # sort the phones in decreasing order of count.
    for phone,lengths in sorted(phone_to_lengths.items(), key = lambda x : -sum(x[1].values())):
        frequency_percentage = sum(lengths.values()) * 100.0 / tot_num_phones
        if frequency_percentage < args.frequency_cutoff_percentage:
            continue

        duration_median = GetPercentile(lengths, 0.5)
        duration_percentile_95 = GetPercentile(lengths, 0.95)
        duration_mean = GetMean(lengths)

        text = boundary_to_text[boundary_type]  # e.g. 'At utterance begin'.
        try:
            phone_text = phone_int2text[phone]
        except:
            sys.exit("analyze_phone_length_stats.py: phone {0} is not covered on phones.txt "
                     "(lang/alignment mismatch?)".format(phone))
        print("{text}, {phone_text} accounts for {percent}% of phone occurrences, with "
              "duration (median, mean, 95-percentile) is ({median},{mean},{percentile95}) frames.".format(
                text = text, phone_text = phone_text,
                percent = "%.1f" % frequency_percentage,
                median = duration_median, mean = "%.1f" % duration_mean,
                percentile95 = duration_percentile_95))


## Print stats on frequency and average length of word-internal optional-silences.
## For optional-silence only, subtract the begin and end-utterance stats from the 'all'
## stats, to get the stats excluding initial and final phones.
total_frames['internal'] = total_frames['all'] - total_frames['begin'] - total_frames['end']
total_phones['internal'] = total_phones['all'] - total_phones['begin'] - total_phones['end']

internal_opt_sil_phone_lengths = dict(phone_lengths['all'][optional_silence_phone])
# internal_opt_sil_phone_lenghts is a dict from length to count.
for length in list(internal_opt_sil_phone_lengths.keys()):
    # subtract the counts for begin and end from the overall counts to get the
    # word-internal count.
    internal_opt_sil_phone_lengths[length] -= (phone_lengths['begin'][optional_silence_phone][length] +
                                               phone_lengths['end'][optional_silence_phone][length])
    if internal_opt_sil_phone_lengths[length] == 0:
        del internal_opt_sil_phone_lengths[length]

if total_phones['internal'] != 0.0:
    total_internal_optsil_frames = sum([ float(l * c) for l,c in internal_opt_sil_phone_lengths.items() ])
    total_optsil_frames = sum([ float(l * c)
                                for l,c in phone_lengths['all'][optional_silence_phone].items() ])
    opt_sil_internal_frame_percent = total_internal_optsil_frames * 100.0 / total_frames['internal']
    opt_sil_total_frame_percent = total_optsil_frames * 100.0 / total_frames['all']
    internal_frame_percent = total_frames['internal'] * 100.0 / total_frames['all']

    print("The optional-silence phone {0} occupies {1}% of frames overall ".format(
            optional_silence_phone_text, "%.1f" % opt_sil_total_frame_percent))
    hours_total = total_frames['all'] / 360000.0;
    hours_nonsil = (total_frames['all'] - total_optsil_frames) / 360000.0
    print("Limiting the stats to the {0}% of frames not covered by an utterance-[begin/end] phone, "
          "optional-silence {1} occupies {2}% of frames.".format("%.1f" % internal_frame_percent,
                                                                 optional_silence_phone_text,
                                                                 "%.1f" % opt_sil_internal_frame_percent))
    print("Assuming 100 frames per second, the alignments represent {0} hours of data, "
          "or {1} hours if {2} frames are excluded.".format(
            "%.1f" % hours_total, "%.1f" % hours_nonsil, optional_silence_phone_text))

    opt_sil_internal_phone_percent = (sum(internal_opt_sil_phone_lengths.values()) *
                                      100.0 / total_phones['internal'])
    duration_median = GetPercentile(internal_opt_sil_phone_lengths, 0.5)
    duration_mean = GetMean(internal_opt_sil_phone_lengths)
    duration_percentile_95 = GetPercentile(internal_opt_sil_phone_lengths, 0.95)
    print("Utterance-internal optional-silences {0} comprise {1}% of utterance-internal phones, with duration "
          "(median, mean, 95-percentile) = ({2},{3},{4})".format(
                optional_silence_phone_text, "%.1f" % opt_sil_internal_phone_percent,
                duration_median, "%0.1f" % duration_mean, duration_percentile_95))
