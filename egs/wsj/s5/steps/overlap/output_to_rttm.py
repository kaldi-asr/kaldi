#!/usr/bin/env python

# Copyright 2017  Vimal Manohar
#           2018  Capital One (Author: Zhiyuan Guan)
# Apache 2.0

"""
This script converts frame-level overlap detector marks (in kaldi
integer vector text archive format) into kaldi segments and utt2spk.
The input integer vectors are expected to contain '1' for silence frames,
'2' for speech frames of single speaker, and '3' for overlap frames.
"""

from __future__ import print_function
import argparse
import logging
import sys

sys.path.insert(0, 'steps')
import libs.common as common_lib

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(pathname)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

global_verbose = 0


def get_args():
    parser = argparse.ArgumentParser(
        description="""
This script converts frame-level speech activity detection marks (in kaldi
integer vector text archive format) into kaldi segments and utt2spk.
The input integer vectors are expected to contain 1 for silence frames
and 2 for speech frames.
""",
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--verbose", type=int, choices=[0, 1, 2, 3],
                        default=0, help="Higher verbosity for more logging")

    parser.add_argument("--utt2dur", type=str,
                        help="File containing durations of utterances.")

    parser.add_argument("--frame-shift", type=float, default=0.01,
                        help="Frame shift to convert frame indexes to time")

    parser.add_argument("--segment-padding", type=float, default=0.2,
                        help="Additional padding on speech segments. But we "
                             "ensure that the padding does not go beyond the "
                             "adjacent segment.")

    parser.add_argument("--min-segment-dur", type=float, default=0,
                        help="Minimum duration (in seconds) required for a segment "
                             "to be included. This is before any padding. Segments "
                             "shorter than this duration will be removed.")

    parser.add_argument("--merge-consecutive-max-dur", type=float, default=0,
                        help="Merge consecutive segments as long as the merged "
                             "segment is no longer than this many seconds. The segments "
                             "are only merged if their boundaries are touching. "
                             "This is after padding by --segment-padding seconds."
                             "0 means do not merge. Use 'inf' to not limit the duration.")

    parser.add_argument("--region-type", type=str, default="overlap",
                        help="Specify if overlap or single-speaker or silence region "
                        "to output in the rttm")

    parser.add_argument("in_ovl", type=str,
                        help="Input file containing alignments in "
                             "text archive format")

    parser.add_argument("out_rttm", type=str,
                        help="Output kaldi segments file")

    args = parser.parse_args()

    global global_verbose
    global_verbose = args.verbose

    logger.info("Setting verbosity to {0}".format(global_verbose))

    if args.verbose >= 3:
        logger.setLevel(logging.DEBUG)
        handler.setLevel(logging.DEBUG)
    return args


def to_str(segment):
    assert len(segment) == 3
    return "[{0:.3f}, {1:.3f}, {2}]".format(segment[0], segment[1],
                                            segment[2])


class SegmenterStats(object):
    """Stores stats about the post-process stages"""

    def __init__(self):
        self.num_segments_initial = 0
        self.num_short_segments_filtered = 0
        self.num_merges = 0
        self.num_segments_final = 0
        self.initial_duration = 0.0
        self.padding_duration = 0.0
        self.filter_short_duration = 0.0
        self.final_duration = 0.0

    def add(self, other):
        """Adds stats from another object"""
        self.num_segments_initial += other.num_segments_initial
        self.num_short_segments_filtered += other.num_short_segments_filtered
        self.num_merges += other.num_merges
        self.num_segments_final += other.num_segments_final
        self.initial_duration += other.initial_duration
        self.filter_short_duration += other.filter_short_duration
        self.padding_duration += other.padding_duration
        self.final_duration += other.final_duration

    def __str__(self):
        return ("num-segments-initial={num_segments_initial}, "
                "num-short-segments-filtered={num_short_segments_filtered}, "
                "num-merges={num_merges}, "
                "num-segments-final={num_segments_final}, "
                "initial-duration={initial_duration}, "
                "filter-short-duration={filter_short_duration}, "
                "padding-duration={padding_duration}, "
                "final-duration={final_duration}".format(
            num_segments_initial=self.num_segments_initial,
            num_short_segments_filtered=self.num_short_segments_filtered,
            num_merges=self.num_merges,
            num_segments_final=self.num_segments_final,
            initial_duration=self.initial_duration,
            filter_short_duration=self.filter_short_duration,
            padding_duration=self.padding_duration,
            final_duration=self.final_duration))


def process_label(text_label):
    """Processes an input integer label and returns a 1, 2 or 3,
    where 1 is for silence, 2 is for single-speaker region, and 
    3 is for overlap.

    Arguments:
        text_label -- input label (must be integer)
    """
    prev_label = int(text_label)
    if prev_label not in [1, 2, 3]:
        raise ValueError("Expecting label to be 0 (silence), 1 (single speaker) or 2 (overlap); "
                         "got {}".format(prev_label))

    return prev_label


class Segmentation(object):
    """Stores segmentation for an utterances"""

    region_to_label = {'silence':1, 'single':2, 'overlap':3}

    def __init__(self, region_type):
        self.segments = None
        self.label = self.region_to_label[region_type]
        self.region_type = region_type
        self.stats = SegmenterStats()

    def initialize_segments(self, alignment, frame_shift=0.01):
        """Initializes segments from input alignment.
        The alignment is frame-level overlap detection marks,
        each of which must be 1, 2, or 3."""
        self.segments = []

        assert len(alignment) > 0

        prev_label = None
        prev_length = 0
        for i, text_label in enumerate(alignment):
            if prev_label is not None and int(text_label) != prev_label:
                if prev_label == self.label:
                    self.segments.append(
                        [float(i - prev_length) * frame_shift,
                         float(i) * frame_shift, prev_label])
                    self.stats.initial_duration += (prev_length * frame_shift)
                prev_label = process_label(text_label)
                prev_length = 0
            elif prev_label is None:
                prev_label = process_label(text_label)

            prev_length += 1

        if prev_length > 0 and prev_label == self.label:
            self.segments.append(
                [float(len(alignment) - prev_length) * frame_shift,
                 float(len(alignment)) * frame_shift, prev_label])
            self.stats.initial_duration += (prev_length * frame_shift)

        self.stats.num_segments_initial = len(self.segments)
        self.stats.num_segments_final = len(self.segments)
        self.stats.final_duration = self.stats.initial_duration

    def filter_short_segments(self, min_dur):
        """Filters out segments with durations shorter than 'min_dur'."""
        if min_dur <= 0:
            return

        segments_kept = []
        for segment in self.segments:
            assert segment[2] == self.label, segment
            dur = segment[1] - segment[0]
            if dur < min_dur:
                self.stats.filter_short_duration += dur
                self.stats.num_short_segments_filtered += 1
            else:
                segments_kept.append(segment)
        self.segments = segments_kept
        self.stats.num_segments_final = len(self.segments)
        self.stats.final_duration -= self.stats.filter_short_duration

    def pad_segments(self, segment_padding, max_duration=float("inf")):
        """Pads segments by duration 'segment_padding' on either sides, but
        ensures that the segments don't go beyond the neighboring segments
        or the duration of the utterance 'max_duration'."""
        if max_duration == None:
            max_duration = float("inf")
        for i, segment in enumerate(self.segments):
            assert segment[2] == self.label, segment
            segment[0] -= segment_padding  # try adding padding on the left side
            self.stats.padding_duration += segment_padding
            if segment[0] < 0.0:
                # Padding takes the segment start to before the beginning of the utterance.
                # Reduce padding.
                self.stats.padding_duration += segment[0]
                segment[0] = 0.0
            if i >= 1 and self.segments[i - 1][1] > segment[0]:
                # Padding takes the segment start to before the end the previous segment.
                # Reduce padding.
                self.stats.padding_duration -= (
                        self.segments[i - 1][1] - segment[0])
                segment[0] = self.segments[i - 1][1]

            segment[1] += segment_padding
            self.stats.padding_duration += segment_padding
            if segment[1] >= max_duration:
                # Padding takes the segment end beyond the max duration of the utterance.
                # Reduce padding.
                self.stats.padding_duration -= (segment[1] - max_duration)
                segment[1] = max_duration
            if (i + 1 < len(self.segments)
                    and segment[1] > self.segments[i + 1][0]):
                # Padding takes the segment end beyond the start of the next segment.
                # Reduce padding.
                self.stats.padding_duration -= (
                        segment[1] - self.segments[i + 1][0])
                segment[1] = self.segments[i + 1][0]
        self.stats.final_duration += self.stats.padding_duration

    def merge_consecutive_segments(self, max_dur):
        """Merge consecutive segments (happens after padding), provided that
        the merged segment is no longer than 'max_dur'."""
        if max_dur <= 0 or not self.segments:
            return

        merged_segments = [self.segments[0]]
        for segment in self.segments[1:]:
            assert segment[2] == self.label, segment
            if segment[0] == merged_segments[-1][1] and \
                    segment[1] - merged_segments[-1][0] <= max_dur:
                # The segment starts at the same time the last segment ends,
                # and the merged segment is shorter than 'max_dur'.
                # Extend the previous segment.
                merged_segments[-1][1] = segment[1]
                self.stats.num_merges += 1
            else:
                merged_segments.append(segment)

        self.segments = merged_segments
        self.stats.num_segments_final = len(self.segments)

    def write(self, key, file_handle):
        """Write segments to RTTM file"""
        if global_verbose >= 2:
            logger.info("For key {key}, got stats {stats}".format(
                key=key, stats=self.stats))
        rttm_str = "SPEAKER {0} 1 {1:7.3f} {2:7.3f} <NA> <NA> {3} <NA> <NA>"
        for segment in self.segments:
            print(rttm_str.format(key, segment[0], segment[1] - segment[0], self.region_type),
                file=file_handle)


def run(args):
    """The main function that does everything."""
    utt2dur = {}
    if args.utt2dur is not None:
        with common_lib.smart_open(args.utt2dur) as utt2dur_fh:
            for line in utt2dur_fh:
                parts = line.strip().split()
                if len(parts) != 2:
                    raise RuntimeError("Unable to parse line '{0}' in {1}"
                                       "".format(line.strip(), args.utt2dur))
                utt2dur[parts[0]] = float(parts[1])

    global_stats = SegmenterStats()
    with common_lib.smart_open(args.in_ovl) as in_ovl_fh, \
            common_lib.smart_open(args.out_rttm, 'w') as out_rttm_fh:
        for line in in_ovl_fh:
            parts = line.strip().split()
            utt_id = parts[0]

            if len(parts) < 2:
                raise RuntimeError("Unable to parse line '{0}' in {1}"
                                   "".format(line.strip(),
                                             in_ovl_fh))

            segmentation = Segmentation(args.region_type)
            segmentation.initialize_segments(
                parts[1:], args.frame_shift)
            segmentation.filter_short_segments(args.min_segment_dur)
            segmentation.pad_segments(args.segment_padding,
                                             None if args.utt2dur is None
                                             else utt2dur[utt_id])
            segmentation.merge_consecutive_segments(args.merge_consecutive_max_dur)
            segmentation.write(utt_id, out_rttm_fh)
            global_stats.add(segmentation.stats)
    logger.info(global_stats)


def main():
    """Parses arguments and calls the run method"""
    args = get_args()
    try:
        run(args)
    except Exception:
        raise


if __name__ == '__main__':
    main()
