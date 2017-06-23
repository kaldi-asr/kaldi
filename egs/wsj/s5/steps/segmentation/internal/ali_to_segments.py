#! /usr/bin/env python

# Copyright 2017  Vimal Manohar
# Apache 2.0

"""
This script converts alignments into kaldi segments and utt2spk.
The label 1 is for silence and label 2 is for speech.
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
    This script converts alignments into kaldi segments and utt2spk.
    The label 1 is for silence and label 2 is for speech.""",
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--verbose", type=int, choices=[0, 1, 2, 3],
                        default=0, help="Higher verbosity for more logging")

    parser.add_argument("--sad-map", type=str,
                        help="File containing mapping from alignment labels "
                             "to SAD labels -- "
                             "1 for non-speech and 2 for speech")
    parser.add_argument("--utt2dur", type=str,
                        help="File containing durations of utterances.")
    parser.add_argument("--frame-shift", type=float, default=0.01,
                        help="Frame shift to convert frame indexes to time")

    parser.add_argument("--segment-padding", type=float, default=0.2,
                        help="Additional padding on speech segments")
    parser.add_argument("--max-intersegment-duration", type=float, default=0.3,
                        help="""Merge speech segments if the
                        intersegment silence is less than this duration""")
    parser.add_argument("--min-segment-duration", type=float, default=0.3,
                        help="""Remove segments that are smaller than this
                        duration.""")
    parser.add_argument("--max-segment-duration", type=float, default=10.0,
                        help="""If segment is longer than this duration, then
                        split it into overlapping segments.""")
    parser.add_argument("--overlap-duration", type=float, default=1.0,
                        help="""Overlap duration when splitting segments.""")
    parser.add_argument("--max-remaining-duration", type=float, default=2.0,
                        help="""If the duration of segment remaining after
                        splitting is smaller than this amount then the
                        segment is not split.""")


    parser.add_argument("in_alignments", type=str,
                        help="Input file containing alignments in "
                        "text archive format")
    parser.add_argument("out_segments", type=str,
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
        self.num_initial_segments = 0
        self.initial_duration = 0.0
        self.padding_duration = 0.0
        self.num_segments_after_merging = 0
        self.num_segments_removed = 0
        self.num_segments_split = 0
        self.num_final_segments = 0
        self.final_duration = 0.0

    def add(self, other):
        """Adds stats from another object"""
        self.num_initial_segments += other.num_initial_segments
        self.initial_duration += other.initial_duration
        self.padding_duration = other.padding_duration
        self.num_segments_after_merging = other.num_segments_after_merging
        self.num_segments_removed = other.num_segments_removed
        self.num_segments_split = other.num_segments_split
        self.num_final_segments = other.num_final_segments
        self.final_duration = other.final_duration

    def __str__(self):
        return ("num-initial-segments={num_initial_segments}, "
                "initial-duration={initial_duration}, "
                "padding-duration={padding_duration}, "
                "num-segments-after-merging={num_segments_after_merging}, "
                "num-segments-removed={num_segments_removed}, "
                "num-segments-split={num_segments_split}, "
                "num-final-segments={num_final_segments}, "
                "final-duration={final_duration}".format(
                    num_initial_segments=self.num_initial_segments,
                    initial_duration=self.initial_duration,
                    padding_duration=self.padding_duration,
                    num_segments_after_merging=self.num_segments_after_merging,
                    num_segments_removed=self.num_segments_removed,
                    num_segments_split=self.num_segments_split,
                    num_final_segments=self.num_final_segments,
                    final_duration=self.final_duration))


def process_label(text_label, sad_map=None):
    """Processes an input integer label and returns a 1 or 2,
    where 1 is for silence and 2 is for speech.

    Arguments:
        text_label -- input label (must be integer)
        sad_map -- if provided must be a dictionary mapping the input integer
                   label to 1 or 2.
    """
    prev_label = int(text_label)
    if sad_map is not None:
        try:
            prev_label = sad_map[prev_label]
        except KeyError as e:
            logger.error("Label %d was not seen in --sad-map",
                         prev_label)
            raise e
    if prev_label not in [1, 2]:
        raise ValueError("Expecting label to 1 (non-speech) or 2 (speech); "
                         "got {0}".format(prev_label))

    return prev_label


class Segmentation(object):
    """Stores segmentation for an utterances"""
    def __init__(self, segmenter_stats=None):
        self.segments = None
        self.stats = (segmenter_stats if segmenter_stats is None
                      else SegmenterStats())

    def initialize_segments(self, alignment, frame_shift=0.01, sad_map=None):
        """Initializes segments from input alignment."""
        self.segments = []

        assert len(alignment) > 0

        prev_label = None
        prev_length = 0
        for i, text_label in enumerate(alignment):
            if prev_label is not None and int(text_label) != prev_label:
                if prev_label == 2:
                    self.segments.append(
                        [float(i - prev_length) * frame_shift,
                         float(i) * frame_shift, prev_label])

                prev_label = process_label(text_label, sad_map)
                prev_length = 0
                self.stats.initial_duration += (prev_length * frame_shift)
            elif prev_label is None:
                prev_label = process_label(text_label, sad_map)

            prev_length += 1

        if prev_length > 0 and prev_label == 2:
            self.segments.append(
                [float(len(alignment) - prev_length) * frame_shift,
                 float(len(alignment)) * frame_shift, prev_label])
            self.stats.initial_duration += (prev_length * frame_shift)

        self.stats.num_initial_segments = len(self.segments)

    def pad_speech_segments(self, segment_padding, max_duration=float("inf")):
        """Pads segments by duration 'segment_padding' on either sides, but
        ensures that the segments don't go beyond the neighboring segments
        or the duration of the utterance 'max_duration'."""
        for i, segment in enumerate(self.segments):
            assert segment[2] == 2, segment
            segment[0] -= segment_padding
            self.stats.padding_duration += segment_padding
            if segment[0] < 0.0:
                self.stats.padding_duration += segment[0]
                segment[0] = 0.0
            if i > 1 and self.segments[i-1][1] > segment[0]:
                self.stats.padding_duration -= (
                    self.segments[i-1][1] - segment[0])
                segment[0] = self.segments[i-1][1]

            segment[1] += segment_padding
            self.stats.padding_duration += segment_padding
            if segment[1] >= max_duration:
                self.stats.padding_duration -= (segment[1] - max_duration)
                segment[1] = max_duration
            if (i + 1 < len(self.segments)
                    and segment[1] > self.segments[i+1][0]):
                self.stats.padding_duration -= (
                    segment[1] - self.segments[i+1][0])
                segment[1] = self.segments[i+1][0]

    def merge_adjacent_segments(self, max_intersegment_duration):
        """Merges speech segments that are closer than
        'max_intersegment_duration' apart.
        """
        new_segments = []

        segment_start = -1
        segment_end = -1

        for i, segment in enumerate(self.segments):
            if i == 0:
                segment_start = segment[0]
                segment_end = segment[1]
            elif segment[0] - self.segments[i-1][1] < max_intersegment_duration:
                logger.debug("Merging segment {0} and {1}"
                             "".format(to_str(self.segments[i-1]),
                                       to_str(segment)))
                segment_end = segment[1]
            else:
                new_segments.append((segment_start, segment_end, 2))
                segment_start = segment[0]
                segment_end = segment[1]

        if segment_start > -1:
            new_segments.append((segment_start, segment_end, 2))

        self.segments = new_segments
        self.stats.num_segments_after_merging = len(self.segments)

    def remove_short_segments(self, min_segment_duration):
        """Removes short segments that are shorter than 'min_segment_duration'
        long"""
        new_segments = []

        for segment in self.segments:
            if segment[1] - segment[0] < min_segment_duration:
                logger.debug("Removing segment {0}".format(to_str(segment)))
            else:
                new_segments.append(segment)

        self.segments = new_segments
        self.stats.num_segments_removed = (
            self.stats.num_segments_after_merging - len(self.segments))

    def split_long_segments(self, max_segment_duration, overlap_duration,
                            max_remaining_duration):
        """Splits segments that are longer than
        'max_segment_duration + max_remaining_duration' seconds long and
        creates overlapping segments of length 'max_segment_duration' and
        overlap 'overlap_duration'.
        Splitting is stopped if the final chunk created would be smaller than
        'max_remaining_duration'.
        """
        new_segments = []

        for segment in self.segments:
            segment_start = segment[0]
            dur = segment[1] - segment[0]
            if dur > max_segment_duration + max_remaining_duration:
                self.stats.num_segments_split += 1

            while dur > max_segment_duration + max_remaining_duration:
                new_segments.append(
                    (segment_start,
                     segment_start + max_segment_duration, 2))
                segment_start += max_segment_duration - overlap_duration
                dur -= (max_segment_duration - overlap_duration)
            new_segments.append((segment_start, segment_start + dur))

        self.segments = new_segments
        self.stats.num_final_segments = len(self.segments)
        self.stats.final_duration = sum([x[1] - x[0] for x in self.segments])

    def write(self, key, file_handle):
        """Write segments to file"""
        if global_verbose >= 2:
            logger.info("For key {key}, got stats {stats}".format(
                key=key, stats=self.stats))
        for segment in self.segments:
            seg_id = "{key}-{st:07d}-{end:07d}".format(
                key=key, st=int(segment[0] * 100), end=int(segment[1] * 100))
            print ("{seg_id} {key} {st:.2f} {end:.2f}".format(
                seg_id=seg_id, key=key, st=segment[0], end=segment[1]),
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

    sad_map = {}
    if args.sad_map is not None:
        with common_lib.smart_open(args.sad_map) as sad_map_fh:
            for line in sad_map_fh:
                parts = line.strip().split()
                if len(parts) != 2:
                    raise RuntimeError("Unable to parse line '{0}' in {1}"
                                       "".format(line.strip(), sad_map_fh))
                if int(parts[1]) not in [1, 2]:
                    raise ValueError("Expecting the second field in {0} to be "
                                     "1 or 2".format(sad_map_fh))
                sad_map[parts[0]] = int(parts[1])

    global_stats = SegmenterStats()
    with common_lib.smart_open(args.in_alignments) as in_alignments_fh, \
            common_lib.smart_open(args.out_segments, 'w') as out_segments_fh:
        for line in in_alignments_fh:
            parts = line.strip().split()
            utt_id = parts[0]

            if len(parts) < 2:
                raise RuntimeError("Unable to parse line '{0}' in {1}"
                                   "".format(line.strip(),
                                             in_alignments_fh))

            utt_stats = SegmenterStats()
            segmentation = Segmentation(utt_stats)
            segmentation.initialize_segments(
                parts[1:], args.frame_shift,
                sad_map=sad_map if args.sad_map is not None else None)
            segmentation.pad_speech_segments(args.segment_padding,
                                             None if args.utt2dur is None
                                             else utt2dur[utt_id])
            segmentation.merge_adjacent_segments(args.max_intersegment_duration)
            segmentation.remove_short_segments(args.min_segment_duration)
            segmentation.split_long_segments(
                args.max_segment_duration, args.overlap_duration,
                args.max_remaining_duration)
            segmentation.write(utt_id, out_segments_fh)
            global_stats.add(utt_stats)
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
