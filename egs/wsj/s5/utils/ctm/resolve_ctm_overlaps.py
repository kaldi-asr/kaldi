#! /usr/bin/env python

# Copyright 2014  Johns Hopkins University (Authors: Daniel Povey)
#           2014  Vijayaditya Peddinti
#           2016  Vimal Manohar
# Apache 2.0.

"""
Script to combine ctms with overlapping segments.
The current approach is very simple. It ignores the words,
which are hypothesized in the half of the overlapped region
that is closer to the utterance boundary.
So if there are two segments
in the region 0s to 30s and 25s to 55s, with overlap of 5s,
the last 2.5s of the first utterance i.e. from 27.5s to 30s is truncated
and the first 2.5s of the second utterance i.e. from 25s to 27.s is truncated.
"""

from __future__ import print_function
from __future__ import division
import argparse
import collections
import logging

from collections import defaultdict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s [%(pathname)s:%(lineno)s - '
    '%(funcName)s - %(levelname)s ] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def get_args():
    """gets command line arguments"""

    usage = """ Python script to resolve overlaps in ctms.  May be used with
                utils/data/subsegment_data_dir.sh. """
    parser = argparse.ArgumentParser(usage)
    parser.add_argument('segments', type=argparse.FileType('r'),
                        help='use segments to resolve overlaps')
    parser.add_argument('ctm_in', type=argparse.FileType('r'),
                        help='input_ctm_file')
    parser.add_argument('ctm_out', type=argparse.FileType('w'),
                        help='output_ctm_file')
    parser.add_argument('--verbose', type=int, default=0,
                        help="Higher value for more verbose logging.")
    args = parser.parse_args()

    if args.verbose > 2:
        logger.setLevel(logging.DEBUG)
        handler.setLevel(logging.DEBUG)

    return args


def read_segments(segments_file):
    """Read from segments and returns two dictionaries,
    {utterance-id: (recording_id, start_time, end_time)}
    {recording_id: list-of-utterances}
    """
    segments = {}
    reco2utt = defaultdict(list)

    num_lines = 0
    for line in segments_file:
        num_lines += 1
        parts = line.strip().split()
        assert len(parts) in [4, 5]
        segments[parts[0]] = (parts[1], float(parts[2]), float(parts[3]))
        reco2utt[parts[1]].append(parts[0])

    logger.info("Read %d lines from segments file %s",
                num_lines, segments_file.name)
    segments_file.close()

    return segments, reco2utt


def read_ctm(ctm_file, segments):
    """Read CTM from ctm_file into a dictionary of values indexed by the
    recording.
    It is assumed to be sorted by the recording-id and utterance-id.

    Returns a dictionary {recording : ctm_lines}
        where ctm_lines is a list of lines of CTM corresponding to the
        utterances in the recording.
        The format is as follows:
        [[(utteranceA, channelA, start_time1, duration1, hyp_word1, conf1),
          (utteranceA, channelA, start_time2, duration2, hyp_word2, conf2),
          ...
          (utteranceA, channelA, start_timeN, durationN, hyp_wordN, confN)],
         [(utteranceB, channelB, start_time1, duration1, hyp_word1, conf1),
          (utteranceB, channelB, start_time2, duration2, hyp_word2, conf2),
          ...],
         ...
         [...
          (utteranceZ, channelZ, start_timeN, durationN, hyp_wordN, confN)]
        ]
    """
    ctms = {}

    num_lines = 0
    for line in ctm_file:
        num_lines += 1
        parts = line.split()

        utt = parts[0]
        reco = segments[utt][0]

        if (reco, utt) not in ctms:
            ctms[(reco, utt)] = []

        ctms[(reco, utt)].append([parts[0], parts[1], float(parts[2]),
                                  float(parts[3])] + parts[4:])

    logger.info("Read %d lines from CTM %s", num_lines, ctm_file.name)

    ctm_file.close()
    return ctms


def resolve_overlaps(ctms, segments):
    """Resolve overlaps within segments of the same recording.

    Returns new lines of CTM for the recording.

    Arguments:
        ctms - The CTM lines for a single recording. This is one value stored
            in the dictionary read by read_ctm(). Assumes that the lines
            are sorted by the utterance-ids.
            The format is the following:
            [[(utteranceA, channelA, start_time1, duration1, hyp_word1, conf1),
              (utteranceA, channelA, start_time2, duration2, hyp_word2, conf2),
              ...
              (utteranceA, channelA, start_timeN, durationN, hyp_wordN, confN)
             ],
             [(utteranceB, channelB, start_time1, duration1, hyp_word1, conf1),
              (utteranceB, channelB, start_time2, duration2, hyp_word2, conf2),
              ...],
             ...
             [...
              (utteranceZ, channelZ, start_timeN, durationN, hyp_wordN, confN)]
            ]
        segments - Dictionary containing the output of read_segments()
            { utterance_id: (recording_id, start_time, end_time) }
        """
    total_ctm = []
    if len(ctms) == 0:
        raise RuntimeError('CTMs for recording is empty. '
                           'Something wrong with the input ctms')

    # First column of first line in CTM for first utterance
    next_utt = ctms[0][0][0]
    for utt_index, ctm_for_cur_utt in enumerate(ctms):
        if utt_index == len(ctms) - 1:
            break

        if len(ctm_for_cur_utt) == 0:
            next_utt = ctms[utt_index + 1][0][0]
            continue

        cur_utt = ctm_for_cur_utt[0][0]
        if cur_utt != next_utt:
            logger.error(
                "Current utterance %s is not the same as the next "
                "utterance %s in previous iteration.\n"
                "CTM is not sorted by utterance-id?",
                cur_utt, next_utt)
            raise ValueError

        # Assumption here is that the segments are written in
        # consecutive order?
        ctm_for_next_utt = ctms[utt_index + 1]
        next_utt = ctm_for_next_utt[0][0]
        if segments[next_utt][1] < segments[cur_utt][1]:
            logger.error(
                "Next utterance %s <= Current utterance %s. "
                "CTM is not sorted by start-time of utterance-id.",
                next_utt, cur_utt)
            raise ValueError

        try:
            # length of this utterance
            window_length = segments[cur_utt][2] - segments[cur_utt][1]

            # overlap of this segment with the next segment
            # i.e. current_utterance_end_time - next_utterance_start_time
            # Note: It is possible for this to be negative when there is
            # actually no overlap between consecutive segments.
            try:
                overlap = segments[cur_utt][2] - segments[next_utt][1]
            except KeyError:
                logger("Could not find utterance %s in segments",
                       next_utt)
                raise

            if overlap > 0 and segments[next_utt][2] <= segments[cur_utt][2]:
                # Next utterance is entirely within this utterance.
                # So we leave this ctm as is and make the next one empty.
                total_ctm.extend(ctm_for_cur_utt)
                ctms[utt_index + 1] = []
                continue

            # find a break point (a line in the CTM) for the current utterance
            # i.e. the first line that has more than half of it outside
            # the first half of the overlap region.
            # Note: This line will not be included in the output CTM, which is
            # only upto the line before this.
            try:
                index = next(
                    (i for i, line in enumerate(ctm_for_cur_utt)
                     if (line[2] + line[3] / 2.0
                         > window_length - overlap / 2.0)))
            except StopIteration:
                # It is possible for such a word to not exist, e.g the last
                # word in the CTM is longer than overlap length and starts
                # before the beginning of the overlap.
                # or the last word ends before the middle of the overlap.
                index = len(ctm_for_cur_utt)

            # Ignore the hypotheses beyond this midpoint. They will be
            # considered as part of the next segment.
            total_ctm.extend(ctm_for_cur_utt[:index])

            # Find a break point (a line in the CTM) for the next utterance
            # i.e. the first line that has more than half of it outside
            # the first half of the overlap region.
            try:
                index = next(
                    (i for i, line in enumerate(ctm_for_next_utt)
                    if line[2] + line[3] / 2.0 > overlap / 2.0))
            except StopIteration:
                # This can happen if there is no word hypothesized after
                # half the overlap region.
                ctms[utt_index + 1] = []
                continue

            if index > 0:
                # Update the ctm_for_next_utt to include only the lines
                # starting from index.
                ctms[utt_index + 1] = ctm_for_next_utt[index:]
            # else leave the ctm as is.
        except:
            logger.error("Could not resolve overlaps between CTMs for "
                         "%s and %s", cur_utt, next_utt)
            logger.error("Current CTM:")
            for line in ctm_for_cur_utt:
                logger.error(ctm_line_to_string(line))
            logger.error("Next CTM:")
            for line in ctm_for_next_utt:
                logger.error(ctm_line_to_string(line))
            raise

    # merge the last ctm entirely
    total_ctm.extend(ctms[-1])

    return total_ctm


def ctm_line_to_string(line):
    """Converts a line of CTM to string."""
    return "{0} {1} {2} {3} {4}".format(line[0], line[1], line[2], line[3],
                                        " ".join(line[4:]))


def write_ctm(ctm_lines, out_file):
    """Writes CTM lines stored in a list to file."""
    for line in ctm_lines:
        print(ctm_line_to_string(line), file=out_file)


def run(args):
    """this method does everything in this script"""
    segments, reco2utt = read_segments(args.segments)
    ctms = read_ctm(args.ctm_in, segments)

    for reco, utts in reco2utt.items():
        ctms_for_reco = []
        for utt in sorted(utts, key=lambda x: segments[x][1]):
            if (reco, utt) in ctms:
                ctms_for_reco.append(ctms[(reco, utt)])
        if len(ctms_for_reco) == 0:
            logger.info("CTM for recording {0} was empty".format(reco))
            continue
        try:
            # Process CTMs in the recordings
            ctms_for_reco = resolve_overlaps(ctms_for_reco, segments)
            write_ctm(ctms_for_reco, args.ctm_out)
        except Exception:
            logger.error("Failed to process CTM for recording %s",
                         reco)
            raise
    args.ctm_out.close()
    logger.info("Wrote CTM for %d recordings.", len(ctms))


def main():
    """The main function which parses arguments and call run()."""
    args = get_args()
    try:
        run(args)
    except:
        logger.error("Failed to resolve overlaps", exc_info=True)
        raise SystemExit(1)
    finally:
        try:
            for f in [args.segments, args.ctm_in, args.ctm_out]:
                if f is not None:
                    f.close()
        except IOError:
            logger.error("Could not close some files. "
                         "Disk error or broken pipes?")
            raise
        except UnboundLocalError:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
