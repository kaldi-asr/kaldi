#! /usr/bin/env python

# Copyright 2014  Johns Hopkins University (Authors: Daniel Povey)
#           2014  Vijayaditya Peddinti
#           2016  Vimal Manohar
# Apache 2.0.

"""
Script to combine ctms edits with overlapping segments obtained from
smith-waterman alignment. This script is similar to utils/ctm/resolve_ctm_edits.py,
where the overlapping region is just split in two. The approach here is a
little more advanced since we have access to the WER
(w.r.t. the reference text). It finds the WER of the overlapped region
in the two overlapping segments, and chooses the better one.
"""

from __future__ import print_function
import argparse
import collections
import logging

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

    usage = """ Python script to resolve overlaps in ctms """
    parser = argparse.ArgumentParser(usage)
    parser.add_argument('segments', type=argparse.FileType('r'),
                        help='use segments to resolve overlaps')
    parser.add_argument('ctm_edits_in', type=argparse.FileType('r'),
                        help='input_ctm_file')
    parser.add_argument('ctm_edits_out', type=argparse.FileType('w'),
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
    reco2utt = collections.defaultdict(list)

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


def read_ctm_edits(ctm_edits_file, segments):
    """Read CTM from ctm_edits_file into a dictionary of values indexed by the
    recording.
    It is assumed to be sorted by the recording-id and utterance-id.

    Returns a dictionary {recording : ctm_edit_lines}
        where ctm_lines is a list of lines of CTM corresponding to the
        utterances in the recording.
        The format is as follows:
        [[(utteranceA, channelA, start_time1, duration1, hyp_word1, conf1, ref_word1, edit_type1),
          (utteranceA, channelA, start_time2, duration2, hyp_word2, conf2, ref_word2, edit_type2),
          ...
          (utteranceA, channelA, start_timeN, durationN, hyp_wordN, confN, ref_wordN, edit_typeN)],
         [(utteranceB, channelB, start_time1, duration1, hyp_word1, conf1, ref_word1, edit_type1),
          (utteranceB, channelB, start_time2, duration2, hyp_word2, conf2, ref_word2, edit_type2),
          ...],
         ...
         [...
          (utteranceZ, channelZ, start_timeN, durationN, hyp_wordN, confN, ref_wordN, edit_typeN)]
        ]

    Arguments:
        segments - Dictionary containing the output of read_segments()
            { utterance_id: (recording_id, start_time, end_time) }
    """
    ctm_edits = {}

    num_lines = 0
    for line in ctm_edits_file:
        num_lines += 1
        parts = line.split()

        utt = parts[0]
        reco = segments[utt][0]

        if (reco, utt) not in ctm_edits:
            ctm_edits[(reco, utt)] = []

        ctm_edits[(reco, utt)].append(
            [parts[0], parts[1], float(parts[2]), float(parts[3]),
             parts[4], float(parts[5])] + parts[6:])

    logger.info("Read %d lines from CTM %s", num_lines, ctm_edits_file.name)

    ctm_edits_file.close()
    return ctm_edits


def wer(ctm_edit_lines):
    num_words = 0
    num_incorrect_words = 0
    for line in ctm_edit_lines:
        if line[7] != 'sil':
            num_words += 1
            if line[7] in ['ins', 'del', 'sub']:
                num_incorrect_words += 1
    if num_words == 0 and num_incorrect_words > 0:
        return float('inf')
    if num_words == 0 and num_incorrect_words == 0:
        return 0
    return (float(num_incorrect_words) / num_words, -num_words)


def choose_best_ctm_lines(first_lines, second_lines,
                          window_length, overlap_length):
    """Returns ctm lines that have lower WER. If the WER is the lines with
    the higher number of words is returned.
    """
    i, best_lines = min((0, first_lines), (1, second_lines),
                        key=lambda x: wer(x[1]))

    return i


def resolve_overlaps(ctm_edits, segments):
    """Resolve overlaps within segments of the same recording.

    Returns new lines of CTM for the recording.

    Arguments:
        ctm_edits - The CTM lines for a single recording. This is one value
            stored in the dictionary read by read_ctm(). Assumes that the lines
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
            Expects this to be non-empty.
        segments - Dictionary containing the output of read_segments()
            { utterance_id: (recording_id, start_time, end_time) }
        """
    total_ctm_edits = []
    assert len(ctm_edits) > 0

    # First column of first line in CTM for first utterance
    next_utt = ctm_edits[0][0][0]
    for utt_index, ctm_edits_for_cur_utt in enumerate(ctm_edits):
        if utt_index == len(ctm_edits) - 1:
            break

        if len(ctm_edits_for_cur_utt) == 0:
            next_utt = ctm_edits[utt_index + 1][0][0]
            continue

        cur_utt = ctm_edits_for_cur_utt[0][0]
        if cur_utt != next_utt:
            logger.error(
                "Current utterance %s is not the same as the next "
                "utterance %s in previous iteration.\n"
                "CTM is not sorted by utterance-id?",
                cur_utt, next_utt)
            raise ValueError

        # Assumption here is that the segments are written in
        # consecutive order in time.
        ctm_edits_for_next_utt = ctm_edits[utt_index + 1]
        next_utt = ctm_edits_for_next_utt[0][0]
        if segments[next_utt][1] < segments[cur_utt][1]:
            logger.error(
                "Next utterance %s <= Current utterance %s. "
                "CTM edits is not sorted by utterance-id.",
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

            # find the first word that is in the overlap
            # at the end of the cur utt
            try:
                cur_utt_end_index = next(
                    (i for i, line in enumerate(ctm_edits_for_cur_utt)
                     if line[2] + line[3] / 2.0 > window_length - overlap))
            except StopIteration:
                cur_utt_end_index = len(ctm_edits_for_cur_utt)

            cur_utt_end_lines = ctm_edits_for_cur_utt[cur_utt_end_index:]

            # find the last word that is not in the overlap
            # at the beginning of the next utt
            try:
                next_utt_start_index = next(
                    (i for i, line in enumerate(ctm_edits_for_next_utt)
                     if line[2] + line[3] / 2.0 > overlap))
            except StopIteration:
                next_utt_start_index = 0

            next_utt_start_lines = ctm_edits_for_next_utt[:
                                                          next_utt_start_index]

            choose_index = choose_best_ctm_lines(
                cur_utt_end_lines, next_utt_start_lines,
                window_length, overlap)

            # Ignore the hypotheses beyond this midpoint. They will be
            # considered as part of the next segment.
            if choose_index == 1:
                total_ctm_edits.extend(
                    ctm_edits_for_cur_utt[:cur_utt_end_index])
            else:
                total_ctm_edits.extend(ctm_edits_for_cur_utt)

            if choose_index == 0 and next_utt_start_index > 0:
                # Update the ctm_edits_for_next_utt to include only the lines
                # starting from index.
                ctm_edits[utt_index + 1] = (
                    ctm_edits_for_next_utt[next_utt_start_index:])
            # else leave the ctm_edits as is.
        except:
            logger.error("Could not resolve overlaps between CTM edits for "
                         "%s and %s", cur_utt, next_utt)
            logger.error("Current CTM:")
            for line in ctm_edits_for_cur_utt:
                logger.error(ctm_edit_line_to_string(line))
            logger.error("Next CTM:")
            for line in ctm_edits_for_next_utt:
                logger.error(ctm_edit_line_to_string(line))
            raise

    # merge the last ctm entirely
    total_ctm_edits.extend(ctm_edits[-1])

    return total_ctm_edits


def ctm_edit_line_to_string(line):
    """Converts a line of CTM edit to string."""
    return "{0} {1} {2} {3} {4} {5} {6}".format(line[0], line[1], line[2],
                                                line[3], line[4], line[5],
                                                " ".join(line[6:]))


def write_ctm_edits(ctm_edit_lines, out_file):
    """Writes CTM lines stored in a list to file."""
    for line in ctm_edit_lines:
        print(ctm_edit_line_to_string(line), file=out_file)


def run(args):
    """this method does everything in this script"""
    segments, reco2utt = read_segments(args.segments)
    ctm_edits = read_ctm_edits(args.ctm_edits_in, segments)

    for reco, utts in reco2utt.iteritems():
        ctm_edits_for_reco = []
        for utt in sorted(utts, key=lambda x: segments[x][1]):
            if (reco, utt) in ctm_edits:
                ctm_edits_for_reco.append(ctm_edits[(reco, utt)])
        try:
            if len(ctm_edits_for_reco) == 0:
                logger.warn('CTMs for recording %s is empty.',
                             reco)
                continue   # Go to the next recording

            # Process CTMs in the recordings
            ctm_edits_for_reco = resolve_overlaps(ctm_edits_for_reco, segments)
            write_ctm_edits(ctm_edits_for_reco, args.ctm_edits_out)
        except Exception:
            logger.error("Failed to process CTM edits for recording %s",
                         reco)
            raise
    args.ctm_edits_out.close()
    logger.info("Wrote CTM for %d recordings.", len(ctm_edits))


def main():
    """The main function which parses arguments and call run()."""
    args = get_args()
    try:
        run(args)
    except:
        logger.error("Failed to resolve overlaps", exc_info=True)
        raise RuntimeError
    finally:
        try:
            for f in [args.segments, args.ctm_edits_in, args.ctm_edits_out]:
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
