#! /usr/bin/env python

# Copyright 2016    Vimal Manohar
# Apache 2.0.

"""This script process a 1995 CSR-IV annotation file and writes to
utt2spk, segments and text files.
"""

from __future__ import print_function
import argparse
import os
import logging
import re
from bs4 import BeautifulSoup
import hub4_utils


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(pathname)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def get_args():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser("Process 1995 CSR-IV HUB4 transcripts")

    parser.add_argument("--noise-word", default="<NOISE>",
                        help="Word to add in-place of noise words")
    parser.add_argument("--spoken-noise-word",
                        default="<SPOKEN_NOISE>",
                        help="Word to add in-place of speaker noise words")
    parser.add_argument("in_file", type=argparse.FileType('r'),
                        help="Input transcript file")
    parser.add_argument("segments_file", type=argparse.FileType('a'),
                        help="Output segments file")
    parser.add_argument("utt2spk_file", type=argparse.FileType('a'),
                        help="Output utt2spk file")
    parser.add_argument("text_file", type=argparse.FileType('a'),
                        help="Output text file")

    args = parser.parse_args()
    return args


class Segment(object):
    """Class to store an utterance (segment)"""

    def __init__(self, reco_id, spk=None, start_time=-1,
                 end_time=-2, text=""):
        """The arguments are straight-forward.
        spk can be None if speaker is not known, in which case the utterance-id
        and speaker-id are made the same.
        end_time can be -1 to mean the end of the recording.
        """
        self.reco_id = reco_id
        self.spk = spk
        self.start_time = float(start_time)
        self.end_time = float(end_time)
        self.text = text

    def get_utt_id(self):
        """Return the utterance-id, which is
        <recording-id>-<start-frame>-<end-frame> if spk is not known.
        Otherwise it is speaker-id is added as a suffix to <recording-id>
        above.
        """
        if self.spk is None:
            return "{reco}-{0:06d}-{1:06d}".format(
                int(self.start_time * 100), int(self.end_time * 100),
                reco=self.reco_id)
        return "{reco}-{spk}-{0:06d}-{1:06d}".format(
            int(self.start_time * 100), int(self.end_time * 100),
            reco=self.reco_id, spk=self.spk)

    def get_spk_id(self):
        """Returns the speaker-id appended to the recording-id, if speaker is
        known. Otherwise returns the utterance-id as speaker-id.
        """
        if self.spk is None:
            return "{reco}-{0:06d}-{1:06d}".format(
                int(self.start_time * 100), int(self.end_time * 100),
                reco=self.reco_id)
        return "{reco}-{spk}".format(reco=self.reco_id, spk=self.spk)

    def write_utt2spk(self, out_file):
        """Writes this segment's entry into utt2spk file."""
        print ("{0} {1}".format(self.get_utt_id(), self.get_spk_id()),
               file=out_file)

    def write_segment(self, out_file):
        """Writes this segment's entry into segments file."""
        print ("{0} {1} {2:.3f} {3:.3f}".format(
                    self.get_utt_id(), self.reco_id,
                    self.start_time, self.end_time),
               file=out_file)

    def write_text(self, out_file):
        """Writes this segment's entry into kaldi text file."""
        print ("{0} {1}".format(self.get_utt_id(), self.text),
               file=out_file)


def write_segments(segments, args):
    """Write segments with non-empty transcripts."""
    for segment in segments:
        if len(segment.text) == 0:
            continue
        segment.write_utt2spk(args.utt2spk_file)
        segment.write_segment(args.segments_file)
        segment.write_text(args.text_file)


def process_text(text, noise_word, spoken_noise_word):
    """Returns normalized text"""
    text = re.sub(r"\[pause\]", "", text)
    text = hub4_utils.normalize_csr_transcript(text, noise_word,
                                               spoken_noise_word)
    return text


test_spk_matcher = re.compile(r"(\S+)\(bt=(\S+)\set=(\S+)\):\s(.+)$")
train_spk_matcher = re.compile(r"(\S+):\s(.+)$")


def process_story_content(args, reco_id, content,
                          start_time, end_time):
    """Process the contents in a story and converts into a set of segments.

    Arguments:
        args -- A reference to the CLI arguments
        reco_id -- Recording id
        content -- A string containing all the contents of a story (or the
                   stuff before the story like the credits and announcements).
                   It is split on a double-newline characters.
        start_time -- Start time of this 'story'.
        end_time -- End time of this 'story'.
    """

    segments = []
    segment_tmp = Segment(reco_id=reco_id, spk=None,
                          start_time=start_time, end_time=-2, text="")

    for line in content.split('\n\n'):
        line = re.sub('\n', ' ', line)

        if len(line) == 0 or re.match(r"\[[^]]+\]$|\s*$", line):
            continue

        m = test_spk_matcher.match(line)
        if m:
            # A line of story in test file that has start and end times
            # and speaker name.
            spk = m.group(1)
            bt = float(m.group(2))
            et = float(m.group(3))

            # Once we know the end-time of the temporary segment, we can
            # write that out (Only if it is non-empty).
            if len(segment_tmp.text) > 0:
                segment_tmp.end_time = bt
            segments.append(segment_tmp)
            segment_tmp = Segment(reco_id, spk=None, start_time=et)

            text = process_text(m.group(4), args.noise_word,
                                args.spoken_noise_word)
            if len(text) == 0 or re.match(r"\[[^]]+\]$|\s*$", text):
                continue
            segments.append(Segment(reco_id=reco_id, spk=spk,
                                    start_time=bt, end_time=et,
                                    text=text))
            continue

        m = train_spk_matcher.match(line)
        if m:
            # A line of story in train file that has no time segment
            # information. So speaker information is not useful.
            text = process_text(m.group(2), args.noise_word,
                                args.spoken_noise_word)
        else:
            # A line of story that does not have a speaker marking.
            text = process_text(line, args.noise_word, args.spoken_noise_word)
        if len(text) == 0 or re.match(r"\[[^]]+\]$|\s*$", text):
            continue
        segment_tmp.text += ' ' + text

    if len(segment_tmp.text) > 0:
        segment_tmp.end_time = end_time
    segments.append(segment_tmp)

    return segments


def process_float(string):
    string = re.sub(r"'|\"", "", string)
    return float(string)


def run(args):
    base = os.path.basename(args.in_file.name)
    reco_id = os.path.splitext(base)[0]

    doc = ''.join(args.in_file.readlines())

    soup = BeautifulSoup(doc, 'lxml')
    for broadcast in soup.find_all('broadcast'):
        non_story_contents = []
        start_time = 0.0
        end_time = -1.0
        for s in broadcast.children:
            try:
                if s.name == 'story':
                    story_begin_time = process_float(s['bt'])
                    story_end_time = process_float(s['et'])
                    for x in s.find_all('language') + s.find_all('sung'):
                        x.replaceWithChildren()
                    if len(non_story_contents):
                        end_time = story_begin_time
                        segments = process_story_content(
                            args, reco_id, ' '.join(non_story_contents),
                            start_time=start_time, end_time=end_time)
                        write_segments(segments, args)
                        non_story_contents = []
                        start_time = story_end_time
                    segments = process_story_content(
                        args, reco_id,
                        ' '.join([str(x) for x in s.children]),
                        start_time=story_begin_time, end_time=story_end_time)
                    write_segments(segments, args)
                elif (s.name is not None and s.name != "language"
                      and s.name != 'sung'):
                    raise RuntimeError(
                        "Expected a NavigableString or <story> "
                        "or <language> or <sung>; got {0}".format(s))
                elif s.name == "language" or s.name == "sung":
                    non_story_contents.append(
                        ' '.join([str(x) for x in s.children]))
                else:
                    non_story_contents.append(str(s))
            except RuntimeError:
                raise
            except Exception:
                logger.error("Failed to process broadcast children %s", s)
                raise
        # End for loop over broadcast children
        if len(non_story_contents) > 0:
            segments = process_story_content(
                args, reco_id, ' '.join(non_story_contents),
                start_time=start_time, end_time=-1)
            write_segments(segments, args)


def main():
    try:
        args = get_args()
        run(args)
    except Exception:
        raise
    finally:
        for f in [args.in_file, args.segments_file,
                  args.utt2spk_file, args.text_file]:
            if f is not None:
                f.close()


if __name__ == '__main__':
    main()
