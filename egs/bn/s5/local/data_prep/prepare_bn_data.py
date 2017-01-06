#! /usr/bin/env python

from __future__ import print_function
import argparse
import glob
import logging
import os
import re
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(pathname)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def get_args():
    parser = argparse.ArgumentParser("Prepare BN corpus.")
    parser.add_argument("--split-at-sync", type=str,
                        choices=["true", "false"], default="false",
                        help="If true, creates separate segments split "
                        "at each sync tag.")
    parser.add_argument("audio_source_dir", type=str,
                        help="Source directory of audio of BN corpus "
                        "(LDC97S44)")
    parser.add_argument("text_source_dir", type=str,
                        help="Source directory of text of BN corpus "
                        "(LDC97T22)")
    parser.add_argument("dir", type=str,
                        help="Output directory to write the kaldi files")

    args = parser.parse_args()

    args.split_at_sync = bool(args.split_at_sync == "true")
    return args


class Segment(object):
    """A class to store a segment with start time, end time, recording id,
    speaker, and the text.
    """
    def __init__(self, reco_id, speaker=None):
        self.reco_id = reco_id
        self.text = None
        self.start_time = -1
        self.end_time = -1
        if speaker is not None:
            self.speaker = speaker
        else:
            self.speaker = reco_id

    def write_segment(self, out_file):
        """writes segment in kaldi segments format"""
        print("{0} {1} {2} {3}".format(self.utt_id(), self.reco_id,
                                       self.start_time, self.end_time),
              file=out_file)

    def write_utt2spk(self, out_file):
        """writes speaker information in kaldi utt2spk format"""
        print("{0} {1}".format(self.utt_id(), self.speaker),
              file=out_file)

    def write_text(self, out_file):
        print("{0} {1}".format(self.utt_id(), self.text),
              file=out_file)

    def check(self):
        """checks if this is a valid segment"""
        assert self.end_time > self.start_time

    def utt_id(self):
        """returns the utterance id created from the recording id and
        the timing information"""
        return ("{spkr}-{0}-{1:06d}-{2:06d}".format(
            self.reco_id, int(self.start_time * 100),
            int(self.end_time * 100), spkr=self.speaker))

    def duration(self):
        """returns the duration of the segment"""
        return self.end_time - self.start_time


def process_segment_soup(reco_id, soup, split_at_sync=False):
    """Processes the input segment soup into a list of objects of class
    Segment.
    If split_at_sync is False, then only a segment is created for the soup
    without consideration to the sync tags.
    """
    start_time = float(soup['s_time'])
    end_time = float(soup['e_time'])
    speaker = soup['speaker']

    segments = []

    create_new_segment = True
    for x in soup.children:
        try:
            if x.name == "sync":
                assert not create_new_segment
                if not split_at_sync:
                    continue
                start_time = float(x['time'])
                segments[-1].end_time = start_time
                create_new_segment = True
            elif x.name == "background" or x.name == "comment":
                continue
            else:
                if create_new_segment:
                    assert split_at_sync or len(segments) == 0
                    segment = Segment(reco_id, speaker)
                    segment.text = x.encode('ascii').strip().replace('\n', ' ')
                    segment.start_time = start_time
                    segment.end_time = end_time
                    if segment.duration() > 0:
                        segments.append(segment)
                    create_new_segment = False
                else:
                    segments[-1].text += (
                        ' ' + x.encode('ascii').strip().replace('\n', ' '))
        except Exception:
            logger.error("Error processing element %s", x)
            raise

    return segments


def process_transcription(transcription_file, segments_handle, utt2spk_handle,
                          text_handle, split_at_sync=False):
    """Processes transcription file into segments."""
    doc = ''.join(open(transcription_file).readlines())
    tag_matcher = re.compile(r"(<(Sync|Background)[^>]+>)")
    doc_modified = tag_matcher.sub(r"\1</\2>", doc)

    soup = BeautifulSoup(doc_modified, 'lxml')

    reco_id, ext = os.path.splitext(os.path.basename(transcription_file))
    reco_id = reco_id.strip('_')  # remove trailing underscores in the name

    for episode in soup.find_all("episode"):
        for section in episode.find_all("section"):
            s_time = section['s_time']
            e_time = section['e_time']
            section_type = section['type']

            logger.debug("Processing section st = %d, end = %d, "
                         "type = %s", s_time, e_time, section_type)

            for seg in section.find_all("segment"):
                try:
                    segments = process_segment_soup(
                        reco_id, seg, split_at_sync=split_at_sync)
                    for s in segments:
                        if s.duration() == 0:
                            continue
                        s.write_segment(segments_handle)
                        s.write_utt2spk(utt2spk_handle)
                        s.write_text(text_handle)
                except Exception:
                    logger.error("Failed processing segment %s", seg)
                    raise


def _run(args):
    if not os.path.isdir(args.dir):
        os.makedirs(args.dir)

    with open(os.path.join(args.dir, "wav.scp"), 'w') as wav_scp_handle:
        for file_ in glob.glob("{0}/{1}/*.sph".format(args.audio_source_dir,
                                                      "data")):
            reco, ext = os.path.splitext(os.path.basename(file_))
            reco = reco.strip('_')

            print("{0} sox {1} -c 1 -r 16000 -t wav - |".format(
                reco, file_), file=wav_scp_handle)

    segments_handle = open(os.path.join(args.dir, "segments"), 'w')
    utt2spk_handle = open(os.path.join(args.dir, "utt2spk"), 'w')
    text_handle = open(os.path.join(args.dir, "text"), 'w')
    for dir_ in glob.glob("{0}/{1}/*/".format(args.text_source_dir,
                                              "hub4_eng_train_trans")):
        for x in glob.glob("{0}/*.txt".format(dir_)):
            try:
                process_transcription(x, segments_handle, utt2spk_handle,
                                      text_handle,
                                      split_at_sync=args.split_at_sync)
            except Exception:
                logger.error("Failed to process file %s",
                             x)
                raise
    segments_handle.close()
    utt2spk_handle.close()
    text_handle.close()


def main():
    try:
        args = get_args()
        _run(args)
    except Exception:
        raise


if __name__ == '__main__':
    main()
