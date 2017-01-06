# Copyright 2016    Vimal Manohar
# Apache 2.0.

"""This module contains utilities for preparing the HUB4 broadcast news
evaluation corpora.
"""

import sys
import os


def parse_uem_line(reco, line):
    """This method parses a 'line' from the UEM for recording 'reco'
    and returns the line converted to kaldi segments format.
    The format of UEM is
    <file-id> <channel> <start-time> <end-time>

    We force the channel to be 1 and take the file-id to be the recording-id.
    """
    line = line.strip()
    if len(line) == 0 or line[0:2] == ";;":
        continue
    parts = line.split()

    # The channel ID is expected to be 1.
    if parts[1] != "1":
        raise TypeError("Invalid line {0}".format(line))

    start_time = float(parts[2])
    end_time = float(parts[3])

    utt = "{0}-{1:06d}-{2:06d}".format(reco, int(start_time * 100),
                                       int(end_time * 100))
    return "{0} {1} {2} {3}".format(utt, reco, start_time, end_time)


def parse_cmu_seg_line(reco, line):
    """This line parses a 'line' from the CMU automatic segmentation for
    recording 'reco'.
    The CMU segmentation has the following format:
    <file> <channel> <speaker|environment> <start-time> <end-time> <condition>

    We force the channel to be 1 and take the file-id to be the recording-id.
    """
    line = line.strip()
    if len(line) == 0 or line[0:2] == ";;":
        continue
    parts = line.split()

    # Actually a file, but we assuming 1-1 mapping to recording and force
    # channel to be 1.
    reco = parts[0]

    # The channel ID is expected to be 1.
    if parts[1] != "1":
        raise TypeError("Invalid line {0}".format(line))
    spk = parts[2]
    start_time = float(parts[3])
    end_time = float(parts[4])

    utt = "{spk}-{0}-{1:06d}-{2:06d}".format(reco, int(start_time * 100),
                                             int(end_time * 100), spk=spk)

    segment_line = "{0} {1} {st:.3f} {end:.3f}".format(
        utt, reco, st=start_time, end=end_time)
    utt2spk_line = "{0} {1}".format(utt, spk)

    return (segment_line, utt2spk_line)
