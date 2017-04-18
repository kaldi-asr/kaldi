#!/usr/bin/python
# -*- encoding: utf-8 -*-
__author__ = 'Feiteng'
import sys
import os
import logging
import threading

from optparse import OptionParser

# 439532__150509__YjEwMWQwMDAwMGYzOTJjNA==__-1__54251c45636d734cc9837700_52ccc52bfcfff2bc45000c78.m4a.pb.raw
# reversed_time/date/user_id/transaction_id/activity_id_sentence_id.format.pb

def parse_raw_line(line):
    splits = line.strip().replace('.m4a.pb.raw', '').split('__')
    assert len(splits) == 4
    (reversed_time, date, transaction_id, speaker, as_id) = splits
    (activity_id, sentence_id) = splits[-1].split('_')
    new_wav_file = '_'.join([speaker, activity_id, sentence_id, reversed_time, date]) + '.wav'
    return (speaker, new_wav_file)


def parse_trans_line(line):
    splits = line.strip().split()
    assert len(splits) >= 2
    (speaker, new_wav_file) = parse_raw_line(splits[0])
    transcription = ' '.join(splits[1:])
    return (new_wav_file, transcription)


def gen_ffmpeg_command(raw_name, new_wav_name, todir):
    ft = "-ac 1 -ar 16000 -f s16le"
    cmd = ["ffmpeg -y -loglevel panic", ft, "-i", raw_name,
           ft, todir + '/' + new_wav_name + ' </dev/null']
    return ' '.join(cmd)


def run(raws_file, trans_file, raw_todir, doc_todir):
    speaker_raws = {}
    speaker_trans = {}
    with open(raws_file) as rf:
        for line in rf:
            line = line.strip()
            (speaker, new_wav_file) = parse_raw_line(line)
            if speaker not in speaker_raws:
                speaker_raws[speaker] = []
                if not os.path.isdir(raw_todir + '/' + speaker):
                    os.mkdir(raw_todir + '/' + speaker)
            speaker_raws[speaker].append(gen_ffmpeg_command(line, new_wav_file, raw_todir))
    with open(trans_file) as tf:
        for line in tf:
            line = line.strip()
            (new_wav_file, transcription) = parse_trans_line(line)
            assert new_wav_file not in speaker_trans
            speaker_trans[new_wav_file] = transcription

    raw_line_num = sum([len(speaker_raws[speaker]) for speaker in speaker_raws])
    print "Find %d raw files, Transcription %d" % (raw_line_num, len(speaker_trans))
    # write raw -> wav
    with open(doc_todir + "/ffmpeg.cmd", 'w') as f:
        for speaker in speaker_raws:
            for line in speaker_raws[speaker]:
                print >>f, line
    # write transcriptions
    with open(doc_todir + "/transcriptions.all", 'w') as f:
        for line in speaker_trans:
            print >>f, line


if __name__ == "__main__":
    usage = '%prog [options] raw.find trans.txt wav_data_dir doc_dir '
    parser = OptionParser(usage)
    parser.add_option('--xxoo', dest='xxoo', default=False)
    (opt, argv) = parser.parse_args()

    if len(argv) != 4:
        print parser.print_help()
        exit(-1)
    run(argv[0], argv[1], argv[2], argv[3])
