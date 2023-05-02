#! /usr/bin/env python3

import sys
import os
import re


def normalize_text(text):
    parts = text.strip().split()

    for i, w in enumerate(parts):
        if w in ["<no-speech>", "--", ".", "?", "~"]:
            parts[i] = ""
        elif w == "%incomplete":
            parts[i] = "<unk>"
        elif w in ["<cough>", "<laugh>", "<lipsmack>", "<hes>"]:
            parts[i] = "<spnoise>"
        elif w in ["<breath>", "<sta>"]:
            parts[i] = "<noise>"
        elif w in ["<int>", "(())", "<foreign>", "<overlap>", "<misc>"]:
            parts[i] = "<unk>"

        # change *word* into word
        parts[i] = re.sub(r"^[*](\S+)[*]$", r"\1", parts[i])

    return re.sub(r"\s+", " ", " ".join(parts))


def write_segment(start_time, end_time, text, reco_id,
                  segments_fh, utt2spk_fh, text_fh):
    assert end_time > start_time

    text = normalize_text(text)

    utt_id = "{reco_id}-{st:06d}-{end:06d}".format(
        reco_id=reco_id,
        st=int(start_time * 100), end=int(end_time * 100))

    print ("{utt_id} {reco_id} {st} {end}"
           "".format(utt_id=utt_id, reco_id=reco_id,
                     st=start_time, end=end_time),
           file=segments_fh)
    print ("{utt_id} {reco_id}"
           "".format(utt_id=utt_id, reco_id=reco_id),
           file=utt2spk_fh)
    print ("{utt_id} {text}"
           "".format(utt_id=utt_id, text=text),
           file=text_fh)


def parse_calls_transcript_file(transcript_file, segments_fh,
                                utt2spk_fh, text_fh):
    base_name = os.path.basename(transcript_file)
    file_id = re.sub(".transcription.txt", "", base_name)

    inline_start_time = -1
    outline_start_time = -1

    i = 0

    for line in open(transcript_file):
        parts = line.strip().split()

        if i == 0 and not parts[0].startswith('0'):
            raise Exception("Transcript file {0} does not start with 0.000"
                            "".format(transcript_file))
        i += 1

        start_time = float(parts[0])
        if len(parts) == 1:
            # Last line in the file
            write_segment(inline_start_time, start_time, inline_text, file_id + "_inLine",
                          segments_fh, utt2spk_fh, text_fh)
            write_segment(outline_start_time, start_time, outline_text, file_id + "_outLine",
                          segments_fh, utt2spk_fh, text_fh)
            break

        assert parts[1] in ["inLine", "outLine"]

        if parts[1] == "inLine":
            reco_id = file_id + "_inLine"
            if inline_start_time >= 0:
                write_segment(inline_start_time, start_time, inline_text, reco_id,
                              segments_fh, utt2spk_fh, text_fh)
            inline_text = " ".join(parts[2:])
            inline_start_time = start_time
        else:
            reco_id = file_id + "_outLine"
            if outline_start_time >= 0:
                write_segment(outline_start_time, start_time, outline_text, reco_id,
                              segments_fh, utt2spk_fh, text_fh)
            outline_text = " ".join(parts[2:])
            outline_start_time = start_time


def parse_non_calls_transcript_file(transcript_file, segments_fh,
                                    utt2spk_fh, text_fh):
    base_name = os.path.basename(transcript_file)
    file_id = re.sub(".transcription.txt", "", base_name)

    start_time = -1
    i = 0

    with open(transcript_file) as fh:
        line = fh.readline().strip()
        if not line.startswith('['):
            raise Exception("Transcript file {0} does not start with [0.000"
                            "".format(transcript_file))
        try:
            start_time  = float(re.sub(r"\[([^\]]+)\]", r"\1", line))
        except Exception:
            print("Could not parse line {0}".format(line), file=sys.stderr)
            raise

        text = fh.readline()
        while text != '':
            text = text.strip()
            line = fh.readline().strip()
            if not line.startswith('['):
                raise Exception("Time-stamp in transcript file {0} does not start with [; error parsing line {1} after text {2}"
                                "".format(transcript_file, line, text))
            try:
                end_time  = float(re.sub(r"\[([^\]]+)\]", r"\1", line))
            except Exception:
                print("Could not parse line {0}".format(line), file=sys.stderr)
                raise

            write_segment(start_time, end_time, text, file_id,
                          segments_fh, utt2spk_fh, text_fh)
            start_time = end_time
            text = fh.readline()


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print ("Usage: {0} <corpus-root-dir> <calls-list> <non-calls-list> <data-dir>",
               file=sys.stderr)
        raise SystemExit(1)

    root_path = sys.argv[1]
    calls_list = open(sys.argv[2]).readlines()
    non_calls_list = open(sys.argv[3]).readlines()
    data_dir = sys.argv[4]

    wav_scp_fh = open("{0}/wav.scp".format(data_dir), 'w')
    utt2spk_fh = open("{0}/utt2spk".format(data_dir), 'w')
    reco2file_and_channel_fh = open(
        "{0}/reco2file_and_channel".format(data_dir), 'w')
    text_fh = open("{0}/text".format(data_dir), 'w')
    segments_fh = open("{0}/segments".format(data_dir), 'w')

    for line in calls_list:
        file_id = line.strip()
        transcript_file = (
            "{root_path}/transcription/{file_id}.transcription.txt"
            "".format(root_path=root_path, file_id=file_id))
        wav_file = "{root_path}/src/{file_id}.wav".format(
            root_path=root_path, file_id=file_id)

        for channel in [1, 2]:
            reco_id = file_id + ("_inLine" if channel == 1 else "_outLine")
            print ("{reco_id} {file_id} {channel}"
                   "".format(reco_id=reco_id, file_id=file_id,
                             channel="A" if channel == 1 else "B"),
                   file=reco2file_and_channel_fh)
            print ("{reco_id} sox {wav_file} -r 8000 -b 16 -c 1 -t wav - remix {channel} |"
                   "".format(reco_id=reco_id, wav_file=wav_file, channel=channel),
                   file=wav_scp_fh)

        parse_calls_transcript_file(transcript_file, segments_fh,
                                    utt2spk_fh, text_fh)

    for line in non_calls_list:
        file_id = line.strip()
        transcript_file = (
            "{root_path}/transcription/{file_id}.transcription.txt"
            "".format(root_path=root_path, file_id=file_id))
        wav_file = "{root_path}/src/{file_id}.wav".format(
            root_path=root_path, file_id=file_id)

        print ("{file_id} {file_id} 1"
               "".format(file_id=file_id),
               file=reco2file_and_channel_fh)
        print ("{reco_id} sox {wav_file} -r 8000 -b 16 -c 1 -t wav - |"
               "".format(reco_id=file_id, wav_file=wav_file),
               file=wav_scp_fh)

        parse_non_calls_transcript_file(transcript_file, segments_fh,
                                        utt2spk_fh, text_fh)

    wav_scp_fh.close()
    utt2spk_fh.close()
    reco2file_and_channel_fh.close()
    text_fh.close()
    segments_fh.close()
