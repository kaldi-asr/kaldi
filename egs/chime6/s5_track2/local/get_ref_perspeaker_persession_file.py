#! /usr/bin/env python
# Copyright   2019   Ashish Arora
# Apache 2.0.
"""This script splits a kaldi (text) file
  into per_speaker per_session reference (text) file"""

import argparse

def get_args():
    parser = argparse.ArgumentParser(
        description="""This script splits a kaldi text file
        into per_speaker per_session text files""")
    parser.add_argument("input_text_path", type=str,
                        help="path of text file")
    parser.add_argument("output_dir_path", type=str,
                        help="Output path for per_session per_speaker reference files")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    sessionid_speakerid_dict= {}
    spkrid_mapping = {}
    for line in open(args.input_text_path):
        parts = line.strip().split()
        uttid_id = parts[0]
        speakerid = uttid_id.strip().split('_')[0]
        sessionid = uttid_id.strip().split('_')[1]
        sessionid_speakerid = sessionid + '_' + speakerid
        if sessionid_speakerid not in sessionid_speakerid_dict:
            sessionid_speakerid_dict[sessionid_speakerid]=list()
        sessionid_speakerid_dict[sessionid_speakerid].append(line)

    spkr_num = 1
    prev_sessionid = ''
    for sessionid_speakerid in sorted(sessionid_speakerid_dict):
        spkr_id = sessionid_speakerid.strip().split('_')[1]
        curr_sessionid = sessionid_speakerid.strip().split('_')[0]
        if prev_sessionid != curr_sessionid:
            prev_sessionid = curr_sessionid
            spkr_num = 1
        if spkr_id not in spkrid_mapping:
            spkrid_mapping[spkr_id] = spkr_num
            spkr_num += 1

    for sessionid_speakerid in sorted(sessionid_speakerid_dict):
        ref_file = args.output_dir_path + '/ref_' + sessionid_speakerid.split('_')[0] + '_' + str(
            spkrid_mapping[sessionid_speakerid.split('_')[1]])
        ref_writer = open(ref_file, 'w')
        wc_file = args.output_dir_path + '/ref_wc_' + sessionid_speakerid.split('_')[0] + '_' + str(
            spkrid_mapping[sessionid_speakerid.split('_')[1]])
        wc_writer = open(wc_file, 'w')
        combined_ref_file = args.output_dir_path + '/ref_' + sessionid_speakerid.split('_')[0] + '_' + str(
            spkrid_mapping[sessionid_speakerid.split('_')[1]]) + '_comb'
        combined_ref_writer = open(combined_ref_file, 'w')
        utterances = sessionid_speakerid_dict[sessionid_speakerid]
        sessionid_speakerid_utterances = {}
        # sorting utterances by start and end time
        for line in utterances:
            parts = line.strip().split()
            utt_parts = parts[0].strip().split('-')
            time ='-'.join(utt_parts[1:])
            sessionid_speakerid_utterances[time] = line
        text = ''
        uttid_wc = 'utt'
        for time_key in sorted(sessionid_speakerid_utterances):
            parts = sessionid_speakerid_utterances[time_key].strip().split()
            uttid_id = parts[0]
            utt_text = ' '.join(parts[1:])
            text = text + ' ' + ' '.join(parts[1:])
            ref_writer.write(sessionid_speakerid_utterances[time_key])
            length = str(len(utt_text.split()))
            uttid_id_len = uttid_id + ":" + length
            uttid_wc = uttid_wc + ' ' + uttid_id_len
        combined_utterance = 'utt' + " " + text
        combined_ref_writer.write(combined_utterance)
        combined_ref_writer.write('\n')
        combined_ref_writer.close()
        wc_writer.write(uttid_wc)
        wc_writer.write('\n')
        wc_writer.close()
        ref_writer.close()

if __name__ == '__main__':
    main()
