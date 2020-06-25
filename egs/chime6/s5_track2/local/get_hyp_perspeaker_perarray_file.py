#! /usr/bin/env python
# Copyright   2019   Ashish Arora
# Apache 2.0.
"""This script splits a kaldi (text) file
  into per_array per_session per_speaker hypothesis (text) files"""

import argparse
def get_args():
    parser = argparse.ArgumentParser(
        description="""This script splits a kaldi text file
        into per_array per_session per_speaker  text files""")
    parser.add_argument("input_text_path", type=str,
                        help="path of text files")
    parser.add_argument("output_dir_path", type=str,
                        help="Output path for per_array per_session per_speaker reference files")
    args = parser.parse_args()
    return args


def main():
    # S09_U06.ENH-4-704588-704738
    args = get_args()
    sessionid_micid_speakerid_dict= {}
    for line in open(args.input_text_path):
        parts = line.strip().split()
        uttid_id = parts[0]
        temp = uttid_id.strip().split('.')[0]
        micid = temp.strip().split('_')[1]
        speakerid = uttid_id.strip().split('-')[1]
        sessionid = uttid_id.strip().split('_')[0]
        sessionid_micid_speakerid = sessionid + '_' + micid + '_' + speakerid
        if sessionid_micid_speakerid not in sessionid_micid_speakerid_dict:
            sessionid_micid_speakerid_dict[sessionid_micid_speakerid]=list()
        sessionid_micid_speakerid_dict[sessionid_micid_speakerid].append(line)

    for sessionid_micid_speakerid in sorted(sessionid_micid_speakerid_dict):
        hyp_file = args.output_dir_path + '/' + 'hyp' + '_' + sessionid_micid_speakerid
        hyp_writer = open(hyp_file, 'w')
        combined_hyp_file = args.output_dir_path + '/' + 'hyp' + '_' + sessionid_micid_speakerid + '_comb'
        combined_hyp_writer = open(combined_hyp_file, 'w')
        utterances = sessionid_micid_speakerid_dict[sessionid_micid_speakerid]
        # sorting utterances by start and end time
        sessionid_micid_speakerid_utterances={}
        for line in utterances:
            parts = line.strip().split()
            utt_parts = parts[0].strip().split('-')
            time ='-'.join(utt_parts[2:])
            sessionid_micid_speakerid_utterances[time] = line
        text = ''
        for time_key in sorted(sessionid_micid_speakerid_utterances):
            parts = sessionid_micid_speakerid_utterances[time_key].strip().split()
            text = text + ' ' + ' '.join(parts[1:])
            hyp_writer.write(sessionid_micid_speakerid_utterances[time_key])
        combined_utterance = 'utt' + " " + text
        combined_hyp_writer.write(combined_utterance)
        combined_hyp_writer.write('\n')
        combined_hyp_writer.close()
        hyp_writer.close()


if __name__ == '__main__':
    main()

