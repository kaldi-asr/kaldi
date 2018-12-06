#!/usr/bin/env python3

""" This script reads image and transcription mapping and creates the following files :text, utt2spk, images.scp.
  Eg. local/process_waldo_data.py lines/hyp_line_image_transcription_mapping_kaldi.txt data/test
  Eg. text file: LDC0001_000404_NHR_ARB_20070113.0052_11_LDC0001_00z2 ﻮﺠﻫ ﻮﻌﻘﻟ ﻍﺍﺮﻗ ﺢﺗّﻯ ﺎﻠﻨﺧﺎﻋ
      utt2spk file: LDC0001_000397_NHR_ARB_20070113.0052_11_LDC0001_00z1 LDC0001
      images.scp file: LDC0009_000000_arb-NG-2-76513-5612324_2_LDC0009_00z0
      data/local/lines/1/arb-NG-2-76513-5612324_2_LDC0009_00z0.tif
"""

import argparse
import os
import sys

parser = argparse.ArgumentParser(description="Creates text, utt2spk and images.scp files",
                                 epilog="E.g.  " + sys.argv[0] + " data/train data/local/lines ",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('image_transcription_file', type=str,
                    help='Path to the file containing line image path and transcription information')
parser.add_argument('out_dir', type=str,
                    help='directory location to write output files.')
args = parser.parse_args()


def read_image_text(image_text_path):
    """ Given the file path containing, mapping information of line image
     and transcription, it returns a dict. The dict contains this mapping
    info. It can be accessed via line_id and will provide transcription.
    Returns:
    --------
    dict: line_id and transcription mapping
    """
    image_transcription_dict = dict()
    with open(image_text_path, encoding='utf-8') as f:
        for line in f:
            line_vect = line.strip().split(' ')
            image_path = line_vect[0]
            line_id = os.path.basename(image_path).split('.png')[0]
            transcription = line_vect[1:]
            joined_transcription = list()
            for word in transcription:
                joined_transcription.append(word)
            joined_transcription = " ".join(joined_transcription)
            image_transcription_dict[line_id] = joined_transcription
    return image_transcription_dict


### main ###
print("Processing '{}' data...".format(args.out_dir))
text_file = os.path.join(args.out_dir, 'text')
text_fh = open(text_file, 'w', encoding='utf-8')
utt2spk_file = os.path.join(args.out_dir, 'utt2spk')
utt2spk_fh = open(utt2spk_file, 'w', encoding='utf-8')
image_file = os.path.join(args.out_dir, 'images.scp')
image_fh = open(image_file, 'w', encoding='utf-8')

image_transcription_dict = read_image_text(args.image_transcription_file)
for line_id in sorted(image_transcription_dict.keys()):
        writer_id = line_id.strip().split('_')[-3]
        updated_line_id = line_id + '.png'
        image_file_path = os.path.join('lines', updated_line_id)
        text = image_transcription_dict[line_id]
        utt_id = line_id
        text_fh.write(utt_id + ' ' + text + '\n')
        utt2spk_fh.write(utt_id + ' ' + writer_id + '\n')
        image_fh.write(utt_id + ' ' + image_file_path + '\n')

