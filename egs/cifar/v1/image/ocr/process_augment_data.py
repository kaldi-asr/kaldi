#!/usr/bin/env python3

# Copyright   2017 Yiwen Shao
#             2018 Ashish Arora

# Apache 2.0
# This script reads the feats.scp file and regenerate images.scp, text,
# utt2spk and spk2utt. Apart from original images.scp, text, utt2spk new 
# values were created due to augmentation (doubles the data). Hence to 
# map newly created feats, to text, images.scp and utt2spk this script is used.

import os
import argparse

parser = argparse.ArgumentParser(description="""Regenerate images.scp, text,
                    utt2spk and spk2utt from feats.scp for augment data.""")
parser.add_argument('dir', type=str,
                    help='directory of images.scp')
args = parser.parse_args()

text_file = os.path.join(args.dir, 'backup', 'text')
text_dict = dict()  # stores image_id and text

with open(text_file) as text_fh:
    for uttid_text in text_fh:
        uttid_text = uttid_text.strip()
        uttid_text_vect = uttid_text.split(" ")
        uttid = uttid_text_vect[0]
        image_id = uttid
        text_vect = uttid_text_vect[1:]
        text = " ".join(text_vect)
        text_dict[image_id] = text

utt2spk_file = os.path.join(args.dir, 'backup', 'utt2spk')
uttid_to_spk = dict()  # stores image_id and speaker

with open(utt2spk_file) as utt2spk_fh:
    for uttid_spk in utt2spk_fh:
        uttid_spk = uttID_spk.strip()
        uttid_spk_vect = uttid_spk.split(" ")
        uttid = uttid_spk_vect[0]
        image_id = uttid
        spk = uttid_spk_vect[1]
        uttid_to_spk[image_id] = spk

image_file = os.path.join(args.dir, 'backup', 'images.scp')
uttID_path_dict = dict()  # stores image_id and image path

with open(image_file) as image_fh:
    for uttid_path in image_fh:
        uttid_path = uttid_path.strip()
        uttid_path_vect = uttid_path.split(" ")
        uttid = uttid_path_vect[0]
        image_id = uttid
        path = uttid_path_vect[1]
        uttid_path_dict[image_id] = path


image_file = os.path.join(args.dir + '/', 'images.scp')
image_fh = open(image_file, 'w')

text_file = os.path.join(args.dir + '/', 'text')
text_fh = open(text_file, 'w')

utt2spk_file = os.path.join(args.dir + '/', 'utt2spk')
utt2spk_fh = open(utt2spk_file, 'w')

feats_scp_file = os.path.join(args.dir, 'feats.scp')
with open(feats_scp_file) as feats_scp_fh:
    for uttid_image_path in feats_scp_fh:
        uttid_image_path = uttID_image_path.strip()
        uttid_path_vect = uttID_image_path.split(" ")
        uttid = uttid_path_vect[0]
        imageid = "_".join(uttid.split("_")[:-1])
        text_fh.write(uttid + ' ' + text_dict[imageid] + '\n')
        utt2spk_fh.write(uttid + ' ' + uttid_spk_dict[imageid] + '\n')
        image_fh.write(uttid + ' ' + uttid_path_dict[imageid] + '\n')

