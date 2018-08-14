#!/usr/bin/env python3
import os
import argparse

parser = argparse.ArgumentParser(
    description="""Regenerate images.scp, text, utt2spk and spk2utt from feats.scp for augment data""")
parser.add_argument(
    'dir', type=str, help='directory of images.scp')
args = parser.parse_args()


text_file = os.path.join(args.dir, 'backup', 'text')
text_dict = dict()  # stores imageID and text

with open(text_file) as text_fh:
    for uttID_text in text_fh:
        uttID_text = uttID_text.strip()
        uttID_text_vect = uttID_text.split(" ")
        uttID = uttID_text_vect[0]
        imageID = uttID
        text_vect = uttID_text_vect[1:]
        text = " ".join(text_vect)
        text_dict[imageID] = text
        # print "%s: %s" % (imageID, text)

utt2spk_file = os.path.join(args.dir, 'backup', 'utt2spk')
uttID_spk_dict = dict()  # stores imageID and speaker

with open(utt2spk_file) as utt2spk_fh:
    for uttID_spk in utt2spk_fh:
        uttID_spk = uttID_spk.strip()
        uttID_spk_vect = uttID_spk.split(" ")
        uttID = uttID_spk_vect[0]
        imageID = uttID
        spk = uttID_spk_vect[1]
        uttID_spk_dict[imageID] = spk
        # print "%s: %s" % (imageID, spk)

image_file = os.path.join(args.dir, 'backup', 'images.scp')
uttID_path_dict = dict()  # stores imageID and image path

with open(image_file) as image_fh:
    for uttID_path in image_fh:
        uttID_path = uttID_path.strip()
        uttID_path_vect = uttID_path.split(" ")
        uttID = uttID_path_vect[0]
        imageID = uttID
        path = uttID_path_vect[1]
        uttID_path_dict[imageID] = path
        # print "%s: %s" % (imageID, path)


image_file = os.path.join(args.dir + '/', 'images.scp')
image_fh = open(image_file, 'w')

text_file = os.path.join(args.dir + '/', 'text')
text_fh = open(text_file, 'w')

utt2spk_file = os.path.join(args.dir + '/', 'utt2spk')
utt2spk_fh = open(utt2spk_file, 'w')

print('generate new files')
feats_scp_file = os.path.join(args.dir, 'feats.scp')
with open(feats_scp_file) as feats_scp_fh:
    for uttID_image_path in feats_scp_fh:
        uttID_image_path = uttID_image_path.strip()
        uttID_path_vect = uttID_image_path.split(" ")
        uttID = uttID_path_vect[0]
        imageID = "_".join(uttID.split("_")[:-1])
        text_fh.write(uttID + ' ' + text_dict[imageID] + '\n')
        utt2spk_fh.write(uttID + ' ' + uttID_spk_dict[imageID] + '\n')
        image_fh.write(uttID + ' ' + uttID_path_dict[imageID] + '\n')

