#!/usr/bin/env python3

""" This script reads xml file and creates the following files :text, utt2spk, images.scp.
    It also creates line images from page image and stores it into
    data/local/rimes_data/train/lines.
  Eg. local/process_data.py data/local/rimes_data/train train
  Eg. text file: writer000000_train2011-0_000001 Je vous adresse ce courrier afin
      utt2spk file: writer000000_train2011-0_000001 writer000000
      images.scp file: writer000000_train2011-0_000001 \
      data/local/rimes_data/train/lines/train2011-0_000001.png
"""

import argparse
import xml.dom.minidom as minidom
from PIL import Image
import os
import random
parser = argparse.ArgumentParser(description="""Creates line images from page image.""")
parser.add_argument('database_path', type=str,
                    help='Path to the downloaded (and extracted) mdacat data')
parser.add_argument('dataset', type=str,
                    help='Subset of data to process.')
parser.add_argument("--augment", type=lambda x: (str(x).lower()=='true'), default=False,
                   help="performs image augmentation")
parser.add_argument('--pixel-scaling', type=int, default=20,
                    help='padding across horizontal/verticle direction')
args = parser.parse_args()

def expand_aabb(left, right, top, bottom, delta_pixel):
    """ Increases size of axis aligned bounding box (aabb).
    """
    left = left - delta_pixel
    right = right + delta_pixel
    top = top - delta_pixel
    bottom = bottom + delta_pixel
    return left, right, top, bottom

def get_line_images_from_page_image(file_name, left, right, top, bottom, line_id):
    """ Given a page image, extracts the line images from it.
    Input
    -----
    file_name (string): name of the page image.
    left, right, top, bottom (int): coordinates corresponding to the line image.
    line_id (int): line number on the page image.
    """
    page_image_path = os.path.join(page_image_folder, file_name)
    im = Image.open(page_image_path)
    box = (left, top, right, bottom)
    region = im.crop(box)
    base_name = os.path.splitext(os.path.basename(file_name))[0]
    line_image_file_name = base_name + '_' +  str(line_id).zfill(6) + '.png'
    imgray = region.convert('L')
    line_image_path = os.path.join(args.database_path, 'line_image', args.dataset, line_image_file_name)
    imgray.save(line_image_path)
    return base_name, line_image_path

def write_kaldi_process_data_files(base_name, line_id, text):
    """creates files requires for dictionary and feats.scp.
    Input
    -----
    image_path (string): name of the page image.
    line_id (str): line number on the page image.
    text: transcription of the line image.
    base_name (string): 
    """
    writer_id = str(base_name.split('-')[1])
    writer_id = str(writer_id).zfill(6)
    writer_id = 'writer' + writer_id
    utt_id = writer_id + '_' + base_name + '_' +  str(line_id).zfill(6)
    line_image_file_name = base_name + '_' +  str(line_id).zfill(6) + '.png'
    image_path = os.path.join(args.database_path, 'line_image', args.dataset, line_image_file_name)
    text_fh.write(utt_id + ' ' + text + '\n')
    utt2spk_fh.write(utt_id + ' ' + writer_id + '\n')
    image_fh.write(utt_id + ' ' + image_path + '\n')

### main ###
text_file = os.path.join('data', args.dataset, 'text')
text_fh = open(text_file, 'w', encoding='utf-8')
utt2spk_file = os.path.join('data', args.dataset, 'utt2spk')
utt2spk_fh = open(utt2spk_file, 'w', encoding='utf-8')
image_file = os.path.join('data', args.dataset, 'images.scp')
image_fh = open(image_file, 'w', encoding='utf-8')

xml_path = os.path.join(args.database_path, 'xml', args.dataset) + '/rimes_2011.xml'
page_image_folder = os.path.join(args.database_path, 'page_image', args.dataset)
doc = minidom.parse(xml_path)
single_page = doc.getElementsByTagName('SinglePage')
for page in single_page:
    file_name = page.getAttribute('FileName')
    line = page.getElementsByTagName('Line')
    id = 0
    for node in line:
        id += 1
        bottom = int(node.getAttribute('Bottom'))
        left = int(node.getAttribute('Left'))
        right = int(node.getAttribute('Right'))
        top = int(node.getAttribute('Top'))
        text = node.getAttribute('Value')
        text_vect = text.split() # this is to avoid non-utf-8 spaces
        text = " ".join(text_vect)
        if args.augment:
            base_name, image_path = get_line_images_from_page_image(file_name, left, right, top, bottom, str(id))
            write_kaldi_process_data_files(base_name, str(id), text)
            additional_pixel = random.randint(1, args.pixel_scaling)
            left, right, top, bottom = expand_aabb(left, right, top, bottom, args.pixel_scaling + additional_pixel + 1)
            line_id = str(id) + '_scale' + str(2)
            base_name, image_path = get_line_images_from_page_image(file_name, left, right, top, bottom, line_id)
            write_kaldi_process_data_files(base_name, line_id, text)
        else:
            base_name, image_path = get_line_images_from_page_image(file_name, left, right, top, bottom, str(id))
            write_kaldi_process_data_files(base_name, str(id), text)
