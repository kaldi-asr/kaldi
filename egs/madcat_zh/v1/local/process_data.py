#!/usr/bin/env python3

# Copyright  2018  Ashish Arora

""" This script reads the extracted IAM database files and creates
    the following files (for the data subset selected via --dataset):
    text, utt2spk, images.scp.

  Eg. local/process_data.py data/local data/train data --dataset train
  Eg. text file: 000_a01-000u-00 A MOVE to stop Mr. Gaitskell from
      utt2spk file: 000_a01-000u-00 000
      images.scp file: 000_a01-000u-00 data/local/lines/a01/a01-000u/a01-000u-00.png
"""

import argparse
import os
import sys
import xml.dom.minidom as minidom
import unicodedata

parser = argparse.ArgumentParser(description="Creates text, utt2spk and images.scp files",
                                 epilog="E.g.  " + sys.argv[0] + "  data/LDC2012T15"
                                 " data/LDC2013T09 data/LDC2013T15 data/madcat.train.raw.lineid "
                                 " data/train data/local/lines ",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('database_path1',
                    help='Path to the downloaded (and extracted) madcat data')
parser.add_argument('data_splits',
                    help='Path to file that contains the train/test/dev split information')
parser.add_argument('out_dir',
                    help='directory location to write output files.')
args = parser.parse_args()


def check_file_location():
    """ Returns the complete path of the page image and corresponding
        xml file.
    Args:

    Returns:
        image_file_name (string): complete path and name of the page image.
        madcat_file_path (string): complete path and name of the madcat xml file
                                  corresponding to the page image.
    """

    madcat_file_path1 = os.path.join(args.database_path1, 'data', 'madcat', base_name + '.madcat.xml')

    image_file_path1 = os.path.join(args.database_path1, 'data', 'images', base_name + '.tif')

    if os.path.exists(madcat_file_path1):
        return madcat_file_path1, image_file_path1, wc_dict1

    return None, None, None


def parse_writing_conditions(writing_conditions):
    """ Returns a dictionary which have writing condition of each page image.
    Args:
         writing_conditions(string): complete path of writing condition file.

    Returns:
        (dict): dictionary with key as page image name and value as writing condition.
    """

    with open(writing_conditions) as f:
        file_writing_cond = dict()
        for line in f:
            line_list = line.strip().split("\t")
            file_writing_cond[line_list[0]] = line_list[3]
    return file_writing_cond


def check_writing_condition(wc_dict):
    """ Checks if a given page image is writing in a given writing condition.
        It is used to create subset of dataset based on writing condition.
    Args:
         wc_dict (dict): dictionary with key as page image name and value as writing condition.

    Returns:
        (bool): True if writing condition matches.
    """

    return True
    writing_condition = wc_dict[base_name].strip()
    if writing_condition != 'IUC':
        return False

    return True


def get_word_line_mapping(madcat_file_path):
    """ Maps every word in the page image to a  corresponding line.
    Args:
         madcat_file_path (string): complete path and name of the madcat xml file
                                  corresponding to the page image.

    Returns:
    """

    doc = minidom.parse(madcat_file_path)
    zone = doc.getElementsByTagName('zone')
    for node in zone:
        line_id = node.getAttribute('id')
        line_word_dict[line_id] = list()
        word_image = node.getElementsByTagName('token-image')
        for tnode in word_image:
            word_id = tnode.getAttribute('id')
            line_word_dict[line_id].append(word_id)
            word_line_dict[word_id] = line_id


def read_text(madcat_file_path):
    """ Maps every word in the page image to a  corresponding line.
    Args:
        madcat_file_path (string): complete path and name of the madcat xml file
                                  corresponding to the page image.

    Returns:
        dict: Mapping every word in the page image to a  corresponding line.
    """

    text_line_word_dict = dict()
    doc = minidom.parse(madcat_file_path)
    segment = doc.getElementsByTagName('segment')
    for node in segment:
        token = node.getElementsByTagName('token')
        for tnode in token:
            segment_id = tnode.getAttribute('id')
            ref_word_id = tnode.getAttribute('ref_id')
            word = tnode.getElementsByTagName('source')[0].firstChild.nodeValue
            word = unicodedata.normalize('NFKC',word)
            ref_line_id = word_line_dict[ref_word_id]
            if ref_line_id not in text_line_word_dict:
                text_line_word_dict[ref_line_id] = list()
            text_line_word_dict[ref_line_id].append(word)
    return text_line_word_dict


def get_line_image_location():
    image_loc_dict = dict()  # Stores image base name and location
    image_loc_vect = input_image_fh.read().strip().split("\n")
    for line in image_loc_vect:
        base_name = os.path.basename(line)
        location_vect = line.split('/')
        location = "/".join(location_vect[:-1])
        image_loc_dict[base_name]=location
    return image_loc_dict


text_file = os.path.join(args.out_dir, 'text')
text_fh = open(text_file, 'w', encoding='utf-8')
utt2spk_file = os.path.join(args.out_dir, 'utt2spk')
utt2spk_fh = open(utt2spk_file, 'w', encoding='utf-8')
image_file = os.path.join(args.out_dir, 'images.scp')
image_fh = open(image_file, 'w', encoding='utf-8')

data_path1 = os.path.join(args.database_path1, 'data')

input_image_file = os.path.join(args.out_dir, 'lines', 'images.scp')
input_image_fh = open(input_image_file, 'r', encoding='utf-8')

writing_conditions1 = os.path.join(args.database_path1, 'docs', 'writing_conditions.tab')

wc_dict1 = parse_writing_conditions(writing_conditions1)
image_loc_dict = get_line_image_location()

image_num = 0
with open(args.data_splits) as f:
    prev_base_name = ''
    for line in f:
        base_name = os.path.splitext(os.path.splitext(line.split(' ')[0])[0])[0]
        if prev_base_name != base_name:
            prev_base_name = base_name
            madcat_xml_path, image_file_path, wc_dict = check_file_location()
            if wc_dict is None or not check_writing_condition(wc_dict):
                continue
            if madcat_xml_path is not None:
                madcat_doc = minidom.parse(madcat_xml_path)
                writer = madcat_doc.getElementsByTagName('writer')
                writer_id = writer[0].getAttribute('id')
                line_word_dict = dict()
                word_line_dict = dict()
                get_word_line_mapping(madcat_xml_path)
                text_line_word_dict = read_text(madcat_xml_path)
                base_name = os.path.basename(image_file_path)
                base_name, b = base_name.split('.tif')
                for lineID in sorted(text_line_word_dict):
                    updated_base_name = "{}_{}.png".format(base_name, str(lineID).zfill(4))
                    location = image_loc_dict[updated_base_name]
                    image_file_path = os.path.join(location, updated_base_name)
                    line = text_line_word_dict[lineID]
                    text = ' '.join(''.join(line))
                    utt_id = "{}_{}_{}_{}".format(writer_id, str(image_num).zfill(6), base_name, str(lineID).zfill(4))
                    text_fh.write(utt_id + ' ' + text + '\n')
                    utt2spk_fh.write(utt_id + ' ' + writer_id + '\n')
                    image_fh.write(utt_id + ' ' + image_file_path + '\n')
                    image_num = image_num + 1
