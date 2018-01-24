#!/usr/bin/env python3

# Copyright      2017  Chun Chieh Chang
#                2017  Ashish Arora

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

parser = argparse.ArgumentParser(description="""Creates text, utt2spk and images.scp files.""")
parser.add_argument('database_path', type=str,
                    help='Path to the downloaded (and extracted) IAM data')
parser.add_argument('data_splits', type=str,
                    help='Path to file that contains the train/test/dev split information')
parser.add_argument('out_dir', type=str,
                    help='Where to write output files.')
parser.add_argument('writing_conditions', type=str,
                    help='if page is written fast/normal/carefully, lined/unlined')
args = parser.parse_args()

def remove_corrupt_xml_files(gedi_file_path):
    doc = minidom.parse(gedi_file_path)
    line_id_list = list()
    unique_lineid = list()
    DL_ZONE = doc.getElementsByTagName('DL_ZONE')
    for node in DL_ZONE:
        line_id = node.getAttribute('lineID')
        if line_id == "":
            continue
        line_id_list.append(int(line_id))
        unique_lineid = list(set(line_id_list))
        unique_lineid.sort()

    #check if the lineID is empty
    if len(line_id_list) == 0:
        return False

    # check if list contain consequtive entries
    if len(sorted(unique_lineid)) != len(range(min(unique_lineid), max(unique_lineid)+1)):
        return False

    # process the file
    return True

def parse_writing_conditions(writing_conditions):
    with open(writing_conditions) as f:
        file_writing_cond = dict()
        for line in f:
            line_list = line.strip().split("\t")
            file_writing_cond[line_list[0]]=line_list[3]
    return file_writing_cond

def check_writing_condition(wc_dict):
    writing_condition = wc_dict[base_name].strip()
    if writing_condition != 'IUC':
        return False

    return True

text_file = os.path.join(args.out_dir, 'text')
text_fh = open(text_file, 'w', encoding='utf-8')
utt2spk_file = os.path.join(args.out_dir, 'utt2spk')
utt2spk_fh = open(utt2spk_file, 'w', encoding='utf-8')
image_file = os.path.join(args.out_dir, 'images.scp')
image_fh = open(image_file, 'w', encoding='utf-8')
writing_conditions = os.path.join(args.writing_conditions, 'writing_conditions.tab')
wc_dict = parse_writing_conditions(writing_conditions)

image_num = 0
with open(args.data_splits) as f:
    prev_base_name = ''
    for line in f:
        base_name = os.path.splitext(os.path.splitext(line.split(' ')[0])[0])[0]
        if prev_base_name != base_name:
            prev_base_name = base_name
            gedi_xml_path = os.path.join(args.database_path, 'gedi', base_name + '.gedi.xml')
            if not check_writing_condition(wc_dict):
               continue
            if remove_corrupt_xml_files(gedi_xml_path):
                madcat_xml_path = os.path.join(args.database_path, 'madcat', line.split(' ')[0])
                madcat_doc = minidom.parse(madcat_xml_path)
                gedi_doc = minidom.parse(gedi_xml_path)

                writer = madcat_doc.getElementsByTagName('writer')
                writer_id = writer[0].getAttribute('id')
                dl_page = gedi_doc.getElementsByTagName('DL_PAGE')
                for page in dl_page:
                    dl_zone = page.getElementsByTagName('DL_ZONE')
                    lines = []
                    for zone in dl_zone:
                        contents = zone.getAttribute('contents')
                        contents = unicodedata.normalize('NFKC',contents)
                        lineID = zone.getAttribute('lineID')
                        if lineID != '':
                            lineID = int(lineID)
                            while len(lines) < lineID:
                                lines.append([])
                            lines[lineID - 1].append(contents)
                    for lineID, line in enumerate(lines, start=1):
                        if line:
                            image_file_name = base_name + '_' + str(lineID).zfill(3) +'.tif'
                            image_file_path = os.path.join(args.database_path, 'lines', image_file_name)
                            text = ''.join(line)
                            utt_id = writer_id + '_' + str(image_num).zfill(6) + '_' + base_name + '_' + str(lineID).zfill(3)
                            text_fh.write(utt_id + ' ' + text + '\n')
                            utt2spk_fh.write(utt_id + ' ' + writer_id + '\n')
                            image_fh.write(utt_id + ' ' + image_file_path + '\n')
                            image_num = image_num + 1
