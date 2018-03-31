#!/usr/bin/env python3

import argparse
import os
import xml.dom.minidom as minidom
from PIL import Image
from scipy.misc import toimage
import numpy as np

parser = argparse.ArgumentParser(description="""Creates line images from page image.""")
parser.add_argument('database_path1', type=str,
                    help='Path to the downloaded (and extracted) mdacat data file 1')
parser.add_argument('database_path2', type=str,
                    help='Path to the downloaded (and extracted) mdacat data file 2')
parser.add_argument('database_path3', type=str,
                    help='Path to the downloaded (and extracted) mdacat data file 3')
parser.add_argument('data_splits', type=str,
                    help='Path to file that contains the train/test/dev split information')
parser.add_argument('--width_buffer', type=int, default=10,
                    help='width buffer across annotate character')
parser.add_argument('--height_buffer', type=int, default=10,
                    help='height buffer across annotate character')
parser.add_argument('--char_width_buffer', type=int, default=50,
                    help='white space between two characters')
parser.add_argument('--char_height_buffer', type=int, default=20,
                    help='starting location from the top of the line')
args = parser.parse_args()


def set_line_image_data(image, line_id, image_file_name):
    base_name = os.path.splitext(os.path.basename(image_file_name))[0]
    image_file_name_wo_tif, b = image_file_name.split('.tif')
    line_id = '_' + line_id.zfill(4)
    line_image_file_name = base_name + line_id + '.tif'
    imgray = image.convert('L')
    imgray_rev_arr = np.fliplr(imgray)
    imgray_rev = toimage(imgray_rev_arr)    
    image_path=os.path.join(line_images_path, 'lines', line_image_file_name)
    imgray_rev.save(image_path)


def merge_characters_into_line_image(region_list):
    # get image width and height
    blank_space = 50
    image_width = blank_space
    image_height = -1
    for x in region_list:
        (width, height) = x.size
        image_width += width + char_width_buffer
        image_height = max(image_height, height)
    image_width += blank_space
    image_height += char_height_buffer * 2

    stitched_image = Image.new('RGB', (image_width, image_height), "white")
    width_offset = blank_space + width_buffer
    for x in reversed(region_list):
        height_offset = int(char_height_buffer)
        stitched_image.paste(im=x, box=(width_offset, height_offset))
        (width, height) = x.size
        width_offset = width_offset + width + char_width_buffer

    return stitched_image


def get_line_images_from_page_image(image_file_name, madcat_file_path):
    im = Image.open(image_file_name)
    doc = minidom.parse(madcat_file_path)
    zone = doc.getElementsByTagName('zone')
    for node in zone:
        id = node.getAttribute('id')
        region_list = list()
        timage = node.getElementsByTagName('token-image')
        for tnode in timage:
            point = tnode.getElementsByTagName('point')
            col, row = [], []
            max_col, max_row, min_col, min_row = '', '', '', ''
            for pnode in point:
                col.append(int(pnode.getAttribute('x')))
                row.append(int(pnode.getAttribute('y')))
                max_col, max_row = max(col) + height_buffer, max(row) + width_buffer
                min_col, min_row = min(col), min(row)
            box = (min_col, min_row, max_col, max_row)
            region = im.crop(box)
            region_list.append(region)

        image = merge_characters_into_line_image(region_list)
        set_line_image_data(image, id, image_file_name)


def check_file_location():

    madcat_file_path1 = os.path.join(args.database_path1, 'madcat', base_name + '.madcat.xml')
    madcat_file_path2 = os.path.join(args.database_path2, 'madcat', base_name + '.madcat.xml')
    madcat_file_path3 = os.path.join(args.database_path3, 'madcat', base_name + '.madcat.xml')

    image_file_path1 = os.path.join(args.database_path1, 'images', base_name + '.tif')
    image_file_path2 = os.path.join(args.database_path2, 'images', base_name + '.tif')
    image_file_path3 = os.path.join(args.database_path3, 'images', base_name + '.tif')

    if os.path.exists(madcat_file_path1):
        return madcat_file_path1, image_file_path1, wc_dict1

    if os.path.exists(madcat_file_path2):
        return madcat_file_path2, image_file_path2, wc_dict2

    if os.path.exists(madcat_file_path3):
        return madcat_file_path3, image_file_path3, wc_dict3

    print("ERROR: path does not exist")
    return None, None, None


def parse_writing_conditions(writing_conditions):
    with open(writing_conditions) as f:
        file_writing_cond = dict()
        for line in f:
            line_list = line.strip().split("\t")
            file_writing_cond[line_list[0]]=line_list[3]
    return file_writing_cond


def check_writing_condition(wc_dict):
    return True
    writing_condition = wc_dict[base_name].strip()
    if writing_condition != 'IUC':
        return False
    
    return True


### main ###
data_path1 = args.database_path1
data_path2 = args.database_path2
data_path3 = args.database_path3
height_buffer = int(args.height_buffer)
width_buffer = int(args.width_buffer)
char_width_buffer = int(args.char_width_buffer)
char_height_buffer = int(args.char_height_buffer)
line_images_path_list = args.database_path1.split('/')
line_images_path = ('/').join(line_images_path_list[:3])

writing_condiiton_folder_list = args.database_path1.split('/')
writing_condiiton_folder1 = ('/').join(writing_condiiton_folder_list[:4])

writing_condiiton_folder_list = args.database_path2.split('/')
writing_condiiton_folder2 = ('/').join(writing_condiiton_folder_list[:4])

writing_condiiton_folder_list = args.database_path3.split('/')
writing_condiiton_folder3 = ('/').join(writing_condiiton_folder_list[:4])


writing_conditions1 = os.path.join(writing_condiiton_folder1, 'docs', 'writing_conditions.tab')
writing_conditions2 = os.path.join(writing_condiiton_folder2, 'docs', 'writing_conditions.tab')
writing_conditions3 = os.path.join(writing_condiiton_folder3, 'docs', 'writing_conditions.tab')

wc_dict1 = parse_writing_conditions(writing_conditions1)
wc_dict2 = parse_writing_conditions(writing_conditions2)
wc_dict3 = parse_writing_conditions(writing_conditions3)

with open(args.data_splits) as f:
    prev_base_name = ''
    for line in f:
        base_name = os.path.splitext(os.path.splitext(line.split(' ')[0])[0])[0]
        if prev_base_name != base_name:
            prev_base_name = base_name
            madcat_file_path, image_file_path, wc_dict = check_file_location()
            if wc_dict == None or not check_writing_condition(wc_dict):
               continue
            if madcat_file_path != None:
                get_line_images_from_page_image(image_file_path, madcat_file_path)
