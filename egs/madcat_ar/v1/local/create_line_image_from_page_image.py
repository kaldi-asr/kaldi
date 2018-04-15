#!/usr/bin/env python3
# Copyright   2018 Ashish Arora
# Apache 2.0
# minimum bounding box part in this script is originally from
#https://github.com/BebeSparkelSparkel/MinimumBoundingBox

""" This module will be used for extracting line images from page image.
 Given the word segmentation (bounding box around a word) for  every word, it will
 extract line segmentation. To extract line segmentation, it will take word bounding
 boxes of a line as input, will create a minimum area bounding box that will contain 
 all corner points of word bounding boxes. The obtained bounding box (will not necessarily 
 be vertically or horizontally aligned). Hence to extract line image from line bounding box,
 page image is rotated and line image is cropped and saved.
 Args:
  database_path1: Path to the downloaded (and extracted) madcat data directory 1
          Eg. /export/corpora/LDC/LDC2012T15
  database_path2: Path to the downloaded (and extracted) madcat data directory 2
          Eg. /export/corpora/LDC/LDC2013T09
  database_path3: Path to the downloaded (and extracted) madcat data directory 3
          Eg. /export/corpora/LDC/LDC2013T15
  data_splits: Path to file that contains the train,test or development split information.
               There are total 3 split files. one of train, test and dev each.
          Eg. /home/kduh/proj/scale2018/data/madcat_datasplit/ar-en/madcat.train.raw.lineid
             groups.google.com_women1000_508c404bd84f8ba3_ARB_20060426_124900_3_LDC0188.madcat.xml s1
             <xml file name> <scribe number>
             <scribe number>: it is the number of time this page has been written
  
  Eg. local/create_line_image_from_page_image.py /export/corpora/LDC/LDC2012T15 /export/corpora/LDC/LDC2013T09 
      /export/corpora/LDC/LDC2013T15 /home/kduh/proj/scale2018/data/madcat_datasplit/ar-en/madcat.train.raw.lineid
"""

import argparse
import os
import xml.dom.minidom as minidom
from PIL import Image
import numpy as np
from scipy.misc import toimage

from scipy.spatial import ConvexHull
from math import atan2, cos, sin, pi, degrees, sqrt
from collections import namedtuple

parser = argparse.ArgumentParser(description="Creates line images from page image",
                                 epilog="E.g. local/create_line_image_from_page_image.py data/LDC2012T15" 
                                             " data/LDC2013T09 data/LDC2013T15 data/madcat.train.raw.lineid ",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('database_path1', type=str,
                    help='Path to the downloaded madcat data directory 1')
parser.add_argument('database_path2', type=str,
                    help='Path to the downloaded madcat data directory 2')
parser.add_argument('database_path3', type=str,
                    help='Path to the downloaded madcat data directory 3')
parser.add_argument('data_splits', type=str,
                    help='Path to file that contains the train/test/dev split information')
args = parser.parse_args()

bounding_box = namedtuple('bounding_box', ('area',
                                         'length_parallel',
                                         'length_orthogonal',
                                         'rectangle_center',
                                         'unit_vector',
                                         'unit_vector_angle',
                                         'corner_points'))


def unit_vector(pt0, pt1):
    """Returns an unit vector that points in the direction of pt0 to pt1.
    Args:
        pt0 (float, float): Point 0. Eg. (1.0, 2.0).
        pt1 (float, float): Point 1. Eg. (3.0, 8.0).

    Returns:
        (float, float): unit vector that points in the direction of pt0 to pt1.
        Eg.  0.31622776601683794, 0.9486832980505138
    """
    dis_0_to_1 = sqrt((pt0[0] - pt1[0])**2 + (pt0[1] - pt1[1])**2)
    return (pt1[0] - pt0[0]) / dis_0_to_1, \
           (pt1[1] - pt0[1]) / dis_0_to_1


def orthogonal_vector(vector):
    """From vector returns a orthogonal/perpendicular vector of equal length.
    Args:
        vector (float, float): A vector. Eg. (0.31622776601683794, 0.9486832980505138).

    Returns:
        (float, float): A vector that points in the direction orthogonal to vector.
        Eg. - 0.9486832980505138,0.31622776601683794
    """
    return -1 * vector[1], vector[0]


def bounding_area(index, hull):
    unit_vector_p = unit_vector(hull[index], hull[index+1])
    unit_vector_o = orthogonal_vector(unit_vector_p)

    dis_p = tuple(np.dot(unit_vector_p, pt) for pt in hull)
    dis_o = tuple(np.dot(unit_vector_o, pt) for pt in hull)

    min_p = min(dis_p)
    min_o = min(dis_o)
    len_p = max(dis_p) - min_p
    len_o = max(dis_o) - min_o

    return {'area': len_p * len_o,
            'length_parallel': len_p,
            'length_orthogonal': len_o,
            'rectangle_center': (min_p + len_p / 2, min_o + len_o / 2),
            'unit_vector': unit_vector_p,
            }


def to_xy_coordinates(unit_vector_angle, point):
    """ Returns converted unit vector coordinates in x, y coordinates.
    Args:
        unit_vector_angle (float): angle of unit vector to be in radians. 
        Eg. 0.1543 .
        point (float, float): Point from origin. Eg. (1.0, 2.0).
    Returns:
        (float, float): converted x,y coordinate of the unit vector.
        Eg. 0.680742447866183, 2.1299271629971663
    """
    angle_orthogonal = unit_vector_angle + pi / 2
    return point[0] * cos(unit_vector_angle) + point[1] * cos(angle_orthogonal), \
           point[0] * sin(unit_vector_angle) + point[1] * sin(angle_orthogonal)


def rotate_points(center_of_rotation, angle, points):
    """ Rotates a point cloud around the center_of_rotation point by angle
    Args:
        center_of_rotation (float, float): angle of unit vector to be in radians.
        Eg. (1.56, -23.4).
        angle (float): angle of rotation to be in radians. Eg. 0.1543 .
        points [(float, float)]: Points to be a list or tuple of points. Points to be rotated. 
        Eg. ((1.56, -23.4), (1.56, -23.4))
    Returns:
        [(float, float)]: Rotated points around center of rotation by angle
        Eg. ((1.16, -12.4), (2.34, -34.4))
    """

    rot_points = []
    ang = []
    for pt in points:
        diff = tuple([pt[d] - center_of_rotation[d] for d in range(2)])
        diff_angle = atan2(diff[1], diff[0]) + angle
        ang.append(diff_angle)
        diff_length = sqrt(sum([d**2 for d in diff]))
        rot_points.append((center_of_rotation[0] + diff_length * cos(diff_angle),
                           center_of_rotation[1] + diff_length * sin(diff_angle)))

    return rot_points


def rectangle_corners(rectangle):
    """ Given rectangle center and its inclination. It returns the corner 
        locations of the rectangle.
    Args:
        rectangle (bounding_box): the output of minimum bounding box rectangle
    Returns:
    [(float, float)]: 4 corner points of rectangle.
        Eg. ((1.0, -1.0), (2.0, -3.0), (3.0, 4.0), (5.0, 6.0))
    """
    corner_points = []
    for i1 in (.5, -.5):
        for i2 in (i1, -1 * i1):
            corner_points.append((rectangle['rectangle_center'][0] + i1 * rectangle['length_parallel'],
                            rectangle['rectangle_center'][1] + i2 * rectangle['length_orthogonal']))

    return rotate_points(rectangle['rectangle_center'], rectangle['unit_vector_angle'], corner_points)


# use this function to find the listed properties of the minimum bounding box of a point cloud
def minimum_bounding_box(points):
    """ Given a point cloud, it returns the minimum area rectangle bounding all 
        the points in the point cloud.
    Args:
        points [(float, float)]: points to be a list or tuple of 2D points
                                 needs to be more than 2 points
    Returns: returns a namedtuple that contains:
             area: area of the rectangle
             length_parallel: length of the side that is parallel to unit_vector
             length_orthogonal: length of the side that is orthogonal to unit_vector
             rectangle_center: coordinates of the rectangle center
             (use rectangle_corners to get the corner points of the rectangle)
             unit_vector: direction of the length_parallel side. RADIANS
             (it's orthogonal vector can be found with the orthogonal_vector function
             unit_vector_angle: angle of the unit vector
             corner_points: set that contains the corners of the rectangle
    """

    if len(points) <= 2: raise ValueError('More than two points required.')

    hull_ordered = [points[index] for index in ConvexHull(points).vertices]
    hull_ordered.append(hull_ordered[0])
    hull_ordered = tuple(hull_ordered)

    min_rectangle = bounding_area(0, hull_ordered)
    for i in range(1, len(hull_ordered)-1):
        rectangle = bounding_area(i, hull_ordered)
        if rectangle['area'] < min_rectangle['area']:
            min_rectangle = rectangle

    min_rectangle['unit_vector_angle'] = atan2(min_rectangle['unit_vector'][1], min_rectangle['unit_vector'][0])
    min_rectangle['rectangle_center'] = to_xy_coordinates(min_rectangle['unit_vector_angle'], min_rectangle['rectangle_center'])

    return bounding_box(
        area=min_rectangle['area'],
        length_parallel=min_rectangle['length_parallel'],
        length_orthogonal=min_rectangle['length_orthogonal'],
        rectangle_center=min_rectangle['rectangle_center'],
        unit_vector=min_rectangle['unit_vector'],
        unit_vector_angle=min_rectangle['unit_vector_angle'],
        corner_points=set(rectangle_corners(min_rectangle))
    )


def get_center(im):
    center_x = im.size[0]/2
    center_y = im.size[1]/2
    return center_x, center_y


def get_horizontal_angle(unit_vector_angle):
    if unit_vector_angle > pi / 2 and unit_vector_angle <= pi:
        unit_vector_angle = unit_vector_angle - pi
    elif unit_vector_angle > -pi and unit_vector_angle < -pi / 2:
        unit_vector_angle = unit_vector_angle + pi

    return unit_vector_angle


def get_smaller_angle(bounding_box):
    unit_vector = bounding_box.unit_vector
    unit_vector_angle = bounding_box.unit_vector_angle
    ortho_vector = orthogonal_vector(unit_vector)
    ortho_vector_angle = atan2(ortho_vector[1], ortho_vector[0])

    unit_vector_angle_updated = get_horizontal_angle(unit_vector_angle)
    ortho_vector_angle_updated = get_horizontal_angle(ortho_vector_angle)

    if abs(unit_vector_angle_updated) < abs(ortho_vector_angle_updated):
        return unit_vector_angle_updated
    else:
        return ortho_vector_angle_updated


def rotated_points(bounding_box, center):
    p1, p2, p3, p4 = bounding_box.corner_points
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    center_x, center_y = center
    rotation_angle_in_rad = -get_smaller_angle(bounding_box)
    x_dash_1 = (x1 - center_x) * cos(rotation_angle_in_rad) - (y1 - center_y) * sin(rotation_angle_in_rad) + center_x
    x_dash_2 = (x2 - center_x) * cos(rotation_angle_in_rad) - (y2 - center_y) * sin(rotation_angle_in_rad) + center_x
    x_dash_3 = (x3 - center_x) * cos(rotation_angle_in_rad) - (y3 - center_y) * sin(rotation_angle_in_rad) + center_x
    x_dash_4 = (x4 - center_x) * cos(rotation_angle_in_rad) - (y4 - center_y) * sin(rotation_angle_in_rad) + center_x

    y_dash_1 = (y1 - center_y) * cos(rotation_angle_in_rad) + (x1 - center_x) * sin(rotation_angle_in_rad) + center_y
    y_dash_2 = (y2 - center_y) * cos(rotation_angle_in_rad) + (x2 - center_x) * sin(rotation_angle_in_rad) + center_y
    y_dash_3 = (y3 - center_y) * cos(rotation_angle_in_rad) + (x3 - center_x) * sin(rotation_angle_in_rad) + center_y
    y_dash_4 = (y4 - center_y) * cos(rotation_angle_in_rad) + (x4 - center_x) * sin(rotation_angle_in_rad) + center_y
    return x_dash_1, y_dash_1, x_dash_2, y_dash_2, x_dash_3, y_dash_3, x_dash_4, y_dash_4


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


def get_line_images_from_page_image(image_file_name, madcat_file_path):
    im = Image.open(image_file_name)
    doc = minidom.parse(madcat_file_path)
    zone = doc.getElementsByTagName('zone')
    for node in zone:
        id = node.getAttribute('id')
        token_image = node.getElementsByTagName('token-image')
        minimum_bounding_box_input = []
        for token_node in token_image:
            word_point = token_node.getElementsByTagName('point')
            col_word, row_word = [], []
            for word_node in word_point:
                col_word.append(int(word_node.getAttribute('x')))
                row_word.append(int(word_node.getAttribute('y')))
                word_coordinate = (int(word_node.getAttribute('x')), int(word_node.getAttribute('y')))
                minimum_bounding_box_input.append(word_coordinate)
        bounding_box = minimum_bounding_box(minimum_bounding_box_input)
        rotation_angle_in_rad = get_smaller_angle(bounding_box)

        img2 = im.rotate(degrees(rotation_angle_in_rad), resample=Image.BICUBIC)
        x_dash_1, y_dash_1, x_dash_2, y_dash_2, x_dash_3, y_dash_3, x_dash_4, y_dash_4 = rotated_points(
            bounding_box, get_center(im))

        min_x = min(x_dash_1, x_dash_2, x_dash_3, x_dash_4)
        min_y = min(y_dash_1, y_dash_2, y_dash_3, y_dash_4)

        max_x = max(x_dash_1, x_dash_2, x_dash_3, x_dash_4)
        max_y = max(y_dash_1, y_dash_2, y_dash_3, y_dash_4)
        box = (min_x, min_y, max_x, max_y)

        region = img2.crop(box)
        set_line_image_data(region, id, image_file_name)


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
            file_writing_cond[line_list[0]] = line_list[3]
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
