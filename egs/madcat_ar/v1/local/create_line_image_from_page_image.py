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

import sys
import argparse
import os
import xml.dom.minidom as minidom
import numpy as np
from math import atan2, cos, sin, pi, degrees, sqrt
from collections import namedtuple

from scipy.spatial import ConvexHull
from skimage.io import imread, imsave
from skimage.transform import rotate
from skimage import img_as_uint

parser = argparse.ArgumentParser(description="Creates line images from page image",
                                 epilog="E.g.  " + sys.argv[0] + "  data/LDC2012T15" 
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

"""
bounding_box is a named tuple which contains:

             area (float): area of the rectangle
             length_parallel (float): length of the side that is parallel to unit_vector
             length_orthogonal (float): length of the side that is orthogonal to unit_vector
             rectangle_center(int, int): coordinates of the rectangle center
             (use rectangle_corners to get the corner points of the rectangle)
             unit_vector (float, float): direction of the length_parallel side.
             (it's orthogonal vector can be found with the orthogonal_vector function
             unit_vector_angle (float): angle of the unit vector to be in radians.
             corner_points [(float, float)]: set that contains the corners of the rectangle
"""

bounding_box_tuple = namedtuple('bounding_box_tuple', 'area '
                                        'length_parallel '
                                        'length_orthogonal '
                                        'rectangle_center '
                                        'unit_vector '
                                        'unit_vector_angle '
                                        'corner_points'
                         )

def unit_vector(pt0, pt1):
    """ Returns an unit vector that points in the direction of pt0 to pt1.
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
    """ From vector returns a orthogonal/perpendicular vector of equal length.
    Args:
        vector (float, float): A vector. Eg. (0.31622776601683794, 0.9486832980505138).

    Returns:
        (float, float): A vector that points in the direction orthogonal to vector.
        Eg. - 0.9486832980505138,0.31622776601683794
    """
    return -1 * vector[1], vector[0]


def bounding_area(index, hull):
    """ Returns a named tuple that mainly contains area of the box that bounds
        the hull. This bounding box orintation is same as the orientation of the 
        lines formed by the point hull[index] and hull[index+1].
    Args:
        index (int): Eg. 1.
        hull [(float, float)]: list or tuple of point cloud
        Eg. ((1.0, -1.0), (2.0, -3.0), (3.0, 4.0), (5.0, 6.0)).

    Returns: a named tuple that contains:
             area: area of the rectangle
             length_parallel: length of the side that is parallel to unit_vector
             length_orthogonal: length of the side that is orthogonal to unit_vector
             rectangle_center: coordinates of the rectangle center
             (use rectangle_corners to get the corner points of the rectangle)
             unit_vector: direction of the length_parallel side.
             (it's orthogonal vector can be found with the orthogonal_vector function
    """
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

    return bounding_box_tuple(
        area=min_rectangle['area'],
        length_parallel=min_rectangle['length_parallel'],
        length_orthogonal=min_rectangle['length_orthogonal'],
        rectangle_center=min_rectangle['rectangle_center'],
        unit_vector=min_rectangle['unit_vector'],
        unit_vector_angle=min_rectangle['unit_vector_angle'],
        corner_points=set(rectangle_corners(min_rectangle))
    )


def get_center(im):
    """ Returns the center pixel location of an image
    Args:
        im: image 

    Returns:
        (int, int): center of the image
        Eg.  2550, 3300
    """
    center = im.shape
    center_x = center[1] / 2
    center_y = center[0] / 2
    return int(center_x), int(center_y)


def get_horizontal_angle(unit_vector_angle):
    """ Returns angle of the unit vector in first or fourth quadrant.
    Args:
        angle (float): angle of the unit vector to be in radians. Eg. 0.01543.

    Returns:
        (float): updated angle of the unit vector to be in radians.
                 It is only in first or fourth quadrant.
        Eg. 0.01543.
    """

    if unit_vector_angle > pi / 2 and unit_vector_angle <= pi:
        unit_vector_angle = unit_vector_angle - pi
    elif unit_vector_angle > -pi and unit_vector_angle < -pi / 2:
        unit_vector_angle = unit_vector_angle + pi

    return unit_vector_angle


def get_smaller_angle(bounding_box):
    """ Returns smallest absolute angle of a rectangle.
    Args:
        rectangle (bounding_box): bounding box rectangle

    Returns:
        (float): smallest angle of the rectangle to be in radians.
        Eg. 0.01543.
    """

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
    """ Rotates the corners of a  bounding box rectangle around the center by smallest angle 
        of the rectangle. It first finds the smallest angle of the rectangle
        then rotates it around the given center point.
    Args:
        rectangle (bounding_box): bounding box rectangle
        center (int, int): center point around which the corners of rectangle are rotated.
        Eg. (2550, 3300).

    Returns: 4 corner points of rectangle.
        Eg. ((1.0, -1.0), (2.0, -3.0), (3.0, 4.0), (5.0, 6.0))
    """

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
    """ Flips a given line image and saves it. Line image file name 
        is formed by appending the line id at the end page image name.
    Args:
        image: line image, non flipped
        line_id (string): id of the line image.
        image_file_name(string): name of the page image.

    Returns:
    """

    base_name = os.path.splitext(os.path.basename(image_file_name))[0]
    image_file_name_wo_tif, b = image_file_name.split('.tif')
    line_id = '_' + line_id.zfill(4)
    line_image_file_name = base_name + line_id + '.tif'
    imgray_rev_arr = np.fliplr(image)
    im = img_as_uint(imgray_rev_arr)
    image_path = os.path.join(line_images_path, 'lines', line_image_file_name)
    imsave(image_path, im)

def get_line_images_from_page_image(image_file_name, madcat_file_path):
    """ Extracts the line image from page image.
    Args:
        image_file_name (string): complete path and name of the page image.
        madcat_file_path (string): complete path and name of the madcat xml file
                                  corresponding to the page image.

    Returns:
    """

    im = imread(image_file_name)
    doc = minidom.parse(madcat_file_path)
    zone = doc.getElementsByTagName('zone')
    for node in zone:
        id = node.getAttribute('id')
        token_image = node.getElementsByTagName('token-image')
        minimum_bounding_box_input = []
        for token_node in token_image:
            word_point = token_node.getElementsByTagName('point')
            for word_node in word_point:
                word_coordinate = (int(word_node.getAttribute('x')), int(word_node.getAttribute('y')))
                minimum_bounding_box_input.append(word_coordinate)
        bounding_box = minimum_bounding_box(minimum_bounding_box_input)
        p1, p2, p3, p4 = bounding_box.corner_points
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        min_x = int(min(x1, x2, x3, x4))
        min_y = int(min(y1, y2, y3, y4))
        max_x = int(max(x1, x2, x3, x4))
        max_y = int(max(y1, y2, y3, y4))
        region_initial = im[min_y:max_y, min_x:max_x]

        if min_y <=0:
            continue

        # #calculate new crop points
        rot_points = []
        p1_new = (x1 - min_x, y1 - min_y)
        p2_new = (x2 - min_x, y2 - min_y)
        p3_new = (x3 - min_x, y3 - min_y)
        p4_new = (x4 - min_x, y4 - min_y)
        rot_points.append(p1_new)
        rot_points.append(p2_new)
        rot_points.append(p3_new)
        rot_points.append(p4_new)

        cropped_bounding_box = bounding_box_tuple(bounding_box.area,
                bounding_box.length_parallel,
                bounding_box.length_orthogonal,
                bounding_box.length_orthogonal,
                bounding_box.unit_vector,
                bounding_box.unit_vector_angle,
                set(rot_points)
            )

        rotation_angle_in_rad = get_smaller_angle(cropped_bounding_box)
        img2 = rotate(region_initial, degrees(rotation_angle_in_rad), order=3)

        x_dash_1, y_dash_1, x_dash_2, y_dash_2, x_dash_3, y_dash_3, x_dash_4, y_dash_4 = rotated_points(
                cropped_bounding_box, get_center(region_initial))


        min_x = int(min(x_dash_1, x_dash_2, x_dash_3, x_dash_4))
        min_y = int(min(y_dash_1, y_dash_2, y_dash_3, y_dash_4))
        max_x = int(max(x_dash_1, x_dash_2, x_dash_3, x_dash_4))
        max_y = int(max(y_dash_1, y_dash_2, y_dash_3, y_dash_4))

        if min_y <= 0:
            continue

        region_final = img2[min_y:max_y, min_x:max_x]
        set_line_image_data(region_final, id, image_file_name)


def check_file_location():
    """ Returns the complete path of the page image and corresponding
        xml file.
    Args:

    Returns:
        image_file_name (string): complete path and name of the page image.
        madcat_file_path (string): complete path and name of the madcat xml file
                                  corresponding to the page image.
    """

    madcat_file_path1 = os.path.join(data_path1, 'madcat', base_name + '.madcat.xml')
    madcat_file_path2 = os.path.join(data_path2, 'madcat', base_name + '.madcat.xml')
    madcat_file_path3 = os.path.join(data_path3, 'madcat', base_name + '.madcat.xml')

    image_file_path1 = os.path.join(data_path1, 'images', base_name + '.tif')
    image_file_path2 = os.path.join(data_path2, 'images', base_name + '.tif')
    image_file_path3 = os.path.join(data_path3, 'images', base_name + '.tif')

    if os.path.exists(madcat_file_path1):
        return madcat_file_path1, image_file_path1, wc_dict1

    if os.path.exists(madcat_file_path2):
        return madcat_file_path2, image_file_path2, wc_dict2

    if os.path.exists(madcat_file_path3):
        return madcat_file_path3, image_file_path3, wc_dict3

    print("ERROR: path does not exist")
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


def allowed_word_segmentation(madcat_file_path):
    doc = minidom.parse(madcat_file_path)
    zone = doc.getElementsByTagName('zone')
    points = []
    for node in zone:
        id = node.getAttribute('id')
        if id != 'z0' and id != 'z1' and id != 'z2' and id != 'z3' :
            continue
        token_image = node.getElementsByTagName('token-image')
        for token_node in token_image:
            word_point = token_node.getElementsByTagName('point')
            for word_node in word_point:
                word_coordinate = (int(word_node.getAttribute('x')), int(word_node.getAttribute('y')))
                points.append(word_coordinate)

    y_val = []
    for point in points:
        y_val.append(point[1])

    min_y = min(y_val)
    if min_y < 100:
        return False
    else:
        return True


### main ###

data_path1 = args.database_path1
data_path2 = args.database_path2
data_path3 = args.database_path3
line_images_path_list = args.database_path1.split('/')
line_images_path = ('/').join(line_images_path_list[:3])

writing_condition_folder_list = args.database_path1.split('/')
writing_condition_folder1 = ('/').join(writing_condition_folder_list[:4])

writing_condition_folder_list = args.database_path2.split('/')
writing_condition_folder2 = ('/').join(writing_condition_folder_list[:4])

writing_condition_folder_list = args.database_path3.split('/')
writing_condition_folder3 = ('/').join(writing_condition_folder_list[:4])


writing_conditions1 = os.path.join(writing_condition_folder1, 'docs', 'writing_conditions.tab')
writing_conditions2 = os.path.join(writing_condition_folder2, 'docs', 'writing_conditions.tab')
writing_conditions3 = os.path.join(writing_condition_folder3, 'docs', 'writing_conditions.tab')

wc_dict1 = parse_writing_conditions(writing_conditions1)
wc_dict2 = parse_writing_conditions(writing_conditions2)
wc_dict3 = parse_writing_conditions(writing_conditions3)

splits_handle = open(args.data_splits, 'r')
splits_data = splits_handle.read().strip().split('\n')
splits_data_new = splits_data[16470:]

prev_base_name = ''
for line in splits_data_new:
    base_name = os.path.splitext(os.path.splitext(line.split(' ')[0])[0])[0]
    if prev_base_name != base_name:
        prev_base_name = base_name
        madcat_file_path, image_file_path, wc_dict = check_file_location()
        if wc_dict == None or not check_writing_condition(wc_dict):
            continue
        if madcat_file_path != None and allowed_word_segmentation(madcat_file_path):
            get_line_images_from_page_image(image_file_path, madcat_file_path)
