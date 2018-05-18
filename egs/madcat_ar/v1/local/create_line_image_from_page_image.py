#!/usr/bin/env python3

# Copyright   2018 Ashish Arora
# Apache 2.0
# minimum bounding box part in this script is originally from
#https://github.com/BebeSparkelSparkel/MinimumBoundingBox
#https://startupnextdoor.com/computing-convex-hull-in-python/
""" This module will be used for extracting line images from page image.
 Given the word segmentation (bounding box around a word) for every word, it will
 extract line segmentation. To extract line segmentation, it will take word bounding
 boxes of a line as input, will create a minimum area bounding box that will contain
 all corner points of word bounding boxes. The obtained bounding box (will not necessarily
 be vertically or horizontally aligned). Hence to extract line image from line bounding box,
 page image is rotated and line image is cropped and saved.
"""

import sys
import argparse
import os
import xml.dom.minidom as minidom
import numpy as np
from math import atan2, cos, sin, pi, degrees, sqrt
from collections import namedtuple

from scipy.spatial import ConvexHull
from PIL import Image
from scipy.misc import toimage
import logging

sys.path.insert(0, 'steps')
logger = logging.getLogger('libs')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(pathname)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

parser = argparse.ArgumentParser(description="Creates line images from page image",
                                 epilog="E.g.  " + sys.argv[0] + "  data/LDC2012T15"
                                             " data/LDC2013T09 data/LDC2013T15 data/madcat.train.raw.lineid "
                                             " data/local/lines ",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('database_path1', type=str,
                    help='Path to the downloaded madcat data directory 1')
parser.add_argument('database_path2', type=str,
                    help='Path to the downloaded madcat data directory 2')
parser.add_argument('database_path3', type=str,
                    help='Path to the downloaded madcat data directory 3')
parser.add_argument('data_splits', type=str,
                    help='Path to file that contains the train/test/dev split information')
parser.add_argument('out_dir', type=str,
                    help='directory location to write output files')
parser.add_argument('--padding', type=int, default=400,
                    help='padding across horizontal/verticle direction')
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
    """ Given two points pt0 and pt1, return a unit vector that
        points in the direction of pt0 to pt1.
    Returns
    -------
    (float, float): unit vector
    """
    dis_0_to_1 = sqrt((pt0[0] - pt1[0])**2 + (pt0[1] - pt1[1])**2)
    return (pt1[0] - pt0[0]) / dis_0_to_1, \
           (pt1[1] - pt0[1]) / dis_0_to_1


def orthogonal_vector(vector):
    """ Given a vector, returns a orthogonal/perpendicular vector of equal length.
    Returns
    ------
    (float, float): A vector that points in the direction orthogonal to vector.
    """
    return -1 * vector[1], vector[0]


def bounding_area(index, hull):
    """ Given index location in an array and convex hull, it gets two points
        hull[index] and hull[index+1]. From these two points, it returns a named
        tuple that mainly contains area of the box that bounds the hull. This
        bounding box orintation is same as the orientation of the lines formed
        by the point hull[index] and hull[index+1].
    Returns
    -------
    a named tuple that contains:
    area: area of the rectangle
    length_parallel: length of the side that is parallel to unit_vector
    length_orthogonal: length of the side that is orthogonal to unit_vector
    rectangle_center: coordinates of the rectangle center
    unit_vector: direction of the length_parallel side.
    (it's orthogonal vector can be found with the orthogonal_vector function)
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
    """ Given angle from horizontal axis and a point from origin,
        returns converted unit vector coordinates in x, y coordinates.
        angle of unit vector should be in radians.
    Returns
    ------
    (float, float): converted x,y coordinate of the unit vector.
    """
    angle_orthogonal = unit_vector_angle + pi / 2
    return point[0] * cos(unit_vector_angle) + point[1] * cos(angle_orthogonal), \
           point[0] * sin(unit_vector_angle) + point[1] * sin(angle_orthogonal)


def rotate_points(center_of_rotation, angle, points):
    """ Rotates a point cloud around the center_of_rotation point by angle
    input
    -----
    center_of_rotation (float, float): angle of unit vector to be in radians.
    angle (float): angle of rotation to be in radians.
    points [(float, float)]: Points to be a list or tuple of points. Points to be rotated.
    Returns
    ------
    [(float, float)]: Rotated points around center of rotation by angle
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
    """ Given rectangle center and its inclination, returns the corner
        locations of the rectangle.
    Returns
    ------
    [(float, float)]: 4 corner points of rectangle.
    """
    corner_points = []
    for i1 in (.5, -.5):
        for i2 in (i1, -1 * i1):
            corner_points.append((rectangle['rectangle_center'][0] + i1 * rectangle['length_parallel'],
                            rectangle['rectangle_center'][1] + i2 * rectangle['length_orthogonal']))

    return rotate_points(rectangle['rectangle_center'], rectangle['unit_vector_angle'], corner_points)


def get_orientation(origin, p1, p2):
    """
    Given origin and two points, return the orientation of the Point p1 with
    regards to Point p2 using origin.
    Returns
    -------
    integer: Negative if p1 is clockwise of p2.
    """
    difference = (
        ((p2[0] - origin[0]) * (p1[1] - origin[1]))
        - ((p1[0] - origin[0]) * (p2[1] - origin[1]))
    )
    return difference


def compute_hull(points):
    """
    Given input list of points, return a list of points that
    made up the convex hull.
    Returns
    -------
    [(float, float)]: convexhull points
    """
    hull_points = []
    start = points[0]
    min_x = start[0]
    for p in points[1:]:
        if p[0] < min_x:
            min_x = p[0]
            start = p

    point = start
    hull_points.append(start)

    far_point = None
    while far_point is not start:
        p1 = None
        for p in points:
            if p is point:
                continue
            else:
                p1 = p
                break

        far_point = p1

        for p2 in points:
            if p2 is point or p2 is p1:
                continue
            else:
                direction = get_orientation(point, far_point, p2)
                if direction > 0:
                    far_point = p2

        hull_points.append(far_point)
        point = far_point
    return hull_points


def minimum_bounding_box(points):
    """ Given a list of 2D points, it returns the minimum area rectangle bounding all
        the points in the point cloud.
    Returns
    ------
    returns a namedtuple that contains:
    area: area of the rectangle
    length_parallel: length of the side that is parallel to unit_vector
    length_orthogonal: length of the side that is orthogonal to unit_vector
    rectangle_center: coordinates of the rectangle center
    unit_vector: direction of the length_parallel side. RADIANS
    unit_vector_angle: angle of the unit vector
    corner_points: set that contains the corners of the rectangle
    """

    if len(points) <= 2: raise ValueError('More than two points required.')

    hull_ordered = [points[index] for index in ConvexHull(points).vertices]
    hull_ordered.append(hull_ordered[0])
    #hull_ordered = compute_hull(points)
    hull_ordered = tuple(hull_ordered)

    min_rectangle = bounding_area(0, hull_ordered)
    for i in range(1, len(hull_ordered)-1):
        rectangle = bounding_area(i, hull_ordered)
        if rectangle['area'] < min_rectangle['area']:
            min_rectangle = rectangle

    min_rectangle['unit_vector_angle'] = atan2(min_rectangle['unit_vector'][1], min_rectangle['unit_vector'][0])
    min_rectangle['rectangle_center'] = to_xy_coordinates(min_rectangle['unit_vector_angle'], min_rectangle['rectangle_center'])

    return bounding_box_tuple(
        area = min_rectangle['area'],
        length_parallel = min_rectangle['length_parallel'],
        length_orthogonal = min_rectangle['length_orthogonal'],
        rectangle_center = min_rectangle['rectangle_center'],
        unit_vector = min_rectangle['unit_vector'],
        unit_vector_angle = min_rectangle['unit_vector_angle'],
        corner_points = set(rectangle_corners(min_rectangle))
    )


def get_center(im):
    """ Given image, returns the location of center pixel
    Returns
    -------
    (int, int): center of the image
    """
    center_x = im.size[0] / 2
    center_y = im.size[1] / 2
    return int(center_x), int(center_y)


def get_horizontal_angle(unit_vector_angle):
    """ Given an angle in radians, returns angle of the unit vector in
        first or fourth quadrant.
    Returns
    ------
    (float): updated angle of the unit vector to be in radians.
             It is only in first or fourth quadrant.
    """
    if unit_vector_angle > pi / 2 and unit_vector_angle <= pi:
        unit_vector_angle = unit_vector_angle - pi
    elif unit_vector_angle > -pi and unit_vector_angle < -pi / 2:
        unit_vector_angle = unit_vector_angle + pi

    return unit_vector_angle


def get_smaller_angle(bounding_box):
    """ Given a rectangle, returns its smallest absolute angle from horizontal axis.
    Returns
    ------
    (float): smallest angle of the rectangle to be in radians.
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
    """ Given the rectangle, returns corner points of rotated rectangle.
        It rotates the rectangle around the center by its smallest angle.
    Returns
    -------
    [(int, int)]: 4 corner points of rectangle.
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


def pad_image(image):
    """ Given an image, returns a padded image around the border.
        This routine save the code from crashing if bounding boxes that are
        slightly outside the page boundary.
    Returns
    -------
    image: page image
    """
    offset = int(args.padding // 2)
    padded_image = Image.new('RGB', (image.size[0] + int(args.padding), image.size[1] + int(args.padding)), "white")
    padded_image.paste(im = image, box = (offset, offset))
    return padded_image


def update_minimum_bounding_box_input(bounding_box_input):
    """ Given list of 2D points, returns list of 2D points shifted by an offset.
    Returns
    ------
    points [(float, float)]: points, a list or tuple of 2D coordinates
    """
    updated_minimum_bounding_box_input = []
    offset = int(args.padding // 2)
    for point in bounding_box_input:
        x, y = point
        new_x = x + offset
        new_y = y + offset
        word_coordinate = (new_x, new_y)
        updated_minimum_bounding_box_input.append(word_coordinate)

    return updated_minimum_bounding_box_input


def set_line_image_data(image, line_id, image_file_name, image_fh):
    """ Given an image, saves a flipped line image. Line image file name
        is formed by appending the line id at the end page image name.
    """

    base_name = os.path.splitext(os.path.basename(image_file_name))[0]
    line_id = '_' + line_id.zfill(4)
    line_image_file_name = base_name + line_id + '.png'
    image_path = os.path.join(args.out_dir, line_image_file_name)
    imgray = image.convert('L')
    imgray_rev_arr = np.fliplr(imgray)
    imgray_rev = toimage(imgray_rev_arr)
    imgray_rev.save(image_path)
    image_fh.write(image_path + '\n')


def get_line_images_from_page_image(image_file_name, madcat_file_path, image_fh):
    """ Given a page image, extracts the line images from it.
    Input
    -----
    image_file_name (string): complete path and name of the page image.
    madcat_file_path (string): complete path and name of the madcat xml file
                                  corresponding to the page image.
    """
    im_wo_pad = Image.open(image_file_name)
    im = pad_image(im_wo_pad)
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
        updated_mbb_input = update_minimum_bounding_box_input(minimum_bounding_box_input)
        bounding_box = minimum_bounding_box(updated_mbb_input)

        p1, p2, p3, p4 = bounding_box.corner_points
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        min_x = int(min(x1, x2, x3, x4))
        min_y = int(min(y1, y2, y3, y4))
        max_x = int(max(x1, x2, x3, x4))
        max_y = int(max(y1, y2, y3, y4))
        box = (min_x, min_y, max_x, max_y)
        region_initial = im.crop(box)
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
        img2 = region_initial.rotate(degrees(rotation_angle_in_rad), resample = Image.BICUBIC)
        x_dash_1, y_dash_1, x_dash_2, y_dash_2, x_dash_3, y_dash_3, x_dash_4, y_dash_4 = rotated_points(
                cropped_bounding_box, get_center(region_initial))

        min_x = int(min(x_dash_1, x_dash_2, x_dash_3, x_dash_4))
        min_y = int(min(y_dash_1, y_dash_2, y_dash_3, y_dash_4))
        max_x = int(max(x_dash_1, x_dash_2, x_dash_3, x_dash_4))
        max_y = int(max(y_dash_1, y_dash_2, y_dash_3, y_dash_4))
        box = (min_x, min_y, max_x, max_y)
        region_final = img2.crop(box)
        set_line_image_data(region_final, id, image_file_name, image_fh)


def check_file_location(base_name, wc_dict1, wc_dict2, wc_dict3):
    """ Returns the complete path of the page image and corresponding
        xml file.
    Returns
    -------
    image_file_name (string): complete path and name of the page image.
    madcat_file_path (string): complete path and name of the madcat xml file
                               corresponding to the page image.
    """
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

    return None, None, None


def parse_writing_conditions(writing_conditions):
    """ Given writing condition file path, returns a dictionary which have writing condition
        of each page image.
    Returns
    ------
    (dict): dictionary with key as page image name and value as writing condition.
    """
    with open(writing_conditions) as f:
        file_writing_cond = dict()
        for line in f:
            line_list = line.strip().split("\t")
            file_writing_cond[line_list[0]] = line_list[3]
    return file_writing_cond


def check_writing_condition(wc_dict, base_name):
    """ Given writing condition dictionary, checks if a page image is writing
        in a specifed writing condition.
        It is used to create subset of dataset based on writing condition.
    Returns
    (bool): True if writing condition matches.
    """
    return True
    writing_condition = wc_dict[base_name].strip()
    if writing_condition != 'IUC':
        return False

    return True


### main ###

def main():

    writing_condition_folder_list = args.database_path1.split('/')
    writing_condition_folder1 = ('/').join(writing_condition_folder_list[:5])

    writing_condition_folder_list = args.database_path2.split('/')
    writing_condition_folder2 = ('/').join(writing_condition_folder_list[:5])

    writing_condition_folder_list = args.database_path3.split('/')
    writing_condition_folder3 = ('/').join(writing_condition_folder_list[:5])

    writing_conditions1 = os.path.join(writing_condition_folder1, 'docs', 'writing_conditions.tab')
    writing_conditions2 = os.path.join(writing_condition_folder2, 'docs', 'writing_conditions.tab')
    writing_conditions3 = os.path.join(writing_condition_folder3, 'docs', 'writing_conditions.tab')

    wc_dict1 = parse_writing_conditions(writing_conditions1)
    wc_dict2 = parse_writing_conditions(writing_conditions2)
    wc_dict3 = parse_writing_conditions(writing_conditions3)

    output_directory = args.out_dir
    image_file = os.path.join(output_directory, 'images.scp')
    image_fh = open(image_file, 'w', encoding='utf-8')

    splits_handle = open(args.data_splits, 'r')
    splits_data = splits_handle.read().strip().split('\n')
    prev_base_name = ''
    for line in splits_data:
        base_name = os.path.splitext(os.path.splitext(line.split(' ')[0])[0])[0]
        if prev_base_name != base_name:
            prev_base_name = base_name
            madcat_file_path, image_file_path, wc_dict = check_file_location(base_name, wc_dict1, wc_dict2, wc_dict3)
            if wc_dict is None or not check_writing_condition(wc_dict, base_name):
                continue
            if madcat_file_path is not None:
                get_line_images_from_page_image(image_file_path, madcat_file_path, image_fh)


if __name__ == '__main__':
      main()

