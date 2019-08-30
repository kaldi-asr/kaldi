#!/usr/bin/env python3

# Copyright      2018  Ashish Arora
#                2018  Chun Chieh Chang

""" This script reads the extracted Farsi OCR (yomdle and slam) database files 
    and creates the following files (for the data subset selected via --dataset):
    text, utt2spk, images.scp.
  Eg. local/process_data.py data/download/ data/local/splits/train.txt data/train
  Eg. text file: english_phone_books_0001_1 To sum up, then, it would appear that
      utt2spk file: english_phone_books_0001_0 english_phone_books_0001
      images.scp file: english_phone_books_0001_0 \
      data/download/truth_line_image/english_phone_books_0001_0.png
"""

import argparse
import numpy as np
import os
import re
import struct
import sys
import unicodedata
from collections import namedtuple
from math import atan2, cos, sin, pi, degrees, sqrt
from PIL import Image
from scipy import misc
from scipy.spatial import ConvexHull

parser = argparse.ArgumentParser(description="Creates text, utt2spk, and images.scp files")
parser.add_argument('database_path', type=str, help='Path to data')
parser.add_argument('out_dir', type=str, help='directory to output files')
parser.add_argument('--padding', type=int, default=100, help='Padding so BBox does not exceed image area')
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
                                        'corner_points')


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
            'rectangle_center': (min_p + float(len_p) / 2, min_o + float(len_o) / 2),
            'unit_vector': unit_vector_p}

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
        corner_points = set(rectangle_corners(min_rectangle)))

def get_center(im):
    """ Given image, returns the location of center pixel
    Returns
    -------
    (int, int): center of the image
    """
    center_x = float(im.size[0]) / 2
    center_y = float(im.size[1]) / 2
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

### main ###
print("Processing '{}' data...".format(args.out_dir))

text_file = os.path.join(args.out_dir, 'text')
text_fh = open(text_file, 'w', encoding='utf-8')
utt2spk_file = os.path.join(args.out_dir, 'utt2spk')
utt2spk_fh = open(utt2spk_file, 'w', encoding='utf-8')
image_file = os.path.join(args.out_dir, 'images.scp')
image_fh = open(image_file, 'w', encoding='utf-8')

for filename in sorted(os.listdir(args.database_path)):
    if filename.endswith('.dgr'):
        with open(os.path.join(args.database_path, filename), 'rb')  as f:
            iHdSize = struct.unpack('i', f.read(4))[0]
            szFormatCode = struct.unpack(''.join('c' for x in range(0,8)), f.read(8))
            szFormatCode = "".join([x.decode('utf8') for x in szFormatCode])
            szIllustr = f.read(iHdSize - 36)
            szCodeType = struct.unpack(''.join(['c' for x in range(0,20)]), f.read(20))
            szCodeType = "".join([x.decode('utf8') for x in szCodeType])
            sCodeLen = struct.unpack('h', f.read(2))[0]
            sBitApp = struct.unpack('h', f.read(2))[0]
            iImgHei = struct.unpack('i', f.read(4))[0]
            iImgWid = struct.unpack('i', f.read(4))[0]
            pDocImg = Image.new('L', (iImgWid, iImgHei), (255))
            iLineNum = struct.unpack('i', f.read(4))[0]
            text_dict = {}
            image_dict = {}
            for i in range(0, iLineNum):
                iWordNum = struct.unpack('i', f.read(4))[0]
                for j in range(0, iWordNum):
                    pWordLabel = f.read(sCodeLen).decode('gb18030', errors='ignore')
                    sTop = struct.unpack('h', f.read(2))[0]
                    sLeft = struct.unpack('h', f.read(2))[0]
                    sHei = struct.unpack('h', f.read(2))[0]
                    sWid = struct.unpack('h', f.read(2))[0]
                    if i in text_dict:
                        text_dict[i] += [pWordLabel]
                    else:
                        text_dict[i] = [pWordLabel]
                    if i in image_dict:
                        image_dict[i] += [[sTop, sLeft, sHei, sWid]]
                    else:
                        image_dict[i] = [[sTop, sLeft, sHei, sWid]]
                    pTmpData = struct.unpack("{}B".format(sHei * sWid), f.read(sHei * sWid))
                    character = misc.toimage(np.array(pTmpData).reshape(sHei, sWid))
                    pDocImg.paste(character, (sLeft, sTop))
            pDocImg.save(os.path.join(args.out_dir, 'data', 'images', os.path.splitext(filename)[0] + '.png'), 'png')
            
            im_page = pad_image(pDocImg)
            for i in range(0, iLineNum):
                text = ""
                points = []
                for j, char in enumerate(text_dict[i]):
                    text += char
                    points.append([image_dict[i][j][1], image_dict[i][j][0]])
                    points.append([image_dict[i][j][1] + image_dict[i][j][3], image_dict[i][j][0]])
                    points.append([image_dict[i][j][1], image_dict[i][j][0] + image_dict[i][j][2]])
                    points.append([image_dict[i][j][1] + image_dict[i][j][3], image_dict[i][j][0] + image_dict[i][j][2]])
                updated_mbb_input = update_minimum_bounding_box_input(points)
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
                region_initial = im_page.crop(box)
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
                    set(rot_points))

                rotation_angle_in_rad = get_smaller_angle(cropped_bounding_box)
                img2 = region_initial.rotate(degrees(rotation_angle_in_rad), resample=Image.BICUBIC)
                x_dash_1, y_dash_1, x_dash_2, y_dash_2, x_dash_3, y_dash_3, x_dash_4, y_dash_4 = rotated_points(
                    cropped_bounding_box, get_center(region_initial))


                min_x = int(min(x_dash_1, x_dash_2, x_dash_3, x_dash_4))
                min_y = int(min(y_dash_1, y_dash_2, y_dash_3, y_dash_4))
                max_x = int(max(x_dash_1, x_dash_2, x_dash_3, x_dash_4))
                max_y = int(max(y_dash_1, y_dash_2, y_dash_3, y_dash_4))
                box = (min_x, min_y, max_x, max_y)
                region_final = img2.crop(box)
                text = text.replace('\x00', '')
                text = unicodedata.normalize('NFC', text)
                image_id = os.path.splitext(filename)[0] + '_' + str(i).zfill(3)
                image_filepath = os.path.join(args.out_dir, 'data', 'images', os.path.splitext(filename)[0] + '_' + str(i).zfill(3) + '.png')
                writer_id = os.path.splitext(filename)[0].split('-')[0]
                region_final.save(image_filepath, 'png')
                
                text_fh.write(image_id + ' ' + text + '\n')
                utt2spk_fh.write(image_id + ' ' + writer_id + '\n')
                image_fh.write(image_id + ' ' + image_filepath + '\n')
