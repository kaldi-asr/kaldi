#!/usr/bin/env python3

"""
GEDI2CSV
Convert GEDI-type bounding boxes to CSV format

GEDI Format Example:
<GEDI xmlns= GEDI_version= GEDI_date=>
    <USER name= date= dateFormat="mm/dd/yyyy hh:mm"> </USER>
    <DL_DOCUMENT src= NrOfPages= docTag=>
        <DL_PAGE gedi_type= src= pageID= width= height=>
            <DL_ZONE gedi_type= id=  Illegible= polygon=  Language= Text_Content= text_raw=> </DL_ZONE>
        </DL_PAGE>
    </DL_DOCUMENT>
</GEDI>

CSV Format Example
ID,name,col1,row1,col2,row2,col3,row3,col4,row4,confidence,truth,pgrot,bbrot,qual,script,lang
0,chinese_scanned_books_0001_0.png,99,41,99,14,754,14,754,41,100,凡我的邻人说是好的，有一大部分在我灵魂中却,0,0.0,0,,zh-cn
"""

import logging
import os
import sys
import time
import glob
import csv
import imghdr
from PIL import Image
import argparse
import pdb
import cv2
import numpy as np
import xml.etree.ElementTree as ET

sin = np.sin
cos = np.cos
pi = np.pi

def Rotate2D(pts, cnt, ang=90):
    M = np.array([[cos(ang),-sin(ang)],[sin(ang),cos(ang)]])
    res = np.dot(pts-cnt,M)+cnt
    return M, res

def npbox2string(npar):
    if np.shape(npar)[0] != 1:
        print('Error during CSV conversion\n')
    c1,r1 = npar[0][0],npar[0][1]
    c2,r2 = npar[0][2],npar[0][3]
    c3,r3 = npar[0][4],npar[0][5]
    c4,r4 = npar[0][6],npar[0][7]

    return c1,r1,c2,r2,c3,r3,c4,r4

# cv2.minAreaRect() returns a Box2D structure which contains following detals - ( center (x,y), (width, height), angle of rotation )
# Get 4 corners of the rectangle using cv2.boxPoints()

class GEDI2CSV(object):

    """ Initialize the extractor"""
    def __init__(self, logger, args):
        self._logger = logger
        self._args = args

    """
    Segment image with GEDI bounding box information
    """
    def csvfile(self, coords, polys, baseName, pgrot):

        """ for writing the files """
        writePath = self._args.outputDir
        if os.path.isdir(writePath) != True:
            os.makedirs(writePath)

        rotlist = []

        header=['ID','name','col1','row1','col2','row2','col3','row3','col4','row4','confidence','truth','pgrot','bbrot','qual','script','lang']
        conf=100
        pgrot = 0
        bbrot = 0
        qual = 0
        script = ''

        write_ctr = 0
        if len(coords) == 0 and len(polys) == 0:
            self._logger.info('Found %s with no text content',(baseName))
            print('...Found %s with no text content' % (baseName))
            return

        strPos = writePath + baseName

        for j in polys:
            try:
                arr = []
                [id,poly_val,text,qual,lang] = j
                script=None
                #print(j)
                for i in poly_val:
                    if len(i.strip()) > 0:
                        #print(i)
                        arr.append(eval(i))

                contour = np.asarray(arr)
                #print(contour)
                convex = cv2.convexHull(contour)
                rect = cv2.minAreaRect(convex)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                box = np.reshape(box,(-1,1)).T
                c1,r1,c2,r2,c3,r3,c4,r4 = npbox2string(box)

                bbrot = 0.0

                rotlist.append([id,baseName + '_' + id + '.png',c1,r1,c2,r2,c3,r3,c4,r4,conf,text,pgrot,bbrot,qual,script,lang])

            except:
                print('...polygon error %s, %s' % (j, baseName))
                continue

        # then write out all of list to file
        with open(strPos + ".csv", "w", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for row in rotlist:
                writer.writerow(row)
                write_ctr += 1

        return write_ctr


def main(args):

    startTime = time.clock()

    writePath = args.outputDir
    print('write to %s' % (writePath))
    if os.path.isdir(writePath) != True:
        os.makedirs(writePath)

    """ Setup logging """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if args.log:
        handler = logging.FileHandler(args.log)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    gtconverter = GEDI2CSV(logger, args)
    namespaces = {"gedi" : "http://lamp.cfar.umd.edu/media/projects/GEDI/"}
    keyCnt=0

    fileCnt = 0
    line_write_ctr = 0
    line_error_ctr = 0
    file_error_ctr = 0
    """
    Get all XML files in the directory and sub folders
    """
    print('reading %s' % (args.inputDir))
    for root, dirnames, filenames in os.walk(args.inputDir, followlinks=True):
        for file in filenames:
            if file.lower().endswith('.xml'):
                fullName = os.path.join(root,file)
                baseName = os.path.splitext(fullName)

                fileCnt += 1

                try:
                    """ read the XML file """
                    tree = ET.parse(fullName)
                except:
                    print('...ERROR parsing %s' % (fullName))
                    file_error_ctr += 1
                    continue

                gedi_root = tree.getroot()
                child = gedi_root.findall('gedi:DL_DOCUMENT',namespaces)[0]
                totalpages = int(child.attrib['NrOfPages'])
                coordinates=[]
                polygons = []

                """ and for each page """
                for i, pgs in enumerate(child.iterfind('gedi:DL_PAGE',namespaces)):

                    if 'GEDI_orientation' not in pgs.attrib:
                        pageRot=0
                    else:
                        pageRot = int(pgs.attrib['GEDI_orientation'])
                        logger.info(' PAGE ROTATION %s, %s' % (fullName, str(pageRot)))

                    """ find children for each page """
                    for zone in pgs.findall('gedi:DL_ZONE',namespaces):

                        if zone.attrib['gedi_type']=='Text' :
                            if zone.get('polygon'):
                                keyCnt+=1
                                polygons.append([zone.attrib['id'],zone.get('polygon').split(';'),
                                                 zone.get('Text_Content'),zone.get('Illegible'),zone.get('Language')])
                            else:
                                print('...Not polygon')


                if len(coordinates) > 0 or len(polygons) > 0:
                    line_write_ctr += gtconverter.csvfile(coordinates, polygons, os.path.splitext(file)[0], pageRot)
                else:
                    print('...%s has no text content' % (baseName[0]))


    print('complete...total files %d, lines written %d, img errors %d, line error %d' % (fileCnt, line_write_ctr, file_error_ctr, line_error_ctr))


def parse_arguments(argv):
    """ Args and defaults """
    parser = argparse.ArgumentParser()

    parser.add_argument('--inputDir', type=str, help='Input directory', default='/data/YOMDLE/final_arabic/xml')
    parser.add_argument('--outputDir', type=str, help='Output directory', default='/exp/YOMDLE/final_arabic/csv_truth/')
    parser.add_argument('--log', type=str, help='Log directory', default='/exp/logs.txt')

    return parser.parse_args(argv)


if __name__ == '__main__':
    """ Run """
    main(parse_arguments(sys.argv[1:]))
