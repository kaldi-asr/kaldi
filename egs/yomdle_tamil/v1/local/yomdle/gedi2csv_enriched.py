#!/usr/bin/env python3

'''
Convert GEDI-type bounding boxes to CSV format
'''

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
    ''' Initialize the extractor'''
    def __init__(self, logger, args):
        self._logger = logger
        self._args = args

    '''
    Segment image with GEDI bounding box information
    '''
    def csvfile(self, coords, polys, baseName, pgrot):

        ''' for writing the files '''
        writePath = self._args.outputDir
        writePath = os.path.join(writePath,'')
        if os.path.isdir(writePath) != True:
            os.makedirs(writePath)

        rotlist = []

        header=['ID','name','col1','row1','col2','row2','col3','row3','col4','row4','confidence','truth','pgrot','bbrot','qual','script','text_type']
        conf=100
        write_ctr = 0
        if len(coords) == 0 and len(polys) == 0:
            self._logger.info('Found %s with no text content',(baseName))
            print('...Found %s with no text content' % (baseName))
            return
            
        strPos = writePath + baseName

        ''' for each group of coordinates '''
        for i in coords:

            [id,x,y,w,h,degrees,text,qual,script,text_type] = i
            contour = np.array([(x,y),(x+w,y),(x+w,y+h),(x,y+h)])
            """First rotate around upper left corner based on orientationD keyword"""
            M, rot = Rotate2D(contour, np.array([x,y]), degrees*pi/180)
            rot = np.int0(rot)

            # rot is the 8 points rotated by degrees
            # pgrot is the rotation after extraction, so save
            # save rotated points to list or array
            rot = np.reshape(rot,(-1,1)).T
            c1,r1,c2,r2,c3,r3,c4,r4 = npbox2string(rot)
            
            bbrot = degrees
            rotlist.append([id,baseName + '_' + id + '.png',c1,r1,c2,r2,c3,r3,c4,r4,conf,text,pgrot,bbrot,qual,script,text_type])

        # if there are polygons, first save the text
        for j in polys:
            arr = []
            [id,poly_val,text,qual,script,text_type] = j
            for i in poly_val:
                arr.append(eval(i))

            contour = np.asarray(arr)
            convex = cv2.convexHull(contour)
            rect = cv2.minAreaRect(convex)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            box = np.reshape(box,(-1,1)).T
            c1,r1,c2,r2,c3,r3,c4,r4 = npbox2string(box)
            
            bbrot = 0.0
            rotlist.append([id,baseName + '_' + id + '.png',c1,r1,c2,r2,c3,r3,c4,r4,conf,text,pgrot,bbrot,qual,script,text_type])
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
    if os.path.isdir(writePath) != True:
        os.makedirs(writePath)
    ''' Setup logging '''
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
    
    '''
    Get all XML files in the directory and sub folders
    '''
    for root, dirnames, filenames in os.walk(args.inputDir, followlinks=True):
        for file in filenames:
            if file.lower().endswith('.xml'):
                fullName = os.path.join(root,file)
                baseName = os.path.splitext(fullName)
                fileCnt += 1
                ''' read the XML file '''
                tree = ET.parse(fullName)
                gedi_root = tree.getroot()
                child = gedi_root.findall('gedi:DL_DOCUMENT',namespaces)[0]
                totalpages = int(child.attrib['NrOfPages'])
                coordinates=[]
                polygons = []
                if args.ftype == 'boxed':
                    fileTypeStr = 'col'
                elif args.ftype == 'transcribed':
                    fileTypeStr = 'Text_Content'
                else:
                    print('Filetype must be either boxed or transcribed!')
                    logger.info('Filetype must be either boxed or transcribed!')
                    sys.exit(-1)
                
                if args.quality == 'both':
                    qualset = {'Regular','Low-Quality'}
                elif args.quality == 'low':
                    qualset = {'Low-Quality'}
                elif args.quality == 'regular':
                    qualset = {'Regular'}
                else:
                    print('Quality must be both, low or regular!')
                    logger.info('Quality must be both, low or regular!')
                    sys.exit(-1)
                    
                    

                ''' and for each page '''
                for i, pgs in enumerate(child.iterfind('gedi:DL_PAGE',namespaces)):
                        
                    if 'GEDI_orientation' not in pgs.attrib:
                        pageRot=0
                    else:
                        pageRot = int(pgs.attrib['GEDI_orientation'])
                        logger.info(' PAGE ROTATION %s, %s' % (fullName, str(pageRot)))

                    ''' find children for each page '''
                    for zone in pgs.findall('gedi:DL_ZONE',namespaces):

                        if zone.attrib['gedi_type']=='Text' and zone.attrib['Type'] in \
                            ('Machine_Print','Confusable_Allograph','Handwriting') and zone.attrib['Quality'] in qualset:
                            if zone.get('polygon'):
                                keyCnt+=1
                                polygons.append([zone.attrib['id'],zone.get('polygon').split(';'),
                                                 zone.get('Text_Content'),zone.get('Quality'),zone.get('Script'),zone.get('Type')])
                            elif zone.get(fileTypeStr) != None:
                                keyCnt+=1
                                coord = [zone.attrib['id'],int(zone.attrib['col']),int(zone.attrib['row']),
                                                    int(zone.attrib['width']), int(zone.attrib['height']),
                                                    float(zone.get('orientationD',0.0)),
                                                    zone.get('Text_Content'),zone.get('Quality'),zone.get('Script'),zone.get('Type')]
                                coordinates.append(coord)

                if len(coordinates) > 0 or len(polygons) > 0:
                    line_write_ctr += gtconverter.csvfile(coordinates, polygons, os.path.splitext(file)[0], pageRot)
                else:
                    print('...%s has no applicable content' % (baseName[0]))

    print('complete...total files %d, lines written %d' % (fileCnt, line_write_ctr))


''' Args and defaults '''
def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--inputDir', type=str, help='Input directory', required=True)
    parser.add_argument('--outputDir', type=str, help='Output directory', required=True)
    parser.add_argument('--ftype', type=str, help='GEDI file type (either "boxed" or "transcribed")', default='transcribed')
    parser.add_argument('--quality', type=str, help='GEDI file quality (either "both" or "low" or "regular")', default='regular')
    parser.add_argument('--log', type=str, help='Log directory', default='./GEDI2CSV_enriched.log')

    return parser.parse_args(argv)

''' Run '''
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))



    


