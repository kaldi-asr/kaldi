import argparse
import os
import sys
import numpy as np
from scipy import misc
import xml.dom.minidom as minidom

doc = minidom.parse('/Users/ashisharora/Desktop/summer/hcr data/xml/a01-000u.xml')
memoryElem = doc.getElementsByTagName('form')[0]
print memoryElem.getAttribute('writer-id')
outerfolder = memoryElem.getAttribute('id')[0:3]
innerfolder = memoryElem.getAttribute('id')

line_img ='/Users/ashisharora/Desktop/summer/hcr data/lines'
line_img = os.path.join(line_img, outerfolder)
line_img = os.path.join(line_img, innerfolder)
line_img = os.path.join(line_img, innerfolder)
memoryElem = doc.getElementsByTagName('line')

ele = memoryElem[0]
print ele.getAttribute('id')
print ele.getAttribute('text')
img_id = ele.getAttribute('id')[-3:]
file_path = line_img + img_id + '.png'
print file_path
img = misc.imread(file_path)
