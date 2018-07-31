#!/usr/bin/env python

# Copyright 2017 Yiwen Shao
# Apache 2.0
import sys
import struct

""" Provide an function to write bmp image to stdout with the input two dimensional
    list.
    usage: bmp_encoder(data, width, height)
    For instance:
    data = [[0, 255, 0, 0, 0, 255], [255, 0, 0, 0, 255, 255]]
    bmp_encoder(data, 2, 2) will write a 2*2 rgb bmp image to the stdout. 
    image :[ green    blue
             red      yellow ]
"""

def bmp_encoder(data, width, height):
    bmp_header = [0] * 16
    header_size = 54 # the bmp header is 54 bytes long
    bmp_header[0] = struct.pack('<B', 0x42) # 1 byte, B
    bmp_header[1] = struct.pack('<B', 0x4D) # 1 byte, D
    bmp_header[2] = struct.pack('<I',0) # 4 bytes, fullsize of file, fill in later
    bmp_header[3] = struct.pack('<I',0) # 4 bytes, This data is reserved but can just be set to 0
    bmp_header[4] = struct.pack('<I',header_size) # 4 bytes, Pixel offset, 54 bytes here
    bmp_header[5] = struct.pack('<I',40) # 4 bytes, BITMAPINFOHEADER
    bmp_header[6] = struct.pack('<I',width) # 4 bytes, width
    bmp_header[7] = struct.pack('<I',height) # 4 bytes, width
    bmp_header[8] = struct.pack('<H',1) # 2 bytes, The number of color planes, must be set to 1
    bmp_header[9] = struct.pack('<H',24) # 2 bytes, The number of bits per pixel. 
    bmp_header[10] = struct.pack('<I',0) # 4 bytes, Disable Compression
    bmp_header[11] = struct.pack('<I',0) # 4 bytes, Size of raw pixel data, fill in later
    bmp_header[12] = struct.pack('<I',2835) # 4 bytes, horizontal resolution. Just leave it at 2835
    bmp_header[13] = struct.pack('<I',2835) # 4 bytes, vertical resolution. Just leave it at 2835.
    bmp_header[14] = struct.pack('<I',0) # 4 bytes, The number of colors, leave at 0 to default to all colors
    bmp_header[15] = struct.pack('<I',0) # 4 bytes, The important colors, leave at 0 to default to all colors
    
    bmp_body = []
    for i in range(height):
        for j in range(width):
            # bmp is in GBR and from bottom to top
            bmp_body.append(struct.pack('<B', data[height -1 - i][j * 3 + 2]))
            bmp_body.append(struct.pack('<B', data[height -1 - i][j * 3 + 1]))
            bmp_body.append(struct.pack('<B', data[height -1 - i][j * 3 + 0]))
        while len(bmp_body) % 4 != 0:
            bmp_body.append(struct.pack('<B',0)) # pad each row with bytes at the end till its length is a multiple of 4
    
    rawpixel_size = len(bmp_body)        
    file_size = rawpixel_size + header_size
    bmp_header[2] = struct.pack('<I',file_size)
    bmp_header[11] = struct.pack('<I',rawpixel_size)
    
    output = bmp_header + bmp_body
    for i in range(len(output)):
        sys.stdout.write(output[i])




