import os
import sys
import numpy as np
from scipy import misc

for _set in {"set_a","set_b","set_c","set_d"}:
    data = "/export/b01/babak/IFN-ENIT/ifnenit_v2.0p1e/data/" + _set + "/tif/"
    for i in range(0,len(os.listdir(data))):
        try:
            im = misc.imread(data + os.listdir(data)[i])
            scale_size = 52
            sx = im.shape[1]
            sy = im.shape[0]
            scale = (1.0 * scale_size) / sy
            nx = int(scale_size)
            ny = int(scale * sx)
            im = misc.imresize(im, (nx, ny))
            padding_x = 5
            padding_y = im.shape[0]
            im_pad = np.concatenate((255 * np.ones((padding_y,padding_x), dtype=int), im), axis=1)
            im_pad1 = np.concatenate((im_pad,255 * np.ones((padding_y, padding_x), dtype=int)), axis=1)
            misc.toimage(im_pad1).save("data/normalized/" + os.listdir(data)[i])
        except:
            print "Error!!!"