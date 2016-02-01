# Copyright 2016  Tsinghua University (Author: Chao Liu).  Apache 2.0.
  
def energy(list mat):
    cdef float e
    cdef int i, j, l
    l = len(mat)
    for i in range(l):
        j = mat[i]
        e += j * j
    e /= l
    return e

def mix(list mat, list noise, int pos, double scale):
    cdef len_noise, len_mat, i, x, y
    ret = []
    len_noise = len(noise)
    len_mat = len(mat)
    for i in range(len_mat):
        x = mat[i]
        y = int(x + scale * noise[pos])
        if y > 32767:
            y = 32767
        elif y < -32768:
            y = -32768
        ret.append(y)
        pos = (pos + 1) % len_noise
    return pos, ret
