#!/usr/bin/env python3

# Copyright 2020   Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import numpy as np


def main():
    data = np.arange(8)
    np.save('float_vector.npy', data.astype('<f4'))
    np.save('double_vector.npy', data.astype('<f8'))

    np.save('float_vector_big_endian.npy', data.astype('>f4'))
    np.save('double_vector_big_endian.npy', data.astype('>f8'))

    data = data.reshape(2, 4)
    np.save('float_matrix.npy', data.astype('<f4'))
    np.save('double_matrix.npy', data.astype('<f8'))

    np.save('float_matrix_big_endian.npy', data.astype('>f4'))
    np.save('double_matrix_big_endian.npy', data.astype('>f8'))


if __name__ == '__main__':
    main()
