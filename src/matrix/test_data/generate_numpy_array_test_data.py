#!/usr/bin/env python3

# Copyright 2020   Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import numpy as np


def main():
    data = np.arange(8).astype(np.float32)
    np.save('float_vector.npy', data)
    np.save('double_vector.npy', data.astype(np.float64))

    data = data.reshape(2, 4).astype(np.float32)
    np.save('float_matrix.npy', data)
    np.save('double_matrix.npy', data.astype(np.float64))


if __name__ == '__main__':
    main()
