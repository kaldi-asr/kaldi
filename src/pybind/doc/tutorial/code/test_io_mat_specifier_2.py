#!/usr/bin/env python3

import numpy as np
import kaldi

wspecifier = 'ark,scp:/tmp/feats.ark,/tmp/feats.scp'

with kaldi.MatrixWriter(wspecifier) as writer:
    m = np.arange(6).reshape(2, 3).astype(np.float32)
    writer.Write(key='foo', value=m)

    g = kaldi.FloatMatrix(2, 2)
    g[0, 0] = 10
    g[1, 1] = 20
    writer.Write('bar', g)

rspecifier = 'scp:/tmp/feats.scp'
with kaldi.SequentialMatrixReader(rspecifier) as reader:
    for key, value in reader:
        assert key in ['foo', 'bar']
        if key == 'foo':
            np.testing.assert_array_equal(value.numpy(), m)
        else:
            np.testing.assert_array_equal(value.numpy(), g.numpy())

with kaldi.RandomAccessMatrixReader(rspecifier) as reader:
    assert 'foo' in reader
    assert 'bar' in reader
    np.testing.assert_array_equal(reader['foo'].numpy(), m)
    np.testing.assert_array_equal(reader['bar'].numpy(), g.numpy())
