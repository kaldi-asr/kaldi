#!/usr/bin/env python3

import kaldi

wspecifier = 'ark,scp:/tmp/ali.ark,/tmp/ali.scp'

with kaldi.IntVectorWriter(wspecifier) as writer:
    writer.Write(key='foo', value=[1, 2, 3])
    writer.Write('bar', [10, 20])

# Note that you do NOT need to close the file.

rspecifier = 'scp:/tmp/ali.scp'
with kaldi.SequentialIntVectorReader(rspecifier) as reader:
    for key, value in reader:
        print(key, value)

rspecifier = 'scp:/tmp/ali.scp'
with kaldi.RandomAccessIntVectorReader(rspecifier) as reader:
    value1 = reader['foo']
    print(value1)

    value2 = reader['bar']
    print(value2)
