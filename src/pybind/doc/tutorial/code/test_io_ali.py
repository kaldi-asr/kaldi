#!/usr/bin/env python3

import kaldi

wspecifier = 'ark,scp:/tmp/ali.ark,/tmp/ali.scp'

writer = kaldi.IntVectorWriter(wspecifier)
writer.Write(key='foo', value=[1, 2, 3])
writer.Write('bar', [10, 20])
writer.Close()

rspecifier = 'scp:/tmp/ali.scp'
reader = kaldi.SequentialIntVectorReader(rspecifier)

for key, value in reader:
    print(key, value)

reader.Close()

reader = kaldi.RandomAccessIntVectorReader(rspecifier)
value1 = reader['foo']
print(value1)

value2 = reader['bar']
print(value2)
reader.Close()
