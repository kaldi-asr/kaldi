#!/usr/bin/env python3

# Copyright 2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import os
import shutil
import tempfile
import unittest

import kaldi


def generate_test_lexicon(d):
    s = 'foo f o o\n'
    s += 'bar b a r\n'

    filename = os.path.join(d, 'lexicon.txt')
    with open(filename, 'w') as f:
        f.write(s)


def generate_test_tokens(d):
    s = '''<eps> 0
<blk> 1
a 2
b 3
f 4
o 5
r 6
'''
    filename = os.path.join(d, 'tokens.txt')
    with open(filename, 'w') as f:
        f.write(s)


def generate_test_text(d):
    s = 'utt1 foo bar bar\n'
    s += 'utt2 bar\n'

    filename = os.path.join(d, 'text')
    with open(filename, 'w') as f:
        f.write(s)


class ConvertTextToLablesTest(unittest.TestCase):

    def test(self):
        d = tempfile.mkdtemp()

        generate_test_lexicon(d)
        generate_test_tokens(d)
        generate_test_text(d)

        cmd = '''
        python3 ./local/convert_text_to_labels.py \
                --lexicon-filename {lexicon} \
                --tokens-filename {tokens} \
                --dir {dir}
        '''.format(lexicon=os.path.join(d, 'lexicon.txt'),
                   tokens=os.path.join(d, 'tokens.txt'),
                   dir=d)

        os.system(cmd)

        rspecifier = 'scp:{}/labels.scp'.format(d)

        reader = kaldi.SequentialIntVectorReader(rspecifier)

        expected_labels = dict()
        expected_labels['utt1'] = [3, 4, 4, 2, 1, 5, 2, 1, 5]
        expected_labels['utt2'] = [2, 1, 5]

        for key, value in reader:
            self.assertTrue(key in expected_labels)
            self.assertEqual(value, expected_labels[key])

        reader.Close()

        shutil.rmtree(d)


if __name__ == '__main__':
    unittest.main()
