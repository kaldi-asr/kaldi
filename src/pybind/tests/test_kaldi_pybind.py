#!/usr/bin/env python3

import unittest
import numpy as np
import os
import sys

# Add .. to the PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
import kaldi


class TestKaldiPybind(unittest.TestCase):

    def test_float_vector(self):
        # test FloatVector
        kp_vector = kaldi.FloatVector(5)

        np_array = kp_vector.numpy()
        self.assertIsInstance(np_array, np.ndarray)

        np_array[2:] = 2.0

        gold = np.array([0, 0, 2, 2, 2])
        self.assertTrue((kp_vector == gold).all())

    def test_float_matrix(self):
        # test FloatMatrix
        kp_matrix = kaldi.FloatMatrix(4, 5)

        kp_matrix[2, 3] = 2.0

        gold = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 2, 0],
            [0, 0, 0, 0, 0],
        ])
        np.testing.assert_array_equal(kp_matrix.numpy(), gold)

    def test_matrix_reader_writer(self):
        kp_matrix = kaldi.FloatMatrix(2, 3)
        wspecifier = 'ark,t:test.ark'
        rspecifier = 'ark:test.ark'
        matrix_writer = kaldi.MatrixWriter(wspecifier)

        kp_matrix[0, 0] = 10

        matrix_writer.Write('id_1', kp_matrix)
        matrix_writer.Close()

        matrix_reader = kaldi.SequentialMatrixReader(rspecifier)
        key = matrix_reader.Key()
        self.assertEqual(key, 'id_1')

        value = matrix_reader.Value()
        gold = np.array([[10, 0, 0], [0, 0, 0]])
        np.testing.assert_array_equal(value.numpy(), gold)

        matrix_reader.Close()

        # test with context manager
        kp_matrix[0, 0] = 20
        with kaldi.MatrixWriter(wspecifier) as writer:
            writer.Write('id_2', kp_matrix)
        with kaldi.SequentialMatrixReader(rspecifier) as reader:
            key = reader.Key()
            self.assertEqual(key, 'id_2')
            value = reader.Value()
            gold = np.array([[20, 0, 0], [0, 0, 0]])
            np.testing.assert_array_equal(value.numpy(), gold)
        
        os.remove('test.ark')

    def test_matrix_reader_iterator(self):
        kp_matrix = kaldi.FloatMatrix(2, 3)
        wspecifier = 'ark,t:test.ark'
        rspecifier = 'ark:test.ark'
        matrix_writer = kaldi.MatrixWriter(wspecifier)
        matrix_writer.Write('id_1', kp_matrix)
        matrix_writer.Close()

        gold_key_list = ['id_1']
        gold_value_list = [np.array([[0, 0, 0], [0, 0, 0]])]
        reader = kaldi.SequentialMatrixReader(rspecifier)
        for (key, value), gold_key, gold_value in zip(reader, gold_key_list,
                                                      gold_value_list):
            self.assertEqual(key, gold_key)
            np.testing.assert_array_equal(value.numpy(), gold_value)
        reader.Close()
        os.remove('test.ark')

    def test_matrix_random_access_reader(self):
        kp_matrix = kaldi.FloatMatrix(2, 3)
        wspecifier = 'ark,t:test.ark'
        rspecifier = 'ark:test.ark'
        matrix_writer = kaldi.MatrixWriter(wspecifier)
        matrix_writer.Write('id_1', kp_matrix)
        matrix_writer.Close()

        reader = kaldi.RandomAccessMatrixReader(rspecifier)
        gold = np.array([[0, 0, 0], [0, 0, 0]])
        self.assertTrue('id_1' in reader)

        np.testing.assert_array_equal(reader['id_1'].numpy(), gold)
        self.assertFalse('id_2' in reader)
        reader.Close()
        os.remove('test.ark')


if __name__ == '__main__':
    unittest.main()
