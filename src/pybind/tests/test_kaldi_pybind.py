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
        return
        # test FloatMatrix
        kp_matrix = kaldi.FloatMatrix(4, 5)

        np_matrix = kp_matrix.numpy()

        np_matrix[2][3] = 2.0

        gold = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 2, 0],
            [0, 0, 0, 0, 0],
        ])
        self.assertTrue((kp_matrix == gold).all())

    def test_matrix_reader_writer(self):
        kp_matrix = kaldi.FloatMatrix(2, 3)
        wspecifier = 'ark,t:test.ark'
        rspecifier = 'ark:test.ark'
        matrix_writer = kaldi.MatrixWriter(wspecifier)

        np_matrix = kp_matrix.numpy()
        np_matrix[0, 0] = 10

        matrix_writer.Write('id_1', kp_matrix)
        matrix_writer.Close()

        matrix_reader = kaldi.SequentialMatrixReader(rspecifier)
        key = matrix_reader.Key()
        self.assertEqual(key, 'id_1')

        value = matrix_reader.Value()
        gold = np.array([[10, 0, 0], [0, 0, 0]])
        self.assertTrue((np.array(value, copy=False) == gold).all())

    def test_matrix_reader_iterator(self):
        kp_matrix = kaldi.FloatMatrix(2, 3)
        wspecifier = 'ark,t:test.ark'
        rspecifier = 'ark:test.ark'
        matrix_writer = kaldi.MatrixWriter(wspecifier)
        matrix_writer.Write('id_1', kp_matrix)
        matrix_writer.Close()

        gold_key_list = ['id_1']
        gold_value_list = [np.array([[0, 0, 0], [0, 0, 0]])]
        for (key, value), gold_key, gold_value in zip(
                kaldi.SequentialMatrixReader(rspecifier), gold_key_list,
                gold_value_list):
            self.assertEqual(key, gold_key)
            self.assertTrue((value == gold_value).all())

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
        self.assertTrue((np.array(reader['id_1']) == gold).all())
        self.assertFalse('id_2' in reader)


if __name__ == '__main__':
    unittest.main()
