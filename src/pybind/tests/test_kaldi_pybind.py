#!/usr/bin/env python3

import unittest
import numpy as np
import os
import sys

# Add .. to the PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
import kaldi_pybind as kp
import kaldi


class TestKaldiPybind(unittest.TestCase):

    def test_float_vector(self):
        # test FloatVector
        print("=====Testing FloatVector in kaldi_pybind=====")
        kp_vector = kp.FloatVector(5)
        print("kp_vector = kp.FloatVector(5)")
        print(kp_vector)

        print("np_array = np.array(kp_vector, copy=False)")
        np_array = np.array(kp_vector, copy=False)
        print(np_array)

        print("np_array[2:5] = 2.0")
        np_array[2:] = 2.0
        print(np_array)

        gold = np.array([0, 0, 2, 2, 2])
        self.assertTrue((np_array == gold).all())

    def test_float_matrix(self):
        # test FloatMatrix
        print("=====Testing FloatMatrix in kaldi_pybind=====")
        kp_matrix = kp.FloatMatrix(4, 5)
        print("kp_matrix = kp.FloatMatrix(4, 5)")
        print(kp_matrix)

        print("np_matrix = np.array(kp_matrix, copy=False)")
        np_matrix = np.array(kp_matrix, copy=False)
        print(np_matrix)

        print("np_matrix[2][3] = 2.0")
        np_matrix[2][3] = 2.0
        print(np_matrix)

        gold = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 2, 0],
            [0, 0, 0, 0, 0],
        ])
        self.assertTrue((np_matrix == gold).all())

    def test_matrix_reader_writer(self):
        print("=====Testing Matrix Reader/Writer in kaldi_pybind=====")
        kp_matrix = kp.FloatMatrix(2, 3)
        wspecifier = "ark,t:test.ark"
        rspecifier = "ark:test.ark"
        matrix_writer = kp.BaseFloatMatrixWriter(wspecifier)
        print("Write id: 'id_1'")
        print("Write matrix: [0 0 0; 0 0 0]")
        matrix_writer.Write("id_1", kp_matrix)
        matrix_writer.Close()

        matrix_reader = kp.SequentialBaseFloatMatrixReader(rspecifier)
        key = matrix_reader.Key()
        print("Read id: {}".format(key))
        self.assertEqual(key, "id_1")

        value = matrix_reader.Value()
        gold = np.array([[0, 0, 0], [0, 0, 0]])
        print("Read matrix: {}".format(value))
        self.assertTrue((np.array(value, copy=False) == gold).all())

    def test_matrix_reader_iterator(self):
        print("=====Testing Matrix Reader Iterator=====")
        kp_matrix = kp.FloatMatrix(2, 3)
        wspecifier = "ark,t:test.ark"
        rspecifier = "ark:test.ark"
        matrix_writer = kp.BaseFloatMatrixWriter(wspecifier)
        print("Write id: 'id_1'")
        print("Write matrix: [0 0 0; 0 0 0]")
        matrix_writer.Write("id_1", kp_matrix)
        matrix_writer.Close()

        gold_key_list = ["id_1"]
        gold_value_list = [np.array([[0, 0, 0], [0, 0, 0]])]
        for (key, value), gold_key, gold_value in zip(
                kaldi.ReaderIterator(kaldi.SequentialMatrixReader(rspecifier)),
                gold_key_list, gold_value_list):
            self.assertEqual(key, gold_key)
            self.assertTrue((value == gold_value).all())
            print(key, value)

    def test_matrix_reader_dict(self):
        print("=====Testing Matrix Reader Dict=====")
        kp_matrix = kp.FloatMatrix(2, 3)
        wspecifier = "ark,t:test.ark"
        rspecifier = "ark:test.ark"
        matrix_writer = kp.BaseFloatMatrixWriter(wspecifier)
        print("Write id: 'id_1'")
        print("Write matrix: [0 0 0; 0 0 0]")
        matrix_writer.Write("id_1", kp_matrix)
        matrix_writer.Close()

        reader_dict = kaldi.ReaderDict(
            kaldi.RandomAccessMatrixReader(rspecifier))
        gold = np.array([[0, 0, 0], [0, 0, 0]])
        self.assertTrue("id_1" in reader_dict)
        self.assertTrue((np.array(reader_dict["id_1"]) == gold).all())
        self.assertFalse("id_2" in reader_dict)


if __name__ == '__main__':
    unittest.main()
