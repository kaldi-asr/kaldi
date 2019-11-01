import unittest
import numpy as np
import os
import sys

# Add .. to the PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))
import kaldi_pybind as kp

class TestKaldiPybind(unittest.TestCase):
    def test_float_vector(self):
        # test FloatVector
        print("=====Testing FloatVector in kaldi_pybind=====")
        kp_vector = kp.FloatVector(10)
        print("kp_vector = kp.FloatVector(10)")
        print(kp_vector)

        print("np_array = np.array(kp_vector, copy=False)")
        np_array = np.array(kp_vector, copy = False)
        print(np_array)

        print("np_array[5:10] = 2.0")
        np_array[5:] = 2.0
        print(np_array)

    def test_float_matrxi(self):
        # test FloatMatrix
        print("=====Testing FloatVector in kaldi_pybind=====")
        kp_matrix = kp.FloatMatrix(4,5)
        print("kp_matrix = kp.FloatMatrix(4,5)")
        print(kp_matrix)

        print("np_matrix = np.array(kp_matrix, copy=False)")
        np_matrix = np.array(kp_matrix, copy = False)
        print(np_matrix)

        print("np_matrix[2][3] = 2.0")
        np_matrix[2][3] = 2.0
        print(np_matrix)

    def test_matrix_reader_writer(self):
        print("=====Testing Matrix Reader/Writer in kaldi_pybind=====")
        kp_matrix = kp.FloatMatrix(2,3)
        wspecifier = "ark,t:test.ark"
        rspecifier = "ark:test.ark"
        matrix_writer = kp.BaseFloatMatrixWriter_Matrix(wspecifier)
        print("Write id: 'id_1'")
        print("Write matrix: [0 0 0; 0 0 0]")
        matrix_writer.Write("id_1", kp_matrix)
        matrix_writer.Close()

        matrix_reader = kp.SequentialBaseFloatMatrixReader_Matrix(rspecifier)
        print("Read id: {}".format(matrix_reader.Key()))
        print("Read matrix: {}".format(matrix_reader.Value()))


if __name__ == '__main__':
    unittest.main()
