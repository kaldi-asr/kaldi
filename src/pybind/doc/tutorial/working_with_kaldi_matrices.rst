
Working with Kaldi's Matrices
=============================

This tutorial demonstrates how to use Kaldi's matrices in Python.

The following table summarizes the matrix types in Kaldi that
have been wrapped to Python.

+----------------------+--------------------+
|     Kaldi Types      |    Python Types    |
+======================+====================+
| ``Vector<float>``    | ``FloatVector``    |
+----------------------+--------------------+
| ``SubVector<float>`` | ``FloatSubVector`` |
+----------------------+--------------------+
| ``Matrix<float>``    | ``FloatMatrix``    |
+----------------------+--------------------+
| ``SubMatrix<float>`` | ``FloatSubMatrix`` |
+----------------------+--------------------+

All of the Python types above can be converted
to Numpy arrays without copying the underlying memory
buffers. In addition, ``FloatSubVector`` and ``FloatSubMatrix``
can be constructed directly from Numpy arrays without data copying.

.. Note::

    Only the **single** precision floating point type has been wrapped
    to Python.

FloatVector
-----------

The following code shows how to use ``FloatVector`` in Python.

.. literalinclude:: ./code/test_float_vector.py
   :caption: Example usage of ``FloatVector``
   :language: python
   :linenos:

Its output is

.. code-block:: sh

    [ 10 0 0 ]

    [ 10 20 0 ]

.. literalinclude:: ./code/test_float_vector.py
   :language: python
   :lineno-start: 1
   :lines: 1
   :linenos:

This is a hint that it needs ``python3``. At present, we support
only ``python3`` in Kaldi Pybind.

.. literalinclude:: ./code/test_float_vector.py
   :language: python
   :lineno-start: 3
   :lines: 3
   :linenos:

This imports the Kaldi Pbyind package. If you encounter an import error, please
make sure that ``PYTHONPATH`` has been set to point to ``KALDI_ROOT/src/pybind``.


.. literalinclude:: ./code/test_float_vector.py
   :language: python
   :lineno-start: 5
   :lines: 5
   :linenos:

This creates an object of ``FloatVector`` containing 3 elements
which are by default initialized to zero.

.. literalinclude:: ./code/test_float_vector.py
   :language: python
   :lineno-start: 7
   :lines: 7
   :linenos:

This prints the value of the ``FloatVector`` object to the console.
Note that you use operator ``()`` in C++ to access the elements of
a ``Vector<float>`` object; Python code uses ``[]``.

.. literalinclude:: ./code/test_float_vector.py
   :language: python
   :lineno-start: 9
   :lines: 9
   :linenos:

This creates a Numpy array object ``g`` from ``f``. No memory is copied here.
``g`` shares the underlying memory with ``f``.

.. literalinclude:: ./code/test_float_vector.py
   :language: python
   :lineno-start: 10
   :lines: 10
   :linenos:

This also changes ``f`` since it shares the same memory with ``g``.
You can verify that ``f`` is changed from the output.

.. HINT::

  We recommend that you invoke the ``numpy()`` method of a ``FloatVector``
  object to get a Numpy ndarray object and to manipulate this Numpy object.
  Since it shares the underlying memory with the ``FloatVector`` object,
  every operation you perform to this Numpy ndarray object is visible to
  the ``FloatVector`` object.

FloatSubVector
--------------

The following code shows how to use ``FloatSubVector`` in Python.

.. literalinclude:: ./code/test_float_subvector.py
   :caption: Example usage of ``FloatSubVector``
   :language: python
   :linenos:

Its output is

.. code-block:: sh

    [ 0. 20. 30.]
    [  0. 100.  30. ]

.. literalinclude:: ./code/test_float_subvector.py
   :language: python
   :lineno-start: 6
   :lines: 6-7
   :linenos:

This creates a ``FloatSubVector`` object ``f`` from a Numpy ndarray object ``v``.
No memory is copied here. ``f`` shares the underlying memory with ``v``.
Note that the ``dtype`` of ``v`` has to be ``np.float32``; otherwise, you will
get a runtime error when creating ``f``.

.. literalinclude:: ./code/test_float_subvector.py
   :language: python
   :lineno-start: 9
   :lines: 9
   :linenos:

This uses ``[]`` to access the elements of ``f``. It also changes ``v``
since ``f`` shares the same memory with ``v``.

.. literalinclude:: ./code/test_float_subvector.py
   :language: python
   :lineno-start: 12
   :lines: 12
   :linenos:

This create a Numpy ndarray object ``g`` from ``f``. No memory is copied here.
``g`` shares the same memory with ``f``.

.. literalinclude:: ./code/test_float_subvector.py
   :language: python
   :lineno-start: 13
   :lines: 13
   :linenos:

This also changes ``v`` because of memory sharing.

FloatMatrix
-----------

The following code shows how to use ``FloatMatrix`` in Python.

.. literalinclude:: ./code/test_float_matrix.py
   :caption: Example usage of ``FloatMatrix``
   :language: python
   :linenos:

Its output is

.. code-block:: sh

  [
   0 0 0
   0 0 100 ]

  [
   200 0 0
   0 0 100 ]

.. literalinclude:: ./code/test_float_matrix.py
   :language: python
   :lineno-start: 5
   :lines: 5
   :linenos:

This creates an object ``f`` of ``FloatMatrix`` with
2 rows and 3 columns.

.. literalinclude:: ./code/test_float_matrix.py
   :language: python
   :lineno-start: 6
   :lines: 6
   :linenos:

This uses ``[]`` to access the elements of ``f``.

.. literalinclude:: ./code/test_float_matrix.py
   :language: python
   :lineno-start: 7
   :lines: 7
   :linenos:

This prints the value of ``f`` to the console.

.. literalinclude:: ./code/test_float_matrix.py
   :language: python
   :lineno-start: 9
   :lines: 9
   :linenos:

This creates a Numpy ndarray object ``g`` from ``f``.
No memory is copied here. ``g`` shares the underlying memory
with ``f``.

.. literalinclude:: ./code/test_float_matrix.py
   :language: python
   :lineno-start: 10
   :lines: 10
   :linenos:

This also changes ``f`` due to memory sharing.

FloatSubMatrix
--------------

The following code shows how to use ``FloatSubMatrix`` in Python.

.. literalinclude:: ./code/test_float_submatrix.py
   :caption: Example usage of ``FloatSubMatrix``
   :language: python
   :linenos:

Its output is

.. code-block:: sh

    [[  1.   2.   3.]
     [ 10.  20. 100.]]

    [[200.   2.   3.]
     [ 10.  20. 100.]]

.. literalinclude:: ./code/test_float_submatrix.py
   :language: python
   :lineno-start: 6
   :lines: 6-7
   :linenos:

This creates an object ``f`` of ``FloatSubMatrix`` from
a Numpy ndarray object ``m``. ``f`` shares the underlying
memory with ``m``. Note that the ``dtype`` of ``m`` has
to be ``np.float32``. Otherwise you will get a runtime error.

.. literalinclude:: ./code/test_float_submatrix.py
   :language: python
   :lineno-start: 9
   :lines: 9-10
   :linenos:

This uses ``[]`` to access the elements of ``f``. Note that
``m`` is also changed due to memory sharing.

.. literalinclude:: ./code/test_float_submatrix.py
   :language: python
   :lineno-start: 13
   :lines: 13
   :linenos:

This creates a Numpy ndarray object from ``f``. No memory
is copied here. ``g`` shares the underlying memory with ``f``.

.. literalinclude:: ./code/test_float_submatrix.py
   :language: python
   :lineno-start: 14
   :lines: 14
   :linenos:

This changes ``f`` and ``m``.
