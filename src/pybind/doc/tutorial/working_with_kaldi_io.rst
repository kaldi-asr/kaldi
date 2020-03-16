
Working with Kaldi's IO
=======================

This tutorial shows how to read and write
ark/scp files in Python.


Reading and Writing Alignment Information
-----------------------------------------

The following class can be used to write alignment
information to files:

- ``IntVectorWriter``

And the following classes can be used to read alignment
information from files:

- ``SequentialIntVectorReader``
- ``RandomAccessIntVectorReader``

The following code shows how to write and read
alignment information.


.. literalinclude:: ./code/test_io_ali.py
   :caption: Example of reading and writing align information
   :language: python
   :linenos:

Its output is

.. code-block:: sh

    foo [1, 2, 3]
    bar [10, 20]
    [1, 2, 3]
    [10, 20]

The output of the following command

.. code:: console

  $ copy-int-vector scp:/tmp/ali.scp ark,t:-

is

.. code-block:: sh

    copy-int-vector scp:/tmp/ali.scp ark,t:-
    foo 1 2 3
    bar 10 20
    LOG (copy-int-vector[5.5.792~1-f5875b]:main():copy-int-vector.cc:83) Copied 2 vectors of int32.

.. literalinclude:: ./code/test_io_ali.py
   :language: python
   :lineno-start: 5
   :lines: 5
   :linenos:

It creates a write specifier ``wspecifier`` indicating that
the alignment information is going to be written into files
``/tmp/ali.ark`` and ``/tmp/ali.scp``.

.. literalinclude:: ./code/test_io_ali.py
   :language: python
   :lineno-start: 8
   :lines: 8
   :linenos:

It writes a list ``[1, 2, 3]`` to file with ``key == foo``.
Note that you can use keyword arguments while writing.

.. literalinclude:: ./code/test_io_ali.py
   :language: python
   :lineno-start: 9
   :lines: 9
   :linenos:

It writes a list ``[10, 20]`` to file with ``key == bar``.

.. literalinclude:: ./code/test_io_ali.py
   :language: python
   :lineno-start: 10
   :lines: 10
   :linenos:

It closes the writer.

.. Note::

  It is a best practice to close the file when it is no longer needed.

.. literalinclude:: ./code/test_io_ali.py
   :language: python
   :lineno-start: 12
   :lines: 12-13
   :linenos:

It creates a **sequential** reader.

.. literalinclude:: ./code/test_io_ali.py
   :language: python
   :lineno-start: 15
   :lines: 15-16
   :linenos:

It uses a ``for`` loop to iterate the reader.

.. literalinclude:: ./code/test_io_ali.py
   :language: python
   :lineno-start: 18
   :lines: 18
   :linenos:

It closes the reader.

.. literalinclude:: ./code/test_io_ali.py
   :language: python
   :lineno-start: 20
   :lines: 20
   :linenos:

It creates a **random access** reader.

.. literalinclude:: ./code/test_io_ali.py
   :language: python
   :lineno-start: 21
   :lines: 21-22
   :linenos:

It reads the value of ``foo`` and prints it out.

.. literalinclude:: ./code/test_io_ali.py
   :language: python
   :lineno-start: 24
   :lines: 24-25
   :linenos:

It reads the value of ``bar`` and prints it out.

.. literalinclude:: ./code/test_io_ali.py
   :language: python
   :lineno-start: 26
   :lines: 26
   :linenos:

Finally, it closes the reader.


------

The following code example achieves the same effect as the above one
except that you do not need to close the file manually.

.. literalinclude:: ./code/test_io_ali_2.py
   :caption: Example of reading and writing align information using ``with``
   :language: python
   :linenos:

Reading and Writing Matrices
----------------------------

Using xfilename
^^^^^^^^^^^^^^^

The following code demonstrates how to read and write
``FloatMatrix`` using ``xfilename``.

.. literalinclude:: ./code/test_io_mat_xfilename.py
   :language: python
   :caption: Example of reading and writing matrices with xfilename
   :linenos:

The output of the above program is

.. code-block:: sh

  [
   10 0
   0 20 ]

.. literalinclude:: ./code/test_io_mat_xfilename.py
   :language: python
   :lineno-start: 5
   :lines: 5-7
   :linenos:

It creates a ``FloatMatrix`` and sets its diagonal to ``[10, 20]``.

.. literalinclude:: ./code/test_io_mat_xfilename.py
   :language: python
   :lineno-start: 9
   :lines: 9-10
   :linenos:

It writes the matrix to ``/tmp/lda.mat`` in binary format.
``kaldi.write_mat`` is used to write the matrix
to the specified file. You can specify whether it is
written in binary format or text format.

.. literalinclude:: ./code/test_io_mat_xfilename.py
   :language: python
   :lineno-start: 12
   :lines: 12-13
   :linenos:

It reads the matrix back and prints it to the console.
Note that you do not need to specify whether the file to
read is in binary or not. ``kaldi.read_mat`` will figure
out the format automatically.


Using specifier
^^^^^^^^^^^^^^^

The following code demonstrates how to read and write
``FloatMatrix`` using ``specifier``.

.. literalinclude:: ./code/test_io_mat_specifier.py
   :language: python
   :caption: Example of reading and writing matrices with specifier
   :linenos:

.. literalinclude:: ./code/test_io_mat_specifier.py
   :language: python
   :lineno-start: 6
   :lines: 6-8
   :linenos:

This creates a matrix writer.

.. literalinclude:: ./code/test_io_mat_specifier.py
   :language: python
   :lineno-start: 10
   :lines: 10-11
   :linenos:

It creates a Numpy array object of type ``np.float32`` and
writes it to file with the key ``foo``. Note that the type
of the Numpy array has to be of type ``np.float32``.
The program throws if the type is not ``np.float32``.

.. literalinclude:: ./code/test_io_mat_specifier.py
   :language: python
   :lineno-start: 13
   :lines: 13-16
   :linenos:

It creates a ``FloatMatrix`` and writes it to file
with the key ``bar``.

.. HINT::

  ``kaldi.MatrixWriter`` accepts Numpy array objects of
  type ``np.float32`` as well as ``kaldi.FloatMatrix`` objects.

.. literalinclude:: ./code/test_io_mat_specifier.py
   :language: python
   :lineno-start: 18
   :lines: 18
   :linenos:

It closes the writer.

.. literalinclude:: ./code/test_io_mat_specifier.py
   :language: python
   :lineno-start: 20
   :lines: 20-21
   :linenos:

It creates a **sequential** matrix reader.

.. literalinclude:: ./code/test_io_mat_specifier.py
   :language: python
   :lineno-start: 21
   :lines: 21-27
   :linenos:

It uses a ``for`` loop to iterate the sequential reader.

.. literalinclude:: ./code/test_io_mat_specifier.py
   :language: python
   :lineno-start: 29
   :lines: 29
   :linenos:

It closes the sequential reader.

.. literalinclude:: ./code/test_io_mat_specifier.py
   :language: python
   :lineno-start: 31
   :lines: 31
   :linenos:

It creates a **random access** matrix reader.

.. literalinclude:: ./code/test_io_mat_specifier.py
   :language: python
   :lineno-start: 32
   :lines: 32-33
   :linenos:

It uses ``in`` to test whether the reader contains a given key.

.. literalinclude:: ./code/test_io_mat_specifier.py
   :language: python
   :lineno-start: 34
   :lines: 34-35
   :linenos:

It uses ``[]`` to read the value of a specified key.

.. literalinclude:: ./code/test_io_mat_specifier.py
   :language: python
   :lineno-start: 36
   :lines: 36
   :linenos:

It closes the random access reader.

------

The following code example achieves the same effect as the above one
except that you do not need to close the file manually.

.. literalinclude:: ./code/test_io_mat_specifier_2.py
   :caption: Example of reading and writing FloatMatrix using ``with``
   :language: python
   :linenos:
