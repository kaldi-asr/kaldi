Other Kaldi utilities
=========================

This page gives an overview of various categories of utility functions that we use in Kaldi code. This excludes important utilities that have been dealt with in their own sections, including the `matrix library <pages/api-undefined.md#matrix>`_\ , `I/O <pages/api-undefined.md#io>`_\ , `logging and error-reporting <pages/api-undefined.md#error>`_\ , and `command-line parsing <pages/api-undefined.md#parse_options>`_.

Text utilities
--------------

In ``text-utils.h`` are various functions for manipulating strings, mostly used in parsing. Important ones include the templated function :cpp:`ConvertStringToInteger()`, and the overloaded :cpp:`ConvertStringToReal()` functions which are defined for float and double. There is also the :cpp:`SplitStringToIntegers()` template whose output is a vector of integers, and :cpp:`SplitStringToVector()` which splits a string into a vector of strings.

STL utilities
-------------

In ``stl-utils.h`` are templated functions for manipulating STL types. A commonly used one is :cpp:`SortAndUniq()`, which sorts and removes duplicates from a vector (of an arbitrary type). The function :cpp:`CopySetToVector()` copies the elements of a set into a vector, and is part of a larger category of similar functions that move data between sets, vectors and maps (see a list in ``stl-utils.h``). There are also the hashing-function types :cpp:`VectorHasher` (for vectors of integers) and :cpp:`StringHasher` (for strings); these are for use with the STL :cpp:`unordered_map` and :cpp:`unordered_set` templates. Another commonly used function is :cpp:`DeletePointers()`, which deletes pointers in a :cpp:`std::vector` of pointers, and sets them to :cpp:`NULL`.

Math utilities
--------------

In kaldi-math.h, apart from a number of standard #defines which are provided in case they are not in the system header math.h, there are some math utility functions. These include most importantly:


*   Functions for random number generation: :cpp:`RandInt()`, :cpp:`RandGauss()`, :cpp:`RandPoisson()`.

*   :cpp:`LogAdd()` and :cpp:`LogSub()` functions

*   Functions for testing and asserting approximate math (in)equalities, i.e. :cpp:`ApproxEqual()`, :cpp:`AssertEqual()`, :cpp:`AssertGeq()` and :cpp:`AssertLeq()`.

Other utilities
---------------

In const-integer-set.h is a class ConstIntegerSet that stores a set of integers in an efficient way and allows fast querying. The caveat is that the set cannot be changed after initializing the object. This is used e.g. in decision-tree code. Depending on the value of the integers in the set, it may store them internally as :cpp:`vector<bool>` or as a sorted vector of integers.

A class Timer for timing programs in a platform-independent way is in timer.h.

Other utility-type functions and classes are in ``simple-io-funcs.h`` and ``hash-list.h``, but these have more specialized uses. Some additional utility functions and macros, mostly quite specialized, that the `matrix <pages/api-undefined.md#matrix>`_ code depends on, are in ``kaldi-utils.h``; these include things like byte swapping, memory alignment, and mechanisms for compile-time assertions (useful in templates).
