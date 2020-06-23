The build process (how Kaldi is compiled)
=========================================

This page describes in general terms how the Kaldi build process works. See also `External matrix libraries <pages/api-undefined.md#matrixwrap>`_ for an explanation of how the matrix code uses external libraries and the linking errors that can arise from this; `Downloading and installing Kaldi <pages/api-undefined.md#install>`_ may also be of interest.

Build process on Windows
------------------------

The build process for Windows is separate from the build process for UNIX-like systems and is described in ``windows/INSTALL`` (tested some time ago with Windows 7 and Microsoft Visual Studio 2013). We use scripts to create the Visual Studio 10.0 solution file. There are two options for the math library on Windows: either Intel MKL, or use Cygwin to compile ATLAS. Detailed instructions are provided. However, note that the Windows setup is becoming out of date and is not regularly tested, and not all the may compile.

How our configure script works (for UNIX variants)
--------------------------------------------------
The ``"configure"`` script, located in ``src/``, should be run by typing ``./configure``. This script takes various options. For instance, you can run   ``./configure --shared`` if you want to build the setup with shared libraries instead of static libraries (this can make the binaries smaller), and it also takes various options to enable you to specify different math libraries, for instance, to use OpenBlas. Look at the top of the script to see some example command lines. The configure script currently supports are Cygwin, Darwin (which is Apple's version of BSD UNIX), and Linux. The configure script produces a file called ``kaldi.mk``. This file will be textually included by the Makefiles in the subdirectories (see below).

Editing kaldi.mk
----------------

Changes that you might want to make to kaldi.mk after running "configure" are the following:

*   Changing the debug level:

    -   The default is ``"-O1"``
    -   Easy to debug binaries can be enabled by uncommenting the options ``"-O0 -DKALDI_PARANOID"``.
    -   For maximum speed and no checking, you can replace the ``"-O0 -DKALDI_PARANOID"`` options with ``"-O2 -DNDEBUG"`` or ``"-O3 -DNDEBUG"``

*   Changing the default precision

    -   To test algorithms in double precision (e.g. if you suspect that roundoff is affecting your results), you can change 0 to 1 in the option ``-DKALDI_DOUBLEPRECISION=0``.

*   Suppressing warnings

    -   To suppress warnings caused by signed-unsigned comparisons in OpenFst code, you can add ``-Wno-sign-compare`` to ``CXXFLAGS``

It is also possible to modify ``kaldi.mk`` to use different math libraries (e.g. to use CLAPACK instead of ATLAS versions of LAPACK functions) and to link to math libraries dynamically instead of statically, but this is quite complicated and we can't give any generic instructions that will enable you to do this (see `External matrix libraries <pages/api-undefined.md#matrixwrap>`_ to understand the compilation process for our math code). It will probably be easier to work with the options to configure script instead.

Targets defined by the Makefiles
--------------------------------

The targets defined by the Makefiles are:

*   ``"make depend"`` will rebuild the dependencies. It's a good idea to run this before building the toolkit. If the .depend files gets out of date (because you haven't run "make depend"), you may get errors that look like this: ``make[1]: *** No rule to make target '/usr/include/foo/bar', needed by 'baz.o'.  Stop.``

*   ``"make all"`` (or just ``"make"``) will compile all the code, including testing code.

*   ``"make test"`` will run the testing code (useful for making sure the build worked on your system, and that you have not introduced bugs)

*   ``"make clean"`` will remove all compiled binaries, .o (object) files and .a (archive) files.

*   ``"make valgrind"`` will run the test programs under valgrind to check for memory leaks.

*   ``"make cudavalgrind"`` will run the test program (in cudamatrix) to check for memory leaks for machine with GPU card which support NVIDIA CUDA and OS with CUDA installation.

Where do the compiled binaries go?
----------------------------------

Currently, the Makefiles do not put the compiled binaries in a special place; they just leave them in the directories where the corresponding code is. Currently, binaries exist in the directories ending with ``bin``, e.g. ``"bin/"``, ``"gmmbin/"``, ``"featbin/"``, ``"fstbin/"``, and ``"lmbin/"`` and so on, all of which are subdirectories of ``"src/"``. In the future, we may designate a single place to put all the binaries.

How our Makefiles work
----------------------

Currently, the file ``src/Makefil``e just invokes the Makefiles in all the source sub-directories (``src/base``, ``src/matrix`` and so on). These directories have their own Makefiles, all of which share a common structure. They all include the line:

   ``include ../kaldi.mk``

This is like an ``#include`` line in C (it includes the text of ``kaldi.mk``). When reading ``kaldi.mk``, bear in mind that it is to be invoked from one directory below where it is actually located (it is located in ``src/``). An example of what the ``kaldi.mk`` file looks like is as follows. This is for a Linux system; we have removed some rules relating to valgrind that are not very important.

.. code-block:: makefile

    ATLASLIBS = /usr/local/lib/liblapack.a /usr/local/lib/libcblas.a \
              /usr/local/lib/libatlas.a /usr/local/lib/libf77blas.a
    CXXFLAGS = -msse -Wall -I.. \
          -DKALDI_DOUBLEPRECISION=0 -msse2 -DHAVE_POSIX_MEMALIGN \
         -DHAVE_EXECINFO_H=1 -rdynamic -DHAVE_CXXABI_H \
          -DHAVE_ATLAS -I ../../tools/ATLAS/include \
           -I ../../tools/openfst/include \
          -g -O0 -DKALDI_PARANOID
    LDFLAGS = -rdynamic
    LDLIBS = ../../tools/openfst/lib/libfst.a -ldl $(ATLASLIBS) -lm
    CC = g++
    CXX = g++
    AR = ar
    AS = as
    RANLIB = ranlib

So ``kaldi.mk`` is responsible for setting up include paths, defining preprocessor variables, setting compile options, linking with libraries, and so on.

Which platforms has Kaldi been compiled on?
-------------------------------------------
We have compiled Kaldi on Windows, Cygwin, various flavors of Linux (including Ubuntu, CentOS, Debian, Red Hat and SUSE), and Darwin. We recommend you use ``g++`` version 4.7 or above, although other compilers such as ``llvm`` and Intel's ``icc`` are also known to work.
