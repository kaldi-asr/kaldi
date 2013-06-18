Intro
-----
The goal of this project is to test
Kaldi decoding pipeline called from Python

Prerequisities
--------------

 * Install *cffi*! See the docs
[http://cffi.readthedocs.org/](http://cffi.readthedocs.org/) for more info.
 * Build kaldi with `OpenBlAS` support and `-fPIC` flags in `CXXFLAGS` or `EXTRA_CXXFLAGS` in the main Makefile
 * Before building Kaldi build `OpenBLAS` and openfst by 

 ```sh
 cd kaldi-trunk/tools
 make openblas
 ```

 and 

```sh
 cd kaldi-trunk/tools
# replace line in kaldi-trunk/tools/Makefile by following "patch" change line 37!
# switching from disable-shared -> enable-shared
*** Makefile 
************
*** 34,38 ****

openfst-1.3.2/Makefile: openfst-1.3.2/.patched
		cd openfst-1.3.2/; \
!		./configure --prefix=`pwd` --enable-static --disable-shared --enable-far --enable-ngram-fsts

--- 34,38 ----

openfst-1.3.2/Makefile: openfst-1.3.2/.patched
		cd openfst-1.3.2/; \
!		./configure --prefix=`pwd` --enable-static --enable-shared --enable-far --enable-ngram-fsts

# and build it
make openfst_tgt
```


Running and building examples
-----------------------------

In order to build shared libraries and run C test binaries
```sh
$make all
```
To run `run.py` set up specify where are the shared libraries. E.g. by running from `kaldi-trunk/src/python-kaldi-decoding`

```sh
LD_LIBRARY_PATH=`pwd`/../../tools/OpenBLAS:`pwd`/../../tools/openfst/lib:`pwd` ./run.py
```


Remarks on linking
-------
 * [How to use dlopen](http://www.isotton.com/devel/docs/C++-dlopen-mini-HOWTO/C++-dlopen-mini-HOWTO.html)
 * [Stackoverflow little off topic explanation](http://stackoverflow.com/questions/12762910/c-undefined-symbols-when-loading-shared-library-with-dlopen)
 * [http://kaldi.sourceforge.net/matrixwrap.html](See Missing the ATLAS implementation of  CLAPACK)
 * I spent a lot of time to set right linking. 
    I was linking `lapack` libraries instead of `lapack_atlas`.
    I was getting error `undefined symbol: clapack_dgetrf`
