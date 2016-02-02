
# Installation instructions for native Windows with Visual Studio

For cygwin installation, see the instructions in `../INSTALL`.

## Notes

* These instructions are valid June 2015, MKL and OpenBLAS are supported
* ATLAS is not supported and I personally have no intention to work on supporting
  it, as it requires whole cygwin environment
* We now (20150613) support CUDA on Windows as well. The build was
  tested on CUDA 7.0. It is possible that the compilation fails
  for significantly older CUDA SDK (less than, say, 5.0)
  Please not that CUDA support for windows is not really that usefull,
  because, the speed benefit during decoding is not large. And for training
  one would have to re-implement the while training pipeline (as the
  bash script wouldn't most probably work)
* While the 32bit project files will still be generated, we don't really
  care if they work or not. They will be removed in the near future.
* The build process were validated using MSVS2013 and partially (MKL only) using MSVS2015-rc
* We support only openfst-1.3.x for now.
* I suggest to have git installed -- not only because we will
  use it to download the source codes (you could download archives
  instead of it), but also because the windows version comes
  with a bunch of useful utilities.
* The examples will assume you have installed the git for windows
  and during the installation you chose the GIT Shell to install as well.
  Moreover, all the commands are issued from the same session.

## Steps

1. Checkout Kaldi trunk, either using the svn from the url https://svn.code.sf.net/p/kaldi/code/trunk
   or using git from https://github.com/kaldi-asr/kaldi.git

   Example:
   
        $ git clone https://github.com/kaldi-asr/kaldi.git kaldi

2. Enter the `(kaldi)/tools` directory in the freshly
   checked-out kaldi repo. All following actions should
   be taken in the tools dir.

   Example:
   
        $ cd (kaldi)/tools
        (kaldi)/tools$ pwd

3. Use git to clone the OpenFST(win) from
       
        https://github.com/jtrmal/openfstwin-1.3.4.git

   Example:
   
        (kaldi)/tools$ git clone https://github.com/jtrmal/openfstwin-1.3.4.git openfst

4. Download pthread-win32 (or wget or curl)

   https://sourceforge.net/projects/pthreads4w/

        (kaldi)/tools$ wget http://downloads.sourceforge.net/project/pthreads4w/pthreads-w32-2-9-1-release.zip
        (kaldi)/tools$ mkdir pthreads; cd pthreads
        (kaldi)/tools/pthreads$ unzip ../pthreads-w32-2-9-1-release.zip

5. Use patch (or you can use git patch) to patch the OpenFST(win).

   The patch location is `tools/extras/openfstwin-1.3.4.patch`

   Example:
   
        (kaldi)/tools$ cd openfst
        (kaldi)/tools/openfst$ patch -p1 <../extras/openfstwin-1.3.4.patch

   If you get this error: `Assertion failed: hunk, file ../patch-2.5.9-src/patch.c, line 354`
   it is because the `patch.c` file should have Windows line endings (CRLF) rather than Unix ones (LF).
   
There are two options to use for BLAS (linear algebra): MLK and OpenBLAS. MLK is made by Intel and is optimised
for their processors. Unfortunately it isn't free. OpenBLAS is free alternative with similar performance.

6. If using MLK, install it.

7. If using OpenBLAS, download the binary packages.

   https://sourceforge.net/projects/openblas

        (kaldi)/tools$ wget http://sourceforge.net/projects/openblas/files/v0.2.14/OpenBLAS-v0.2.14-Win64-int32.zip
        (kaldi)/tools$ wget http://sourceforge.net/projects/openblas/files/v0.2.14/mingw64_dll.zip
        (kaldi)/tools$ unzip OpenBLAS-v0.2.14-Win64-int32.zip
        (kaldi)/tools$ unzip mingw64_dll.zip

   **Be careful to download "Win64-int32" and not "Win64-int64"!**

8. If you want enabled CUDA support, download and install NVidia CUDA SDK.
   Be careful and strive for as standard install as possible. The installer
   set certain environment variables on which the MSVC Build rules rely.
   If you call "set" in the command line, you should see:

        (kaldi)/tools $ set | grep CUDA
        CUDA_PATH='C:\Users\Yenda\Downloads\cuda'
        CUDA_PATH_V7_0='C:\Users\Yenda\Downloads\cuda'
        NVCUDASAMPLES7_0_ROOT='C:\Users\Yenda\Downloads\cuda'
        NVCUDASAMPLES_ROOT='C:\Users\Yenda\Downloads\cuda'

   The first one (`CUDA_PATH`) is particularly important.

9. Open the OpenFST solution in VS

   * for VS 2013, the correct solution is in VS2012 directory
   * for VS 2014, the correct solution is in VS2014 directory

   **Switch the configuration to `debug|x64` and build the solution.**

   **Do the same for configuration `release|x64`.**

   If either of the two won't build, you should stop here and start figuring what's different!

10. Enter the `(kaldi)/windows` directory

    Example:
    
         (kaldi)/tools/openfst$ cd ../../windows
         (kaldi)/windows $ pwd

11. Modify the file `variables.props` to reflect
    the correct paths, using your favorite text editor.
    Don't worry, it's a text file, even though you have to be
    careful to keep the structure itself intact

         (kaldi)/windows $ vim variables.props

    If you plan to use MKL, you can ignore the `OPENBLASDIR` path.
    If you plan to use OpenBLAS, you can ignore the `MKLDIR` path.
    No matter what you plan to use, set both the `OPENFST*` and `PTHREADW`
    variables correctly

12. For OpenBLAS support, copy the file `kaldiwin_openblas.props` to `kaldiwin.props`
13. For MKL support, you don't have to do anything, it should work out of the box.
    When you need to switch from OpenBLAS to MKL, copy the `kaldiwin_mkl.props`
    to `kaldiwin.props`

14. Call the script that generates the MSVC solution

         generate_solution.pl --vsver <default|vs2013|vs2015> [--enable-cuda] [--enable-openblas] [--enable-mlk]

    `--enable-mlk` is the default so you shouldn't need to use it. If `--enable-openblas` is passed it disables MLK support.
    CUDA is disabled by default. The default Visual Studio version is 11.0 (Visual Studio 2012).

    For example, for a build using OpenBLAS and VS 2015 you would run:

         (kaldi)/tools$ generate_solution.pl --vsver vs2015 --enable-openblas

    Another example, for OpenBLAS, VS 2013 and CUDA support:

         (kaldi)/tools$ generate_solution.pl --vsver vs2013 --enable-cuda --enable-openblas

15. Open the generated solution in the visual studio and switch to Debug|x64 (or Release|x64) and build.
   Expect 10 projects to fail, majority of them will fail because of missing include `portaudio.h`

------
NOTE: I'm leaving the information about ATLAS here, for reference (also do not forget to consult the README.ATLAS)

(B) either
   (i) compile ATLAS under cygwin [see INSTALL.atlas] and copy
  kaldiwin_atlas.props  to kaldiwin.props

(D)
If you had installed ATLAS, you next have to do this:
[assuming you are one level above this directory]

    cd kaldiwin_vs10_auto/

Type the following (these commands were done from cygwin): note that these
commands are a bit wasteful of disk; you could alternatively ensure that
[root]/tools/ATLAS/cygwin_build/install/lib/ is always on your path when you
run the binaries.

    mkdir -p Debug Release
    cp ../tools/ATLAS/cygwin_build/install/lib/lib_atlas.dll Debug
    cp ../tools/ATLAS/cygwin_build/install/lib/lib_atlas.dll Release

Then build the project with Visual Studio.
