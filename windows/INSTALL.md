
# Installation instructions for native Windows with Visual Studio

For cygwin installation, see the instructions in `../INSTALL`.

## Notes

* The recipes (in egs/) will not work. There is no commitment to support Windows.
  The Windows port of Kaldi is targeted at experienced developers who want 
  to program their own apps using the kaldi libraries and are able to do 
  the troubleshooting on their own. 
* These instructions are valid November 2017, 
  [Intel® MKL](https://software.intel.com/en-us/intel-mkl) and OpenBLAS are supported
* ATLAS is not supported and I personally have no intention to work on supporting
  it, as it requires whole cygwin environment
* For now (20171121), we do not support CUDA. We might add the support again
  in the future, but for now we do not express any commitment to do so.
  You can still generate solutions with CUDA, but we do not provide any support
  and we didn't test if the solutions work or not.
* While the 32bit project files will still be generated, we don't really
  care if they work or not. They will be removed in the near future.
* The build process was validated using MSVC2017. We do not support earlier 
  releases (i.e. MSVC2015 and older). The reason is the C++11 support is still
  very buggy in the MS compiler.
* We support only openfst-1.6.5 for now.
* I suggest to have git installed -- not only because we will
  use it to download the source codes (you could download archives
  instead of it), but also because the windows version comes
  with a bunch of useful utilities.
* The examples will assume you have installed the git for windows
  and during the installation you chose the GIT Shell to install as well.
  Moreover, all the commands are issued from the same session.

## Steps

## Compiling OpenFST

Skip this section, if you have downloaded OpenFST project from https://github.com/kkm000/openfst.git and it already contains openfst.sln file in the root folder. If it is present you can directly open it with Visual Studio 17 and you do not need CMake.
------------------------- 
For compilation of OpenFST, you will need CMake installed. Simply go to https://cmake.org/download/ and download and install.
Then, in the command line, run the following commands. Be very careful about writing the commands verbatim!

        $ git clone https://github.com/kkm000/openfst.git
        $ cd openfst
        $ mkdir build64
        $ cd build64
        $ cmake -G "Visual Studio 15 2017 Win64" ../
        
The last command will generate output looking similarly to this. Do not try to read too much into specific versions of the programs.

        -- The C compiler identification is MSVC 19.11.25547.0
        -- The CXX compiler identification is MSVC 19.11.25547.0
        -- Check for working C compiler: C:/Program Files (x86)/Microsoft Visual Studio/2017/Community/VC/Tools/MSVC/14.11.25503/bin/Hostx86/x64/cl.exe
        -- Check for working C compiler: C:/Program Files (x86)/Microsoft Visual Studio/2017/Community/VC/Tools/MSVC/14.11.25503/bin/Hostx86/x64/cl.exe -- works
        -- Detecting C compiler ABI info
        -- Detecting C compiler ABI info - done
        -- Check for working CXX compiler: C:/Program Files (x86)/Microsoft Visual Studio/2017/Community/VC/Tools/MSVC/14.11.25503/bin/Hostx86/x64/cl.exe
        -- Check for working CXX compiler: C:/Program Files (x86)/Microsoft Visual Studio/2017/Community/VC/Tools/MSVC/14.11.25503/bin/Hostx86/x64/cl.exe -- works
        -- Detecting CXX compiler ABI info
        -- Detecting CXX compiler ABI info - done
        -- Detecting CXX compile features
        -- Detecting CXX compile features - done
        -- The following ICU libraries were not found:
        --   data (required)
        --   i18n (required)
        --   io (required)
        --   test (required)
        --   tu (required)
        --   uc (required)
        -- Failed to find all ICU components (missing: ICU_INCLUDE_DIR ICU_LIBRARY _ICU_REQUIRED_LIBS_FOUND)
        -- Could NOT find ZLIB (missing: ZLIB_LIBRARY ZLIB_INCLUDE_DIR)
        -- Configuring done
        -- Generating done
        -- Build files have been written to: C:/Users/jtrmal/Documents/openfst/build64

In the directory `build64`, find the file `openfst.sln` and open it using Visual Studio 17. 
------------------------- 

   **Switch the configuration to `debug|Win64` and build the solution.**
   **Do the same for configuration `release|Win64`.**

 If either of the two won't build, you should stop here and start figuring what's different!

## Compiling Kaldi   
   
1. Checkout Kaldi trunk, using [git](https://git-for-windows.github.io/) from https://github.com/kaldi-asr/kaldi.git

   Example:
   
        $ git clone https://github.com/kaldi-asr/kaldi.git kaldi

There are two options to use for BLAS (linear algebra): [Intel® MKL](https://software.intel.com/en-us/intel-mkl) and OpenBLAS. [Intel® MKL](https://software.intel.com/en-us/intel-mkl) is made by Intel and is optimised
for their processors. It isn't free, but you can get [Community Licensing for Intel® Performance Libraries
](https://software.intel.com/sites/campaigns/nest/) or as part of Intel product suite if you [qualify as students, educators, academic researchers, and open source contributors](https://software.intel.com/en-us/qualify-for-free-software). OpenBLAS is free alternative with similar performance.

2. If using [Intel® MKL](https://software.intel.com/en-us/intel-mkl), [install it](https://software.intel.com/en-us/intel-mkl/try-buy).

3. If using OpenBLAS, download the binary packages.

   https://sourceforge.net/projects/openblas

        (kaldi)/tools$ curl -L -O http://sourceforge.net/projects/openblas/files/v0.2.14/OpenBLAS-v0.2.14-Win64-int32.zip
        (kaldi)/tools$ curl -L -O http://sourceforge.net/projects/openblas/files/v0.2.14/mingw64_dll.zip
        (kaldi)/tools$ unzip OpenBLAS-v0.2.14-Win64-int32.zip
        (kaldi)/tools$ unzip mingw64_dll.zip

   **Be careful to download "Win64-int32" and not "Win64-int64"!**

4. **For now, we do not support CUDA, nor provide any kind of assistance in getting it work.**
   If you want enabled [CUDA](http://www.nvidia.com/object/cuda_home_new.html) support, download and install [NVIDIA CUDA SDK](https://developer.nvidia.com/cuda-downloads).
   Be careful and strive for as standard install as possible. The installer
   set certain environment variables on which the MSVC Build rules rely.
   If you call "set" in the command line, you should see:

        (kaldi)/tools $ set | grep CUDA
        CUDA_PATH='C:\Users\Yenda\Downloads\cuda'
        CUDA_PATH_V7_0='C:\Users\Yenda\Downloads\cuda'
        NVCUDASAMPLES7_0_ROOT='C:\Users\Yenda\Downloads\cuda'
        NVCUDASAMPLES_ROOT='C:\Users\Yenda\Downloads\cuda'

   The first one (`CUDA_PATH`) is particularly important.


4. Enter the `(kaldi)/windows` directory

    Example:
    
         (kaldi)/$ cd windows
         (kaldi)/windows $ pwd

5. Copy `variables.props.dev` to `variables.props`.
    Then modify the file `variables.props` to reflect
    the correct paths, using your favorite text editor.
    Don't worry, it's a text file, even though you have to be
    careful to keep the structure itself intact

         (kaldi)/windows $ vim variables.props

    If you plan to use MKL, you can ignore the `OPENBLASDIR` path.
    If you plan to use OpenBLAS, you can ignore the `MKLDIR` path.
    No matter what you plan to use, set `OPENFST*` variable correctly.

6. For OpenBLAS support, copy the file `kaldiwin_openblas.props` to `kaldiwin.props`
7. For MKL support, copy the `kaldiwin_mkl.props` to `kaldiwin.props`

8. Call the script that generates the MSVC solution

         generate_solution.pl --vsver <default|vs2017|vs2015> [--enable-cuda] [--enable-openblas] [--enable-mkl]

    `--enable-mkl` is the default so you shouldn't need to use it. If `--enable-openblas` is passed it disables MKL support.
    CUDA is disabled by default. The default Visual Studio version is 15.0 (Visual Studio 2017). 
    Please note that while we support generating the project for Visual Studio 2015, the C++11 support for that compiler
    is rather sub-par, i.e. it won't probably compile. When choosing Visual Studio 2015, you are on your own!

    For example, for a build using OpenBLAS and VS 2017 you would run:

         (kaldi)/windows$ generate_solution.pl --vsver vs2017 --enable-openblas

    Another example, for OpenBLAS, VS 2017 and CUDA support:

         (kaldi)/windows$ generate_solution.pl --vsver vs2017 --enable-cuda --enable-openblas

9. Run the script (kaldi)/windows/get_version.pl:
        
        (kaldi)/windows$ get_version.pl
  
10. Open the generated solution that was created in a subfolder (kaldi)/kaldiwin_vs<version>_<blas-library> 
	in the visual studio and switch to **Debug|x64** (or **Release|x64**) and build.
   Expect 10 projects to fail, majority of them will fail because of missing include `portaudio.h`. The tests will
   fail to compile too -- this is because of deficiency of the script generate_solution.pl. We might fix it
   later on.

------
NOTE: I'm leaving the information about ATLAS here, for reference (also do not forget to consult the `README.ATLAS`)

(B) either
   (i) compile ATLAS under cygwin [see INSTALL.atlas] and copy
  `kaldiwin_atlas.props` to `kaldiwin.props`

(D)
If you had installed ATLAS, you next have to do this:
[assuming you are one level above this directory]

    cd kaldiwin_vs10_auto/

Type the following (these commands were done from cygwin): note that these
commands are a bit wasteful of disk; you could alternatively ensure that
`[root]/tools/ATLAS/cygwin_build/install/lib/` is always on your path when you
run the binaries.

    mkdir -p Debug Release
    cp ../tools/ATLAS/cygwin_build/install/lib/lib_atlas.dll Debug
    cp ../tools/ATLAS/cygwin_build/install/lib/lib_atlas.dll Release

Then build the project with Visual Studio.
