# Install Instruction

Execute following commands in the repo root.

## Build with Old Style Make Generator
```bash
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=../dist .. # configure
cmake --build . --target install -- -j8 # build && install, substitude -j8 with /m:8 if you are on Windows
```

## Build with Ninja Generator
``` bash
mkdir -p build && cd build
cmake -GNinja -DCMAKE_INSTALL_PREFIX=../dist ..
cmake --build . --target install
```

After built, you can find all installed files in <your_repo_root>/dist

# For Advance Configuration

Follow options are currently available:

| Variable               | Available Options         | Default  |
| ---------------------- | ------------------------- | -------- |
| MATHLIB                | OpenBLAS, MKL, Accelerate | OpenBLAS |
| KALDI_BUILD_EXE        | ON,OFF                    | ON |
| KALDI_BUILD_TEST       | ON,OFF                    | ON |
| KALDI_USE_PATCH_NUMBER | ON,OFF                    | OFF |
| BUILD_SHARED_LIBS      | ON,OFF                    | OFF |

Append `-D<Variable>=<Value>` to the configure command to use it, e.g.,
`-DKALDI_BUILD_TEST=OFF` will disable building of test executables. For more
information, please refers to
[CMake Documentation](https://cmake.org/cmake/help/latest/manual/cmake.1.html).
For quick learning CMake usage, LLVM's short introuction will do the trick:
[Basic CMake usage](https://llvm.org/docs/CMake.html#usage),
[Options and variables](https://llvm.org/docs/CMake.html#options-and-variables),
[Frequently-used CMake variables](https://llvm.org/docs/CMake.html#frequently-used-cmake-variables).

NOTE 1: Currently, BUILD_SHARED_LIBS does not work on Windows due to some symbols
        (variables) are not properly exported.

NOTE 2: For scripts users, since you are doing an out of source build, and the
        install destination is at your disposal, the `$PATH` is not configured
        properly in this case. Scripts will not work out of box. See how `$PATH`
        is modified in [path.sh](../egs/wsj/s5/path.sh). You should add
        `<installation_path>/bin` to your `$PATH` before running any scripts.
