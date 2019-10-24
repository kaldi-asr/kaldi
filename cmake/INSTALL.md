# Install Instruction

Execute following commands in the repo root.

## Build with Old Style Make Generator
```bash
mkdir -p build && cd build
cmake  -DCMAKE_INSTALL_PREFIX=../dist .. # configure
cmake --build . --target install -- -j6  # build && install
```

## Build with Ninja Generator
``` bash
mkdir -p build && cd build
cmake -GNinja -DCMAKE_INSTALL_PREFIX=../dist ..
cmake --build . --target install
```

After built, you can find all installed files in <your_repo_root>/dist

# For Advance User

Follow options are currently available:

| Variable               | Available Options       | Default  |
| ---------------------- | ----------------------- | -------- |
| MATHLIB                | OpenBLAS,MKL,Accelerate | OpenBLAS |
| KALDI_BUILD_EXE        | ON,OFF                  | ON |
| KALDI_BUILD_TEST       | ON,OFF                  | ON |
| KALDI_USE_PATCH_NUMBER | ON,OFF                  | OFF |
| BUILD_SHARED_LIBS      | ON,OFF                  | OFF |

NOTE: Currently, BUILD_SHARED_LIBS does not work on Windows due to some symbols
      (variables) are not properly exported.
