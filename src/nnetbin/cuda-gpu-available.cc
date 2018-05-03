// nnetbin/cuda-gpu-available.cc

// Copyright 2015 Brno University of Technology (author: Karel Vesely)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef _MSC_VER
  #include <unistd.h>
  #include <errno.h>
#endif

#include "base/kaldi-common.h"
#include "cudamatrix/cu-device.h"
#include "cudamatrix/cu-matrix.h"

using namespace kaldi;

#if HAVE_CUDA == 1
/**
 * With incorrect CUDA setup, this will trigger "invalid device function" error.
 */
void TestGpuComputation() {
  CuMatrix<BaseFloat> m(100,100);
  m.SetRandn();
  m.ApplySoftMaxPerRow(m);
}
#endif

int main(int argc, char *argv[]) try {
  char hostname[100] = "UNKNOWN-HOSTNAME";
#if !defined(_MSC_VER) && !defined(__CYGWIN__)
  if (gethostname(hostname, 100)) {
    KALDI_WARN << "Cannot get hostname, " << strerror(errno);
  }
#endif
  KALDI_LOG << std::endl << std::endl
    << "### IS CUDA GPU AVAILABLE? '" << hostname << "' ###";
#if HAVE_CUDA == 1
  CuDevice::Instantiate().SelectGpuId("yes");
  fprintf(stderr, "### HURRAY, WE GOT A CUDA GPU FOR COMPUTATION!!! ##\n\n");
  fprintf(stderr, "### Testing CUDA setup with a small computation "
                  "(setup = cuda-toolkit + gpu-driver + kaldi):\n");
  // the test of setup by computation,
  try {
    TestGpuComputation();
  } catch (const std::exception &e) {
    fprintf(stderr, "%s\n", e.what());
    KALDI_LOG << "...\n"
      << "### The CUDA setup is wrong! "
      << "(\"invalid device function\" == problem with 'compute capability' "
      << "in compiled kaldi)\n"
      << "### Before posting the error to forum, please try following:\n"
      << "### 1) update kaldi & cuda-toolkit (& GPU driver),\n"
      << "### 2) re-run 'src/configure',\n"
      << "### 3) re-compile kaldi by 'make clean; make -j depend; make -j'\n"
      << "###\n"
      << "### If the problem persists, please send us your:\n"
      << "### - GPU model name, cuda-toolkit version, driver version "
      << "(run nvidia-smi), variable $(CUDA_ARCH) from src/kaldi.mk";
    return -1;
  }
  fprintf(stderr, "### Test OK!\n");
  return 0;
#else
  std::cerr
    << "### CUDA WAS NOT COMPILED IN! ###" << std::endl
    << "To support CUDA, you must run 'configure' on a machine "
    << "that has the CUDA compiler 'nvcc' available.";
  return 1;
#endif
} catch (const std::exception &e) {
  fprintf(stderr, "%s\n", e.what());
  KALDI_LOG << "...\n"
    << "### WE DID NOT GET A CUDA GPU!!! ###\n"
    << "### If your system has a 'free' CUDA GPU, try re-installing "
    << "latest 'CUDA toolkit' from NVidia (this updates GPU drivers too).\n"
    << "### Otherwise 'nvidia-smi' shows the status of GPUs:\n"
    << "### - The versions should match ('NVIDIA-SMI' and 'Driver Version'), "
    << "otherwise reboot or reload kernel module,\n"
    << "### - The GPU should be unused "
    << "(no 'process' in list, low 'memory-usage' (<100MB), low 'gpu-fan' (<30%)),\n"
    << "### - You should see your GPU (burnt GPUs may disappear from the list until reboot),";
  return -1;
}

