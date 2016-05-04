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

using namespace kaldi;

int main(int argc, char *argv[]) try {
  char hostname[100] = "UNKNOWN-HOSTNAME";
#ifndef _MSC_VER
  if (gethostname(hostname, 100)) {
    KALDI_WARN << "Cannot get hostname, " << strerror(errno);
  }
#endif
  std::cerr
    << "### IS CUDA GPU AVAILABLE? '"
    << hostname << "' ###" << std::endl;
#if HAVE_CUDA == 1
  CuDevice::Instantiate().SelectGpuId("yes");
  std::cerr
    << "### HURRAY, WE GOT A CUDA GPU FOR COMPUTATION!!! ###"
    << std::endl;
  return 0;
#else
  std::cerr
    << "### CUDA WAS NOT COMPILED IN! ###" << std::endl
    << "To support CUDA, you must run 'configure' on a machine "
    << "that has the CUDA compiler 'nvcc' available.";
  return 1;
#endif
} catch (const std::exception &e) {
  std::cerr << e.what();
  std::cerr
    << "### WE DID NOT GET A CUDA GPU!!! ###" << std::endl
    << "### If it's your 1st experiment with CUDA, try reinstalling "
    << "'CUDA toolkit' from NVidia web (it contains the drivers)."
    << std::endl
    << "### In other cases run 'nvidia-smi' in terminal "
    << "(gets installed with display drivers) :"
    << std::endl
    << "### - Check that you see your GPU."
    << std::endl
    << "### - Bad GPUs are reporting error or disappear from the list "
    << "until reboot."
    << std::endl
    << "### - Check 'Memory-Usage' and 'GPU fan', "
    << "which will tell you if the GPU was taken by other process."
    << std::endl
    << "### - Check there is same version of 'NVIDIA-SMI' and "
    << "'Driver', and that it is not too old for your GPU."
    << std::endl;
  return -1;
}

