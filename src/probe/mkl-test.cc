// probe/mkl-test.cc

// Copyright (c) 2015, Johns Hopkins University (Yenda Trmal<jtrmal@gmail.com>)

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

#include <iostream>
#include <mkl.h>

int main(void)
{
  int len=512;
  char buf[len+1];
  mkl_get_version_string(buf, len);
  std::cerr << buf << std::endl;


  float a[] = { 0.11, 0.12,
                0.21, 0.22 };

  float b[] = { 1011, 1012,
                1031, 1032 };

  float c[] = { 0.00, 0.00,
                0.00, 0.00 };

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 2, 2, 2, 1.0, a, 2, b, 2, 0.0, c,2);

  return 0;
}

