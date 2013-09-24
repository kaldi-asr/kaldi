// fgmmbin/fgmm-global-merge.cc

// Copyright 2009-2011  Microsoft Corporation

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

#include "util/common-utils.h"
#include "gmm/full-gmm.h"
#include "gmm/mle-full-gmm.h"

namespace kaldi {

/// merges GMMs by appending Gaussians in "src" to "dst".
/// Renormalizes weights by allocating weight proportional to #Gauss.
void MergeFullGmm(const FullGmm &src, FullGmm *dst) {
  FullGmm dst_copy;
  dst_copy.CopyFromFullGmm(*dst);
  KALDI_ASSERT(src.NumGauss() != 0 && dst_copy.NumGauss()  != 0
               && src.Dim() == dst_copy.Dim());
  int32 src_num_mix = src.NumGauss(), dst_num_mix = dst_copy.NumGauss(),
      num_mix = src_num_mix + dst_num_mix, dim = src.Dim();
  dst->Resize(num_mix, dim);

  std::vector<SpMatrix<BaseFloat> > invcovars(num_mix);
  for(int32 i = 0; i < dst_num_mix; i++) {
    invcovars[i].Resize(dim);
    invcovars[i].CopyFromSp(dst_copy.inv_covars()[i]);
  }
  for(int32 i = 0; i < src_num_mix; i++) {
    invcovars[i+dst_num_mix].Resize(dim);
    invcovars[i+dst_num_mix].CopyFromSp(src.inv_covars()[i]);
  }
  Matrix<BaseFloat> means_invcovars(num_mix, dim);
  means_invcovars.Range(0, dst_num_mix, 0, dim).CopyFromMat(dst_copy.means_invcovars());
  means_invcovars.Range(dst_num_mix, src_num_mix, 0, dim).CopyFromMat(src.means_invcovars());
  dst->SetInvCovarsAndMeansInvCovars(invcovars, means_invcovars);

  Vector<BaseFloat> weights(num_mix); // initialized to zero.
  // weight proportional to #Gaussians, so that if we combine a number of
  // models with same #Gaussians, they all get the same weight.
  BaseFloat src_weight = src_num_mix / static_cast<BaseFloat>(num_mix),
      dst_weight = dst_num_mix / static_cast<BaseFloat>(num_mix);
  weights.Range(0, dst_num_mix).AddVec(dst_weight, dst_copy.weights());
  weights.Range(dst_num_mix, src_num_mix).AddVec(src_weight, src.weights());
  dst->SetWeights(weights);
  dst->ComputeGconsts();
}

}


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Combine a number of GMMs into a larger GMM, with #Gauss = \n"
        "  sum(individual #Gauss)).  Output full GMM, and a text file with\n"
        "  sizes of each individual GMM.\n"
        "Usage: fgmm-global-merge [options] fgmm-out sizes-file-out fgmm-in1 fgmm-in2 ...\n";

    bool binary = true;
    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Read(argc, argv);

    if (po.NumArgs() < 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string fgmm_out_filename = po.GetArg(1),
        sizes_out_filename = po.GetArg(2);

    FullGmm fgmm;
    Output sizes_ko(sizes_out_filename, false); // false == not binary.
    
    for (int i = 3, max = po.NumArgs(); i <= max; i++) {
      std::string stats_in_filename = po.GetArg(i);
      bool binary_read;
      Input ki(stats_in_filename, &binary_read);
      if (i==3) {
        fgmm.Read(ki.Stream(), binary_read);
        sizes_ko.Stream() << fgmm.NumGauss() << ' ';
      } else {
        FullGmm fgmm2;
        fgmm2.Read(ki.Stream(), binary_read);
        sizes_ko.Stream() << fgmm2.NumGauss() << ' ';
        MergeFullGmm(fgmm2, &fgmm);
      }
    }
    sizes_ko.Stream() << "\n";
    
    // Write out the model
    WriteKaldiObject(fgmm, fgmm_out_filename, binary);
    KALDI_LOG << "Written merged GMM to " << fgmm_out_filename;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

