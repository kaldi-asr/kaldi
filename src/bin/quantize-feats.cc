// bin/quantize-feats.cc

// Copyright 2015   Vimal Manohar

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"
#include "hmm/posterior.h"

int main(int argc, char *argv[]) {
  try { using namespace kaldi;
    typedef TableWriter<BasicVectorVectorHolder<bool> >  BooleanVectorVectorWriter;

    const char *usage =
        "Quantize feature values into bins using <bin-boundaries>.\n"
        "<bin-boundaries> is a sorted colon-separated list a:b:c:...:z.\n"
        "The corresponding bins are (-inf,a], (a,b], (b,c], ... (z,inf).\n"
        "Usage: quantize-feats [options] <feature-rspecifier> <bin-boundaries> <bins-wspecifier>\n"
        " e.g.: quantize-feats ark:- -10:0:10 ark,scp:foo.ark,foo.scp\n"
        "See also: copy-feats, copy-matrix\n";

    bool write_boolean_vector = false;

    ParseOptions po(usage);
    
    po.Register("write-boolean-vector", &write_boolean_vector, "Write boolean "
                "vector instead of posteriors");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    int32 num_done = 0;
    
    std::string feats_rspecifier = po.GetArg(1);
    std::string bin_boundaries_str = po.GetArg(2);
    std::string bins_wspecifier = po.GetArg(3);
    
    std::vector<BaseFloat> bin_boundaries;
    if (!SplitStringToFloats(bin_boundaries_str, ":", false,
        &bin_boundaries) || bin_boundaries.empty() 
        || !IsSorted(bin_boundaries) ) {
      KALDI_ERR << "Invalid bin-boundaries string " 
                << bin_boundaries_str;
    }
    
    int32 num_bins = bin_boundaries.size() + 1;

    SequentialBaseFloatMatrixReader feats_reader(feats_rspecifier);
    PosteriorWriter bin_writer(bins_wspecifier);
    BooleanVectorVectorWriter bits_writer(bins_wspecifier);
    
    for (; !feats_reader.Done(); feats_reader.Next(), num_done++) {
      const Matrix<BaseFloat> &feats = feats_reader.Value();
      const std::string &key = feats_reader.Key();
      
      Posterior bins;
      std::vector<std::vector<bool> > bits;

      if (!write_boolean_vector)
        bins.resize(feats.NumRows());
      else 
        bits.resize(feats.NumRows());

      for (size_t t = 0; t < feats.NumRows(); t++) {
        if (!write_boolean_vector)
          bins[t].resize(feats.NumCols());
        else
          bits[t].resize(num_bins * feats.NumCols(), false);
        for (size_t j = 0; j < feats.NumCols(); j++) {
          auto bin = std::lower_bound(bin_boundaries.begin(), 
                                      bin_boundaries.end(), feats(t,j));
          size_t k;
          if (bin != bin_boundaries.end())
            k = static_cast<size_t>(bin - bin_boundaries.begin());
          else 
            k = static_cast<size_t>(bin_boundaries.size());

          if (!write_boolean_vector)
            bins[t][j] = std::make_pair(j * num_bins + k, 1.0);
          else 
            bits[t][j * num_bins + k] = true;
        }
      }

      if (!write_boolean_vector)
        bin_writer.Write(key, bins);
      else
        bits_writer.Write(key, bits);
    }
    
    KALDI_LOG << "Quantized " << num_done << " feature matrices.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

