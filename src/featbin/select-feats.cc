// featbin/select-feats.cc

// Copyright 2012 Korbinian Riedhammer

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

#include <sstream>
#include <algorithm>
#include <iterator>
#include <utility>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace std;
    
    const char *usage =
        "Select certain dimensions of the feature file;  think of it as the unix\n"
        "command cut -f ...\n"
        "Usage: select-feats selection in-rspecifier out-wspecifier\n"
        "  e.g. select-feats 0,24-22,3-12 scp:feats.scp ark,scp:feat-red.ark,feat-red.scp\n"
        "See also copy-feats, extract-rows, subset-feats, subsample-feats\n";
    
    ParseOptions po(usage);
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }    

    string sspecifier = po.GetArg(1);
    string rspecifier = po.GetArg(2);
    string wspecifier = po.GetArg(3);
    
    // set up input (we'll need that to validate the selected indices)
    SequentialBaseFloatMatrixReader kaldi_reader(rspecifier);
    
    if (kaldi_reader.Done()) {
      KALDI_WARN << "Empty archive provided.";
      return 0;
    }
    
    int32 dimIn = kaldi_reader.Value().NumCols();
    int32 dimOut = 0;
    
    // figure out the selected dimensions
    istringstream iss(sspecifier);
    string token;
    vector<pair<int32, int32> > ranges;
    vector<int32> offsets;
    while (getline(iss, token, ',')) {
      size_t p = token.find('-');
      if (p != string::npos) {
        int s, e;
        istringstream(token.substr(0, token.length() - p - 1)) >> s;
        istringstream(token.substr(p+1)) >> e;
        
        if (s < 0 || s > (dimIn-1)) {
          KALDI_ERR << "Invalid range start: " << s;
          return 1;
        } else if (e < 0 || e > (dimIn-1)) {
          KALDI_ERR << "Invalid range end: " << e;
          return 1;
        }
        
        // reverse range? make individual selections
        if (s > e) {
          for (int32 i = s; i >= e; --i) {
            ranges.push_back(pair<int32, int32>(i, i));
            offsets.push_back(dimOut);
            dimOut += 1;
          }
        } else {
          ranges.push_back(pair<int32, int32>(s, e));
          offsets.push_back(dimOut);
          dimOut += (e - s + 1);
        }
      } else {
        int i;
        istringstream(token) >> i;
        
        if (i < 0 || i > (dimIn-1)) {
          KALDI_ERR << "Invalid selection index: " << i;
          return 1;
        }
        
        ranges.push_back(pair<int32, int32>(i, i));
        offsets.push_back(dimOut);
        dimOut += 1;
      }
    }
    
    if (ranges.size() < 1) {
      KALDI_ERR << "No ranges or indices in selection string!";
      return 1;
    }
    
    // set up output
    BaseFloatMatrixWriter kaldi_writer(wspecifier);

    // process all keys
    for (; !kaldi_reader.Done(); kaldi_reader.Next()) {
      Matrix<BaseFloat> feats(kaldi_reader.Value().NumRows(), dimOut);
      
      // extract the desired ranges
      for (int32 i = 0; i < ranges.size(); ++i) {
        int32 f = ranges[i].first;
        int32 ncol = ranges[i].second - f + 1;
        
        feats.Range(0, feats.NumRows(), offsets[i], ncol)
          .CopyFromMat(kaldi_reader.Value().Range(0, feats.NumRows(), f, ncol));
      }
      
      kaldi_writer.Write(kaldi_reader.Key(), feats);
    }
    
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
