// featbin/copy-feats.cc

// Copyright 2012 Korbinian Riedhammer

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


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    
    const char *usage =
      "Merge feature files (assuming they have the same lengths);  think of the\n"
      "unix command paste a b\n"
      "Usage: merge-feats in-rspecifier1 in-rspecifier2 [in-rspecifier3 ...] out-wspecifier\n";
    
    ParseOptions po(usage);
    
    po.Read(argc, argv);
    
    if (po.NumArgs() < 3) {
      po.PrintUsage();
      exit(1);
    }
    
    // last argument is output
    std::string wspecifier = po.GetArg(po.NumArgs());
    BaseFloatMatrixWriter kaldi_writer(wspecifier);
    
    // assemble vector of input readers
    std::vector<SequentialBaseFloatMatrixReader *> input;
    for (int i = 1; i < po.NumArgs(); ++i) {
      std::string rspecifier = po.GetArg(i);
      input.push_back(new SequentialBaseFloatMatrixReader(rspecifier));
    }
    
    bool done = false;
    while (!done) {
      std::string key = "";
      
      Matrix<BaseFloat> feats;
      for (int i = 0; i < input.size(); ++i) {
        SequentialBaseFloatMatrixReader *rd = input[i];
        if (!rd->Done()) {
          if (key.length() == 0) {
            key = rd->Key();
            feats = rd->Value();
            rd->Next();
            continue;
          } else {
            if (key.compare(rd->Key()) != 0) {
              KALDI_ERR << "Error in input " << (i+1) << ";  expected key "
							<< key << " but got " << rd->Key();
            } else if (feats.NumRows() != rd->Value().NumRows()) {
              KALDI_ERR << "Error in input " << (i+1) << ";  wrong number of rows in " << rd->Key();
            }
          }
          int displ = feats.NumCols();
          feats.Resize(feats.NumRows(), feats.NumCols() + rd->Value().NumCols(), kCopyData);
          for (int j = 0; j < feats.NumRows(); j++) {
            for (int k = 0; k < rd->Value().NumCols(); ++k) {
              feats(j, displ + k) = rd->Value()(j, k);
            }
          }
          
          rd->Next();
        } else {
          done = true;
        }
      }	
      if (done) 
        break;
      kaldi_writer.Write(key, feats);
    }
    
    
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


