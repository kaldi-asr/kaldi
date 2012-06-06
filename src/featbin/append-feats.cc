// featbin/append-feats.cc

// Copyright 2009-2011
// Author: Petr Motlicek (motlicek@idiap.ch)

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
        "Append 2 feature-streams [and possibly change format]\n"
        "Usage: append-feats [options] in-rspecifier1 in-rspecifier2 out-wspecifier\n";

    ParseOptions po(usage);
    
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string rspecifier1 = po.GetArg(1);
    std::string rspecifier2 = po.GetArg(2);
    std::string wspecifier = po.GetArg(3);

    BaseFloatMatrixWriter kaldi_writer(wspecifier);



    SequentialBaseFloatMatrixReader kaldi_reader1(rspecifier1);
    SequentialBaseFloatMatrixReader kaldi_reader2(rspecifier2);
    Matrix<BaseFloat> output_feats;

    for (; !kaldi_reader1.Done(); kaldi_reader1.Next()){
     std::cout << "Key: " << kaldi_reader1.Key() << "\n";

     const Matrix<BaseFloat> &feats1  = kaldi_reader1.Value();
     const Matrix<BaseFloat> &feats2  = kaldi_reader2.Value();
     //std::cout << "OUT: " << feats1.NumRows() << " " << feats2.NumRows() << "\n";
     int32 feat1_dim = feats1.NumCols();
     int32 feat2_dim = feats2.NumCols();
     assert(feats1.NumRows() == feats2.NumRows());
     assert(kaldi_reader1.Key() == kaldi_reader2.Key());


 

     output_feats.Resize(feats1.NumRows(),
                        feats1.NumCols() + feats2.NumCols());

     for (int32 r = 0; r < static_cast<int32>(feats1.NumRows()); r++) {
           SubVector<BaseFloat> row(output_feats, r);
           row.SetZero();
           SubVector<BaseFloat> output1(row, 0, feat1_dim);
           output1.AddVec(1.0, feats1.Row(r));
           SubVector<BaseFloat> output2(row, 1*feat1_dim, feat2_dim);
           output2.AddVec(1.0, feats2.Row(r));
     }

     kaldi_writer.Write(kaldi_reader1.Key(), output_feats);
     kaldi_reader2.Next(); // shift the second pointer
   } 

    return 0;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}


