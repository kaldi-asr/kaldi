// featbin/copy-feats-to-sphinx.cc

// Copyright 2013   Petr Motlicek

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
#include "matrix/matrix-common.h"
#include "matrix/matrix-lib.h"

#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Save features as Sphinx files:\n" 
        "Each utterance will be stored as a unique Sphinx file in a specified directory.\n"
        "The Sphinx filename will correspond to the utterance-id (key) in the input table, with the specified extension.\n"
        "Usage: copy-feats-to-sphinx [options] in-rspecifier\n"
        "Example: copy-feats-to-sphinx --output-dir=/tmp/sphinx-features --output-ext=fea  scp:feats.scp\n";

    ParseOptions po(usage);
    std::string dir_out = "./";
    std::string ext_out = "mfc";

    po.Register("output-ext", &ext_out, "Output extension of sphinx files");
    po.Register("output-dir", &dir_out, "Output directory");

    po.Read(argc, argv);

    if (po.NumArgs() != 1) {
      po.PrintUsage();
      exit(1);
    }

    std::string rspecifier = po.GetArg(1);

    // check or create output dir:
    const char * c = dir_out.c_str();
   if ( access( c, 0 ) != 0 ){
    if (mkdir(c, S_IRWXU|S_IRGRP|S_IXGRP) != 0)
       KALDI_ERR << "Could not create output directory: " << dir_out;
    }
    
    // write to the sphinx files
    int32 num_frames, dim, num_done=0;
    SequentialBaseFloatMatrixReader feats_reader(rspecifier);
    for (; !feats_reader.Done(); feats_reader.Next()) {
      std::string utt = feats_reader.Key();
      const Matrix<BaseFloat> &feats = feats_reader.Value();
      num_frames = feats.NumRows(), dim = feats.NumCols();

      Matrix<BaseFloat> output(num_frames, dim, kUndefined);
      std::stringstream ss;
      ss << dir_out << "/" << utt << "." << ext_out; 
      output.Range(0, num_frames, 0, dim).CopyFromMat(feats.Range(0, num_frames, 0, dim));	
      std::ofstream os(ss.str().c_str(), std::ios::out|std::ios::binary);
      WriteSphinx(os, output);
      num_done++;    
    }
    KALDI_LOG << num_done << " Sphinx feature files generated in the direcory: " << dir_out;
    return (num_done != 0 ? 0 : 1);

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
