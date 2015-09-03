// bin/tree-info.cc

// Copyright 2013  Johns Hopkins University (author: Daniel Povey)

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
#include "hmm/hmm-topology.h"
#include "tree/context-dep.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Print information about decision tree (mainly the number of pdfs), to stdout\n"
        "Usage:  tree-info <tree-in>\n";
        
    ParseOptions po(usage);
    
    po.Read(argc, argv);

    if (po.NumArgs() != 1) {
      po.PrintUsage();
      exit(1);
    }

    std::string tree_in_filename = po.GetArg(1);

    ContextDependency ctx_dep;
    ReadKaldiObject(tree_in_filename, &ctx_dep);

    std::cout << "num-pdfs " << ctx_dep.NumPdfs() << std::endl;
    std::cout << "context-width " << ctx_dep.ContextWidth() << std::endl;
    std::cout << "central-position " << ctx_dep.CentralPosition() << std::endl;
    
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
