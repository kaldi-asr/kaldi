// bin/init-tree-special.cc

// Copyright 2009-2011  Microsoft Corporation

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
#include "tree/build-tree.h"
#include "tree/build-tree-utils.h"
#include "tree/context-dep.h"
#include "tree/clusterable-classes.h"
#include "util/text-utils.h"

namespace kaldi {

void ReadTriphones(const std::string &rxfilename,
                   int N,
                   std::vector<std::vector<int32> > *triphones) {
  triphones->clear();
  Input ki(rxfilename, false); // false == text mode.
  std::string line;
  while (std::getline(ki.Stream(), line)) {
    std::vector<int32> context; // a single triphone context.   
    if (!SplitStringToIntegers(line, " \t\n\r", true, &context)
        || context.size() != N)
      KALDI_EXIT << "Bad line " << line << " in triphones file.";
    triphones->push_back(context);
  }
}


}

int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Initialize untied decision tree from a list of separate triphones\n"
        "Usage:  init-tree-special [options] <triphones-file> <topo-file> <list-out> <tree-out>\n"
        " where format of triphones-file is lines like: 1 15 30\n"
        " representing context windows of phones; zero means end/begin of file, and\n"
        " is also used as the context for context-independent phones (e.g. silence).\n"
        " The file \"list-out\" will contain lines like:\n"
        " 0 1 15 10\n"
        " where the first entry is the pdf-class (normally same as HMM state)\n"
        " and the rest are the context window of 3 phones.\n"
        "e.g.: \n"
        " init-tree-special triphones topo statelist tree\n";

    bool binary = false;
    std::string ci_phones_str;
    int N = 3; // context window size.
    int P = 1; // central position in context window.

    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("context-width", &N, "Context window size");
    po.Register("central-position", &P, "Central position in context window");
    po.Register("ci-phones", &ci_phones_str, "Colon-separated list of integer "
                "indices of context-independent phones.");

    
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }
    std::string triphone_rxfilename = po.GetArg(1),
        topo_rxfilename = po.GetArg(2),
        statelist_wxfilename = po.GetArg(3),
        tree_wxfilename = po.GetArg(4);


    // List of triphone contexts.
    // For context-independent phones, the context phones are zero.
    // [zero may appear elsewhere as "unknown context"].
    std::vector<std::vector<int32> > triphones;
    ReadTriphones(triphone_rxfilename, N, &triphones);

    HmmTopology topo;
    {
      bool binary_in;
      Input ki(topo_rxfilename, &binary_in);
      topo.Read(ki.Stream(), binary_in);
    }

    std::vector<int32> phone2num_pdf_classes;
    topo.GetPhoneToNumPdfClasses(&phone2num_pdf_classes);

    std::vector<int32> ci_phones;
    if (!SplitStringToIntegers(ci_phones_str, " \t\n\r", true, &ci_phones))
      KALDI_EXIT << "Invalid option --ci-phones=" << ci_phones_str;
    
    
    //////// Build the tree. ////////////

    // After the next call, "states" would have dimension
    // [triphones.size()][N+1];
    // the last element of each element of "states" will be
    // the number of pdf-classes.
    
    std::vector<std::vector<int32> > states; 
    EventMap *to_pdf = CreateUntiedTree(N, P,
                                        triphones,
                                        phone2num_pdf_classes,
                                        ci_phones,
                                        &states);

    ContextDependency ctx_dep(N, P, to_pdf);  // takes ownership
    // of pointer "to_pdf", so set it NULL.
    to_pdf = NULL;

    {
      Output os(tree_wxfilename, binary);
      ctx_dep.Write(os.Stream(), binary);
    }
    std::cerr << "Wrote tree\n";
    {
      Output os(statelist_wxfilename, false); // write statelist in text format
      // regardless of "binary" option-- not really a Kaldi-format file.
      for (size_t i = 0; i < states.size(); i++) {
        KALDI_ASSERT(states[i].size() == static_cast<size_t>(N+1));
        for (size_t j = 0; j < states[i].size(); j++)
          os.Stream() << states[i][j] << ' ';
        os.Stream() << std::endl;
      }
    }
    std::cerr << "Wrote statelist file\n";    

  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
}
