// bin/build-tree-virtual.cc

// Copyright 2014 Hainan XU

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "util/stl-utils.h"
#include "hmm/hmm-topology.h"
#include "tree/context-dep.h"
#include "tree/context-dep-multi.h"
#include "tree/build-tree.h"
#include "tree/build-tree-virtual.h"
#include "tree/build-tree-utils.h"
#include "tree/clusterable-classes.h"
#include "util/text-utils.h"

using std::string;
using std::vector;
using std::pair;

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Convert virtual tree mapping to a sparse matrix\n"
        "Usage:  convert-treemap-to-matrix [options]"
        " <input-tree-map> <output-sparse-matrix>\n";

    bool binary = true;

    ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string mapping_filename = po.GetArg(1),
                matrix_filename = po.GetArg(2);

    unordered_map<int32, vector<int32> > mapping;
    SparseMatrix<BaseFloat> out;

    {
      bool binary;
      Input ki(mapping_filename, &binary);
      ReadMultiTreeMapping(mapping, ki.Stream(), binary);
      ki.Close();
    }

    MappingToSparseMatrix(mapping, &out);
    WriteKaldiObject(out, matrix_filename, binary);

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
