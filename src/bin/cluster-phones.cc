// bin/cluster-phones.cc

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "tree/context-dep.h"
#include "tree/build-tree.h"
#include "tree/build-tree-utils.h"
#include "tree/context-dep.h"
#include "tree/clusterable-classes.h"
#include "util/text-utils.h"



int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Cluster phones (or sets of phones) into sets for various purposes\n"
        "Usage:  cluster-phones [options] <tree-stats-in> <phone-sets-in> <clustered-phones-out>\n"
        "e.g.: \n"
        " cluster-phones 1.tacc phonesets.txt questions.txt\n";
    // Format of phonesets.txt is e.g.
    // 1
    // 2 3 4
    // 5 6
    // ...
    // Format of questions.txt output is similar, but with more lines (and the same phone
    // may appear on multiple lines).

    // bool binary = true;
    int32 P = 1, N = 3; // Note: N does not matter.
    std::string pdf_class_list_str = "1";  // 1 is just the central position of 3.
    std::string mode = "questions";
    int32 num_classes = -1;

    ParseOptions po(usage);
    // po.Register("binary", &binary, "Write output in binary mode");
    po.Register("central-position", &P, "Central position in context window [must match acc-tree-stats]");
    po.Register("context-width", &N, "Does not have any effect-- included for scripting convenience.");
    po.Register("pdf-class-list", &pdf_class_list_str, "Colon-separated list of HMM positions to consider [Default = 1: just central position for 3-state models].");
    po.Register("mode", &mode, "Mode of operation: \"questions\"->sets suitable for decision trees; \"k-means\"->k-means algorithm, output k classes (set num-classes options)\n");
    po.Register("num-classes", &num_classes, "For k-means mode, number of classes.");


    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }


    std::string stats_rxfilename = po.GetArg(1),
        phone_sets_rxfilename = po.GetArg(2),
        phone_sets_wxfilename = po.GetArg(3);


    BuildTreeStatsType stats;
    {  // Read tree stats.
      bool binary_in;
      GaussClusterable gc;  // dummy needed to provide type.
      Input ki(stats_rxfilename, &binary_in);
      ReadBuildTreeStats(ki.Stream(), binary_in, gc, &stats);
    }

    std::vector<int32> pdf_class_list;
    if (!SplitStringToIntegers(pdf_class_list_str, ":", false, &pdf_class_list)
       || pdf_class_list.empty()) {
      KALDI_ERR << "Invalid pdf-class-list string [expecting colon-separated list of integers]: " 
                 << pdf_class_list_str;
    }

    std::vector<std::vector< int32> > phone_sets;
    if (!ReadIntegerVectorVectorSimple(phone_sets_rxfilename, &phone_sets))
      KALDI_ERR << "Could not read phone sets from "
                 << PrintableRxfilename(phone_sets_rxfilename);

    if (phone_sets.size() == 0)
      KALDI_ERR << "No phone sets in phone sets file ";

    std::vector<std::vector<int32> > phone_sets_out;

    if (mode == "questions") {
      if (num_classes != -1)
        KALDI_ERR << "num-classes option is not (currently) compatible "
            "with \"questions\" mode.";
      AutomaticallyObtainQuestions(stats,
                                   phone_sets,
                                   pdf_class_list,
                                   P,
                                   &phone_sets_out);
    } else if (mode == "k-means") {
      if (num_classes <= 1 ||
		 static_cast<size_t>(num_classes) > phone_sets.size())
        KALDI_ERR << "num-classes invalid: num_classes is " << num_classes
                  << ", number of phone sets is " << phone_sets.size();
      KMeansClusterPhones(stats,
                          phone_sets,
                          pdf_class_list,
                          P,
                          num_classes,
                          &phone_sets_out);
    }

    if (!WriteIntegerVectorVectorSimple(phone_sets_wxfilename, phone_sets_out))
      KALDI_ERR << "Error writing questions to "
                 << PrintableWxfilename(phone_sets_wxfilename);
    else
      KALDI_LOG << "Wrote questions to "<<phone_sets_wxfilename;

    DeleteBuildTreeStats(&stats);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
