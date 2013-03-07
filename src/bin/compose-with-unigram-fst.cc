// bin/compose-with-unigram-fst.cc

// Copyright 2009-2012  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)

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
#include "fstext/fstext-lib.h"
//#include "fst/fstlib.h"
#include "fst/compose.h"
#include "fst/arcsort.h"
#include "fst/topsort.h"



int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;
    using fst::ComposeFst;
    using fst::ArcSort;
    using fst::StdILabelCompare;
    using fst::StdOLabelCompare;
    using fst::TopSort;

    const char *usage =
        "Composes training graphs with unigram grammar FST (ie. injects the LM scores)\n"
        "\n"
        "Usage:   compose-with-unigram-fst [options] unigram-fst graphs-rspecifier graphs-wspecifier\n"
        "e.g.: \n"
        " compose-with-unigram-fst G.fst ark:graph_in.ark ark:graph_out.ark\n";

    ParseOptions po(usage);
    /* No configuration options */
    po.Read(argc, argv);

    if(po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string unigramfst_rxfilename = po.GetArg(1);
    std::string graph_rspecifier = po.GetArg(2);
    std::string graph_wspecifier = po.GetArg(3);

    //Read the unigram fst
    VectorFst<StdArc> *unigram_fst = NULL;
    {
      std::ifstream is(unigramfst_rxfilename.c_str(), 
                       std::ios_base::in|std::ios_base::binary);
      if (!is.good()) KALDI_ERR << "Could not open unigram FST " << unigramfst_rxfilename;
      KALDI_LOG << "Loading unigram FST " << unigramfst_rxfilename;
      unigram_fst = VectorFst<StdArc>::Read(is, fst::FstReadOptions(unigramfst_rxfilename));
      if(NULL == unigram_fst) {
        KALDI_ERR << "Cannot read unigram FST " << unigramfst_rxfilename;
      }
      //sort it along the input symbols
      ArcSort(unigram_fst, StdILabelCompare());
    }

    //Iterate over the graphs
    SequentialTableReader<fst::VectorFstHolder> graph_reader(graph_rspecifier);
    TableWriter<fst::VectorFstHolder> graph_writer(graph_wspecifier);
    int32 num_success = 0; 
    for(; !graph_reader.Done(); graph_reader.Next()) {
      //get the graph
      std::string key = graph_reader.Key();
      VectorFst<StdArc> graph_in = graph_reader.Value();
      //sort the input graph along the output symbols
      ArcSort(&graph_in, StdOLabelCompare());
      //compose with unigram fst
      VectorFst<StdArc> graph_out;

      //KALDI_LOG << "gr_in: states" << graph_in.NumStates(); //<< ", arcs" << graph_in.NumArcs(0);
      //KALDI_LOG << "unigr_in: states" << unigram_fst->NumStates(); // << ", arcs" << unigram_fst.NumArcs(0);
      Compose(graph_in,*unigram_fst,&graph_out);
      //KALDI_LOG << "gr_out: states" << graph_out.NumStates(); // << ", arcs" << graph_out.NumArcs(0);
      //write it
      graph_writer.Write(key,graph_out);
      //count the files
      num_success += 1;
    }

    delete unigram_fst;

    KALDI_LOG << "Successfuly added unigram scores to " 
              << num_success << " training graphs.";
    
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}





