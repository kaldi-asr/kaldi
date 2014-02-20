// bin/vector-sum.cc

// Copyright 2014 (Author: Vimal Manohar)

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

#include <vector>
#include <string>

using std::vector;
using std::string;

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-vector.h"
#include "transform/transform-common.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Add vectors (e.g. weights, transition-accs; speaker vectors)\n"
        "If you need to scale the inputs, use vector-scale on the inputs\n"
        "\n"
        "Usage: vector-sum [options] vector-in-rspecifier1 [vector-in-rspecifier2 vector-in-rspecifier3 ...] (vector-out-wspecifier\n"
        " e.g.: vector-sum ark:1.weights ark:2.weights ark:combine.weights\n";
    
    std::string weight_str;

    ParseOptions po(usage);

    po.Register("weights", &weight_str, "Colon-separated list of weights "
                "for each vector.");

    po.Read(argc, argv);

    int32 num_args = po.NumArgs();
    if (num_args < 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string vector_in_fn1 = po.GetArg(1),
        vector_out_fn = po.GetArg(num_args);

    // Output vector
    BaseFloatVectorWriter vector_writer(vector_out_fn);

    // Input vectors
    SequentialBaseFloatVectorReader vector_reader1(vector_in_fn1);
    std::vector<RandomAccessBaseFloatVectorReader*> vector_readers(num_args-2, 
        static_cast<RandomAccessBaseFloatVectorReader*>(NULL));
    std::vector<std::string> vector_in_fns(num_args-2);
    for (int32 i = 2; i < num_args; ++i) {
      vector_readers[i-2] = new RandomAccessBaseFloatVectorReader(po.GetArg(i));
      vector_in_fns[i-2] = po.GetArg(i);
    }

    std::vector<BaseFloat> weights(num_args-1, 1.0/(num_args-1));
    if (!weight_str.empty()) {
      SplitStringToFloats(weight_str, ":", true, &weights);
    }
 
    int32 n_utts = 0, n_total_vectors = 0, 
          n_success = 0, n_missing = 0, n_other_errors = 0;

    for (; !vector_reader1.Done(); vector_reader1.Next()) {
      std::string key = vector_reader1.Key();
      Vector<BaseFloat> vector1 = vector_reader1.Value();
      vector_reader1.FreeCurrent();
      n_utts++;
      n_total_vectors++;

      Vector<BaseFloat> vector_out(vector1);
      vector_out.Scale(weights[0]);

      for (int32 i = 0; i < num_args-2; ++i) {
        if (vector_readers[i]->HasKey(key)) {
          Vector<BaseFloat> vector2 = vector_readers[i]->Value(key);
          n_total_vectors++;
          if (vector2.Dim() == vector_out.Dim()) {
            vector_out.AddVec(weights[i+1], vector2);
          } else {
            KALDI_WARN << "Dimension mismatch for utterance " << key 
                       << " : " << vector2.Dim() << " for "
                       << "system " << (i + 2) << ", rspecifier: "
                       << vector_in_fns[i] << " vs " << vector_out.Dim() 
                       << " primary vector, rspecifier:" << vector_in_fn1;
            n_other_errors++;
          }
        } else {
          KALDI_WARN << "No vector found for utterance " << key << " for "
                     << "system " << (i + 2) << ", rspecifier: "
                     << vector_in_fns[i];
          n_missing++;
        }
      }

      vector_writer.Write(key, vector_out);
      n_success++;
    }

    KALDI_LOG << "Processed " << n_utts << " utterances: with a total of "
              << n_total_vectors << " vectors across " << (num_args-1)
              << " different systems";
    KALDI_LOG << "Produced output for " << n_success << " utterances; "
              << n_missing << " total missing vectors";
  
    DeletePointers(&vector_readers);

    return(n_success != 0 && n_missing < (n_success - n_missing) ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


