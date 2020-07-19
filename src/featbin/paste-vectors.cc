// featbin/paste-vectors.cc

// Copyright 2020 Ivan Medennikov (STC-innovation Ltd)

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
#include "matrix/kaldi-matrix.h"
#include "util/common-utils.h"

namespace kaldi {

void AppendVectors(const std::vector<Vector<BaseFloat> > &in,
                   Vector<BaseFloat> *out) {
  // Check the lengths
  int32 tot_dim = in[0].Dim();
  for (int32 i = 1; i < in.size(); ++i) {
    int32 dim = in[i].Dim();
    tot_dim += dim;
  }
  out->Resize(tot_dim);
  int32 dim_offset = 0;
  for (int32 i = 0; i < in.size(); ++i) {
    int32 this_dim = in[i].Dim();
    out->Range(dim_offset, this_dim).CopyFromVec(
        in[i].Range(0, this_dim));
    dim_offset += this_dim;
  }
}


}  // namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace std;

    const char *usage =
        "Concatenate vector files.\n"
        "Append to vectors the first archive the vectors under the same key\n"
        "from the second and following archives, and output the result.\n"
        "NOTE: Keys absent from the first archive will not be present in output,\n"
        "even if they occur in the additional input archives.\n"
        "Usage: paste-vectors <in-rspecifier1> <in-rspecifier2> [<in-rspecifier3> ...] <out-wspecifier>\n"
        "See also: paste-feats, copy-feats, copy-matrix, append-vector-to-feats, concat-feats\n";

    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() < 3) {
      po.PrintUsage();
      exit(1);
    }

    if (ClassifyRspecifier(po.GetArg(1), NULL, NULL) == kNoRspecifier) {
      KALDI_ERR << "This program can operate only on tables, not on rxfiles";
    }

    // Last argument is output
    string wspecifier = po.GetArg(po.NumArgs());
    BaseFloatVectorWriter vec_writer(wspecifier);

    // First input is sequential
    string first_rspecifier = po.GetArg(1);
    SequentialBaseFloatVectorReader first_input(first_rspecifier);

    // Assemble vector of other input readers (with random-access)
    vector<RandomAccessBaseFloatVectorReader *> rest_inputs;
    for (int32 i = 2; i < po.NumArgs(); ++i) {
      string rspecifier = po.GetArg(i);
      RandomAccessBaseFloatVectorReader *rd = new RandomAccessBaseFloatVectorReader(rspecifier);
      rest_inputs.push_back(rd);
    }

    int32 num_done = 0, num_err = 0;

    // Main loop
    for (; !first_input.Done(); first_input.Next()) {
      string utt = first_input.Key();
      KALDI_VLOG(2) << "Merging vectors for utterance " << utt;

      // Collect vectors from streams to vector 'vectors'
      vector<Vector<BaseFloat> > vectors(po.NumArgs() - 1);
      vectors[0] = first_input.Value();
      size_t i;
      for (i = 0; i < rest_inputs.size(); ++i) {
        if (rest_inputs[i]->HasKey(utt)) {
          vectors[i + 1] = rest_inputs[i]->Value(utt);
        } else {
          KALDI_WARN << "Missing utt " << utt << " from input "
                     << po.GetArg(i + 2);
          ++num_err;
          break;
        }
      }
      if (i != rest_inputs.size())
        continue;
      Vector<BaseFloat> output;
      AppendVectors(vectors, &output);
      vec_writer.Write(utt, output);
      ++num_done;
    }

    for (int32 i = 0; i < rest_inputs.size(); ++i)
      delete rest_inputs[i];
    rest_inputs.clear();

    KALDI_LOG << "Done " << num_done << " utts, errors on "
              << num_err;

    return (num_done == 0 ? -1 : 0);
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
