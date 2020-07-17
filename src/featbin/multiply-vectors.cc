// featbin/multiply-vectors.cc

// Copyright 2020 Ivan Medennikov (STC-innovations Ltd)


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

// returns true if successfully multiplied.
bool MultiplyVectors(const std::vector<Vector<BaseFloat> > &in,
                     std::string utt,
                     int32 tolerance,
                     Vector<BaseFloat> *out) {
  // Check the lengths
  int32 min_len = in[0].Dim(),
      max_len = in[0].Dim();
  for (int32 i = 1; i < in.size(); ++i) {
    int32 len = in[i].Dim();
    if(len < min_len) min_len = len;
    if(len > max_len) max_len = len;
  }
  if (max_len - min_len > tolerance || min_len == 0) {
    KALDI_WARN << "Length mismatch " << max_len << " vs. " << min_len
               << (utt.empty() ? "" : " for utt ") << utt
               << " exceeds tolerance " << tolerance;
    out->Resize(0);
    return false;
  }
  if (max_len - min_len > 0) {
    KALDI_VLOG(2) << "Length mismatch " << max_len << " vs. " << min_len
                  << (utt.empty() ? "" : " for utt ") << utt
                  << " within tolerance " << tolerance;
  }
  out->Resize(min_len);
  out->Set(1.0);
  for (int32 i = 0; i < in.size(); ++i) {
    out->MulElements(in[i].Range(0, min_len));
  }
  return true;
}


}  // namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace std;

    const char *usage =
        "Multiply vectors frame-by-frame (assuming they have about the same durations, see --length-tolerance);\n"
        "Usage: multiply-vectors <in-rspecifier1> <in-rspecifier2> [<in-rspecifier3> ...] <out-wspecifier>\n"
        " or:  multiply-vectors <in-rxfilename1> <in-rxfilename2> [<in-rxfilename3> ...] <out-wxfilename>\n"
        " e.g. multiply-vectors ark:vec1.ark ark:vec2.ark ark:out.ark\n"
        " or:  multiply-vectors foo.mat bar.mat baz.mat\n"
        "See also: paste-feats, copy-vector, append-vector-to-feats\n";

    ParseOptions po(usage);

    int32 length_tolerance = 0;
    bool binary = true;
    po.Register("length-tolerance", &length_tolerance,
                "If length is different, trim as shortest up to a frame "
                " difference of length-tolerance, otherwise exclude segment.");
    po.Register("binary", &binary, "If true, output files in binary "
                "(only relevant for single-file operation, i.e. no tables)");

    po.Read(argc, argv);

    if (po.NumArgs() < 3) {
      po.PrintUsage();
      exit(1);
    }

    if (ClassifyRspecifier(po.GetArg(1), NULL, NULL)
        != kNoRspecifier) {
      // We're operating on tables, e.g. archives.

      // Last argument is output
      string wspecifier = po.GetArg(po.NumArgs());
      BaseFloatVectorWriter vector_writer(wspecifier);

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
        KALDI_VLOG(2) << "Multiplying vectors for utterance " << utt;

        // Collect features from streams to vector 'vectors'
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
        if (!MultiplyVectors(vectors, utt, length_tolerance, &output)) {
          ++num_err;
          continue; // it will have printed a warning.
        }
        vector_writer.Write(utt, output);
        ++num_done;
      }

      for (int32 i = 0; i < rest_inputs.size(); ++i)
        delete rest_inputs[i];
      rest_inputs.clear();

      KALDI_LOG << "Done " << num_done << " utts, errors on "
                << num_err;

      return (num_done == 0 ? -1 : 0);
    } else {
      // We're operating on rxfilenames|wxfilenames, most likely files.
      std::vector<Vector<BaseFloat> > vectors(po.NumArgs() - 1);
      for (int32 i = 1; i < po.NumArgs(); ++i)
        ReadKaldiObject(po.GetArg(i), &(vectors[i - 1]));
      Vector<BaseFloat> output;
      if (!MultiplyVectors(vectors, "", length_tolerance, &output))
        return 1; // it will have printed a warning.
      std::string output_wxfilename = po.GetArg(po.NumArgs());
      WriteKaldiObject(output, output_wxfilename, binary);
      KALDI_LOG << "Wrote multiplied vector to " << output_wxfilename;
      return 0;
    }
  } catch (const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
