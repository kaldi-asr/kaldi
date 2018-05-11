// xvectorbin/nnet3-xvector-scoring.cc

// Copyright 2013  Daniel Povey
//           2016  David Snyder

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
#include "nnet3/nnet-utils.h"
#include "xvector/xvector.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet3;
  typedef kaldi::int32 int32;
  typedef kaldi::int64 int64;
  try {
    const char *usage =
        "Computes scores between pairs of xvectors.\n"
        "The 'trials-file' has lines of the form\n"
        "<key1> <key2>\n"
        "and the output will have the form\n"
        "<key1> <key2> [<score>]\n"
        "(if either key could not be found, the score field in the output\n"
        "will be absent, and this program will print a warning)\n"
        "\n"
        "Usage:  nnet3-xvector-scoring [options] <raw-nnet-in> "
        "<trials-in> <xvector1-rspecifier> <xvector2-rspecifier> "
        "<scores-out>\n"
        "e.g.: \n"
        "  nnet3-xvector-scoring nnet.final trials ark:spk_xvectors.scp "
        "ark:test_xvectors.scp trials.scored\n"
        "See also: ivector-plda-scoring and ivector-compute-dot-products\n";

    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string nnet_rxfilename = po.GetArg(1),
        trials_rxfilename = po.GetArg(2),
        xvector1_rspecifier = po.GetArg(3),
        xvector2_rspecifier = po.GetArg(4),
        scores_wxfilename = po.GetArg(5);


    int64 num_done = 0, num_err = 0;
    Nnet nnet;
    ReadKaldiObject(nnet_rxfilename, &nnet);
    // We need to ensure that the Nnet has outputs called 's' and 'b'
    // and that 'b' is a scalar and 's' can be interpreted as a symmetric
    // matrix.
    int32 s_index = nnet.GetNodeIndex("s"),
          b_index = nnet.GetNodeIndex("b");
    if (s_index == -1 || b_index == -1)
      KALDI_ERR << "The input Nnet cannot be used for xvector scoring"
                << "because it has no output called 's' or 'b'.";
    if (!nnet.IsOutputNode(s_index) || !nnet.IsOutputNode(b_index))
      KALDI_ERR << "The nodes 's' and 'b' must be output nodes.";

    int32 s_dim = nnet.OutputDim("s"),
          b_dim = nnet.OutputDim("b");
    if (b_dim != 1)
      KALDI_ERR << "The output 'b' is a scalar offset.  Input Nnet has an"
                << "output called 'b' but it has a dimension of " << b_dim;
    int32 d = (0.5) * (1 + sqrt(1 + 8 * s_dim)) - 1;
    if (((d + 1) * d) / 2 != s_dim)
      KALDI_ERR << "Output 's' cannot be interpretedas a symmetric matrix.";
    Vector<BaseFloat> s_vec(s_dim);
    Vector<BaseFloat> b_vec(1);
    GetConstantOutput(nnet, "s", &s_vec);
    GetConstantOutput(nnet, "b", &b_vec);
    SpMatrix<BaseFloat> S(d);
    SubVector<BaseFloat> s_vec_sub(s_vec, 0, s_dim);
    S.CopyFromVec(s_vec_sub);
    BaseFloat b = b_vec(0);

    RandomAccessBaseFloatVectorReader xvector1_reader(xvector1_rspecifier);
    RandomAccessBaseFloatVectorReader xvector2_reader(xvector2_rspecifier);

    Input ki(trials_rxfilename);

    bool binary = false;
    Output ko(scores_wxfilename, binary);
    double sum = 0.0, sumsq = 0.0;

    std::string line;
    while (std::getline(ki.Stream(), line)) {
      std::vector<std::string> fields;
      SplitStringToVector(line, " \t\n\r", true, &fields);
      if (fields.size() != 2) {
        KALDI_ERR << "Bad line " << (num_done + num_err) << " in input "
                  << "(expected two fields: key1 key2): " << line;
      }
      std::string key1 = fields[0], key2 = fields[1];
      if (!xvector1_reader.HasKey(key1)) {
        KALDI_WARN << "Key " << key1 << " not present in 1st table of xvectors.";
        num_err++;
        continue;
      }
      if (!xvector2_reader.HasKey(key2)) {
        KALDI_WARN << "Key " << key2 << " not present in 2nd table of xvectors.";
        num_err++;
        continue;
      }
      const Vector<BaseFloat> &xvector1 = xvector1_reader.Value(key1),
          &xvector2 = xvector2_reader.Value(key2);
      // The following will crash if the dimensions differ, but
      // they would likely also differ for all the xvectors so it's probably
      // best to just crash.
      BaseFloat score = SimilarityScore(xvector1, xvector2, S, b);
      sum += score;
      sumsq += score * score;
      num_done++;
      ko.Stream() << key1 << ' ' << key2 << ' ' << score << std::endl;
    }

    if (num_done != 0) {
      BaseFloat mean = sum / num_done, scatter = sumsq / num_done,
          variance = scatter - mean * mean, stddev = sqrt(variance);
      KALDI_LOG << "Mean score was " << mean << ", standard deviation was "
                << stddev;
    }
    KALDI_LOG << "Processed " << num_done << " trials " << num_err
              << " had errors.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
