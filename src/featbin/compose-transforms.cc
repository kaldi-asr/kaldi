// featbin/compose-transforms.cc

// Copyright 2009-2012  Microsoft Corporation
//                      Johns Hopkins University (author: Daniel Povey)

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
#include "matrix/kaldi-matrix.h"
#include "transform/transform-common.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Compose (affine or linear) feature transforms\n"
        "Usage: compose-transforms [options] (<transform-A-rspecifier>|<transform-A-rxfilename>) "
        "(<transform-B-rspecifier>|<transform-B-rxfilename>) (<transform-out-wspecifier>|<transform-out-wxfilename>)\n"
        " Note: it does matrix multiplication (A B) so B is the transform that gets applied\n"
        "  to the features first.  If b-is-affine = true, then assume last column of b corresponds to offset\n"
        " e.g.: compose-transforms 1.mat 2.mat 3.mat\n"
        "   compose-transforms 1.mat ark:2.trans ark:3.trans\n"
        "   compose-transforms ark:1.trans ark:2.trans ark:3.trans\n"
        " See also: transform-feats, transform-vec, extend-transform-dim, est-lda, est-pca\n";

    bool b_is_affine = false;
    bool binary = true;
    std::string utt2spk_rspecifier;
    ParseOptions po(usage);

    po.Register("utt2spk", &utt2spk_rspecifier, "rspecifier for utterance to speaker map (if mixing utterance and speaker ids)");

    po.Register("b-is-affine", &b_is_affine, "If true, treat last column of transform b as an offset term (only relevant if a is affine)");
    po.Register("binary", &binary, "Write in binary mode (only relevant if output is a wxfilename)");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }


    std::string transform_a_fn = po.GetArg(1);
    std::string transform_b_fn = po.GetArg(2);
    std::string transform_c_fn = po.GetArg(3);

    // all these "fn"'s are either rspecifiers or filenames.

    bool a_is_rspecifier =
        (ClassifyRspecifier(transform_a_fn, NULL, NULL)
         != kNoRspecifier),
        b_is_rspecifier =
        (ClassifyRspecifier(transform_b_fn, NULL, NULL)
         != kNoRspecifier),
        c_is_wspecifier =
        (ClassifyWspecifier(transform_c_fn, NULL, NULL, NULL)
         != kNoWspecifier);


    RandomAccessTokenReader utt2spk_reader;
    if (utt2spk_rspecifier != "") {
      if (!(a_is_rspecifier && b_is_rspecifier))
        KALDI_ERR << "Error: utt2spk option provided compose transforms but "
            "at least one of the inputs is a global transform.";
      if (!utt2spk_reader.Open(utt2spk_rspecifier))
        KALDI_ERR << "Error upening utt2spk map from "
                   << utt2spk_rspecifier;
    }


    if ( (a_is_rspecifier || b_is_rspecifier) !=  c_is_wspecifier)
      KALDI_ERR << "Formats of the input and output rspecifiers/rxfilenames do "
          "not match (if either a or b is an rspecifier, then the output must "
          "be a wspecifier.";


    if (a_is_rspecifier || b_is_rspecifier) {
      BaseFloatMatrixWriter c_writer(transform_c_fn);
      if (a_is_rspecifier) {
        SequentialBaseFloatMatrixReader a_reader(transform_a_fn);
        if (b_is_rspecifier) {  // both are rspecifiers.
          RandomAccessBaseFloatMatrixReader b_reader(transform_b_fn);
          for (;!a_reader.Done(); a_reader.Next()) {
            if (utt2spk_rspecifier != "") {  // assume a is per-utt, b is per-spk.
              std::string utt = a_reader.Key();
              if (!utt2spk_reader.HasKey(utt)) {
                KALDI_WARN << "No speaker provided for utterance " << utt
                           << " (perhaps you wrongly provided utt2spk option to "
                    " compose-transforms?)";
                continue;
              }
              std::string spk = utt2spk_reader.Value(utt);
              if (!b_reader.HasKey(spk)) {
                KALDI_WARN << "Second table does not have key " << spk;
                continue;
              }
              Matrix<BaseFloat> c;
              if (!ComposeTransforms(a_reader.Value(), b_reader.Value(a_reader.Key()),
                                    b_is_affine, &c))
                continue;  // warning will have been printed already.
              c_writer.Write(utt, c);
            } else {  // Normal case: either both per-utterance or both per-speaker.
              if (!b_reader.HasKey(a_reader.Key())) {
                KALDI_WARN << "Second table does not have key " << a_reader.Key();
              } else {
                Matrix<BaseFloat> c;
                if (!ComposeTransforms(a_reader.Value(), b_reader.Value(a_reader.Key()),
                                      b_is_affine, &c))
                  continue;  // warning will have been printed already.
                c_writer.Write(a_reader.Key(), c);
              }
            }
          }
        } else {  // a is rspecifier,  b is rxfilename
          Matrix<BaseFloat> b;
          ReadKaldiObject(transform_b_fn, &b);
          for (;!a_reader.Done(); a_reader.Next()) {
            Matrix<BaseFloat> c;
            if (!ComposeTransforms(a_reader.Value(), b,
                                  b_is_affine, &c))
              continue;  // warning will have been printed already.
            c_writer.Write(a_reader.Key(), c);
          }
        }
      } else {
        Matrix<BaseFloat> a;
        ReadKaldiObject(transform_a_fn, &a);
        SequentialBaseFloatMatrixReader b_reader(transform_b_fn);
        for (; !b_reader.Done(); b_reader.Next()) {
          Matrix<BaseFloat> c;
          if (!ComposeTransforms(a, b_reader.Value(),
                                b_is_affine, &c))
            continue;  // warning will have been printed already.
          c_writer.Write(b_reader.Key(), c);
        }
      }
    } else {  // all are just {rx, wx}filenames.
      Matrix<BaseFloat> a;
      ReadKaldiObject(transform_a_fn, &a);
      Matrix<BaseFloat> b;
      ReadKaldiObject(transform_b_fn, &b);
      Matrix<BaseFloat> c;
      if (!b_is_affine && a.NumRows() == a.NumCols()+1 && a.NumRows() == b.NumRows()
          && a.NumCols() == b.NumCols())
        KALDI_WARN << "It looks like you are trying to compose two affine transforms"
                   << ", but you omitted the --b-is-affine option.";
      if (!ComposeTransforms(a, b, b_is_affine, &c)) exit (1);

      WriteKaldiObject(c, transform_c_fn, binary);
    }
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


