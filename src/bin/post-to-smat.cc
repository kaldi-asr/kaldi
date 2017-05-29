// bin/post-to-smat.cc

// Copyright 2017   Johns Hopkins University (Author: Daniel Povey)

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
#include "hmm/posterior.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "This program turns an archive of per-frame posteriors, e.g. from\n"
        "ali-to-post | post-to-pdf-post,\n"
        "into an archive of SparseMatrix.  This is just a format transformation.\n"
        "This may not make sense if the indexes in question are one-based (at least,\n"
        "you'd have to increase the dimension by one.\n"
        "\n"
        "See also: post-to-phone-post, ali-to-post, post-to-pdf-post\n"
        "\n"
        "Usage:  post-to-smat [options] <posteriors-rspecifier> <sparse-matrix-wspecifier>\n"
        "e.g.: post-to-smat --dim=1038 ark:- ark:-\n";

    ParseOptions po(usage);

    int32 dim = -1;

    po.Register("dim", &dim, "The num-cols in each output SparseMatrix.  All "
                "the integers in the input posteriors are expected to be \n"
                ">= 0 and < dim.  This must be specified.");

    po.Read(argc, argv);

    if (dim <= 0) {
      KALDI_ERR << "The --dim option must be specified.";
    }

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string  posteriors_rspecifier = po.GetArg(1),
        sparse_matrix_wspecifier = po.GetArg(2);


    SequentialPosteriorReader posterior_reader(posteriors_rspecifier);

    TableWriter<KaldiObjectHolder<SparseMatrix<BaseFloat> > > sparse_matrix_writer(
        sparse_matrix_wspecifier);

    int32 num_done = 0;
    for (; !posterior_reader.Done(); posterior_reader.Next()) {
      const kaldi::Posterior &posterior = posterior_reader.Value();
      // The following constructor will throw an error if there is some kind of
      // dimension mismatch.
      SparseMatrix<BaseFloat> smat(dim, posterior);
      sparse_matrix_writer.Write(posterior_reader.Key(), smat);
      num_done++;
    }
    KALDI_LOG << "Done converting " << num_done
              << " posteriors into sparse matrices.";
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
