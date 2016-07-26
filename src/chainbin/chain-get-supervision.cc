// chainbin/chain-get-supervision.cc

// Copyright      2015  Johns Hopkins University (author:  Daniel Povey)

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

#include <sstream>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "chain/chain-supervision.h"

namespace kaldi {
namespace chain {


// This wrapper function does all the job of processing the features and
// lattice into ChainSupervision objects, and writing them out.
static bool ProcessSupervision(const TransitionModel &trans_model,
                               const ContextDependencyInterface &ctx_dep,
                               const ProtoSupervision &proto_sup,
                               const std::string &key,
                               SupervisionWriter *supervision_writer) {
  Supervision supervision;
  if (!ProtoSupervisionToSupervision(ctx_dep, trans_model,
                                     proto_sup, &supervision)) {
    KALDI_WARN << "Failed creating supervision for utterance "
               << key;
    return false;
  }
  if (RandInt(0, 10) == 0)
    supervision.Check(trans_model);

  supervision_writer->Write(key, supervision);
  return true;
}


} // namespace chain
} // namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::chain;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Get a 'chain' supervision object for each file of training data.\n"
        "This will normally be piped into nnet3-chain-get-egs, where it\n"
        "will be split up into pieces and combined with the features.\n"
        "Input can come in two formats: from alignments\n"
        "(from ali-to-phones --write-lenghts=true), or from lattices\n"
        "(e.g. derived from aligning the data, see steps/align_fmllr_lats.sh)\n"
        "that have been converged to phone-level lattices with\n"
        "lattice-align-phones --replace-output-symbols=true.\n"
        "\n"
        "Usage: chain-get-supervision [options] <tree> <transition-model> "
        "[<phones-with-lengths-rspecifier>|<phone-lattice-rspecifier>] <supervision-wspecifier>\n"
        "See steps/nnet3/chain/get_egs.sh for example\n";


    bool lattice_input = false;
    SupervisionOptions sup_opts;

    ParseOptions po(usage);
    sup_opts.Register(&po);
    po.Register("lattice-input", &lattice_input, "If true, expect phone "
                "lattices as input");

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string tree_rxfilename = po.GetArg(1),
        trans_model_rxfilename = po.GetArg(2),
        phone_durs_or_lat_rspecifier = po.GetArg(3),
        supervision_wspecifier = po.GetArg(4);

    TransitionModel trans_model;
    ReadKaldiObject(trans_model_rxfilename, &trans_model);

    ContextDependency ctx_dep;
    ReadKaldiObject(tree_rxfilename, &ctx_dep);

    SupervisionWriter supervision_writer(supervision_wspecifier);

    int32 num_utts_done = 0, num_utts_error = 0;

    if (lattice_input) {
      SequentialCompactLatticeReader clat_reader(phone_durs_or_lat_rspecifier);
      for (; !clat_reader.Done(); clat_reader.Next()) {
        std::string key = clat_reader.Key();
        const CompactLattice &clat = clat_reader.Value();
        ProtoSupervision proto_supervision;
        bool ans = PhoneLatticeToProtoSupervision(sup_opts, clat,
                                                  &proto_supervision);
        if (!ans) {
          KALDI_WARN << "Error creating proto-supervision for utterance " << key;
          num_utts_error++;
          continue;
        }
        if (ProcessSupervision(trans_model, ctx_dep,
                               proto_supervision, key, &supervision_writer))
          num_utts_done++;
        else
          num_utts_error++;
      }
    } else {
      SequentialInt32PairVectorReader phone_and_dur_reader(
          phone_durs_or_lat_rspecifier);
      for (; !phone_and_dur_reader.Done(); phone_and_dur_reader.Next()) {
        std::string key = phone_and_dur_reader.Key();
        const std::vector<std::pair<int32,int32> > &ali =
            phone_and_dur_reader.Value();
        ProtoSupervision proto_supervision;
        AlignmentToProtoSupervision(sup_opts, ali,
                                    &proto_supervision);
        if (ProcessSupervision(trans_model, ctx_dep,
                               proto_supervision, key, &supervision_writer))
          num_utts_done++;
        else
          num_utts_error++;
      }
    }
    KALDI_LOG << "Generated chain supervision information for "
              << num_utts_done << " utterances, errors on "
              << num_utts_error;
    return (num_utts_done > num_utts_error ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
