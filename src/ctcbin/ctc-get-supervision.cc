// ctcbin/ctc-get-supervision.cc

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
#include "ctc/cctc-transition-model.h"
#include "ctc/cctc-supervision.h"

namespace kaldi {
namespace ctc {


// Processes the CctcProtoSupervision object into a CctcSupervision
// object, and writes it to the writer if successful and returns true;
// otherwise returns false.
static bool ProcessSupervision(const CctcSupervisionOptions &opts,
                               const CctcTransitionModel &trans_model,
                               const std::string &key,
                               CctcProtoSupervision *proto_supervision,
                               CctcSupervisionWriter *supervision_writer) {
  MakeSilencesOptional(opts, proto_supervision);
  ModifyProtoSupervisionTimes(opts, proto_supervision);
  AddBlanksToProtoSupervision(proto_supervision);
  CctcSupervision supervision;
  if (!MakeCctcSupervisionNoContext(*proto_supervision, trans_model.NumPhones(),
                                    &supervision)) {
    // the only way this should fail is if we had too many phones for
    // the number of subsampled frames.    
    KALDI_LOG << "Failed to create CtcSupervision for " << key
              << " (because too many phones for too few frames?)";
    return false;
  }
  AddContextToCctcSupervision(trans_model, &supervision);
  supervision_writer->Write(key, supervision);
  return true;
}


} // namespace nnet2
} // namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::ctc;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Get a CTC supervision object for each file of training data.\n"
        "This will normally be piped into nnet3-ctc-get-egs, where it\n"
        "will be split up into pieces and combined with the features.\n"
        "Input can come in two formats: from alignments\n"
        "(from ali-to-phones --write-lenghts=true), or from lattices\n"
        "(e.g. derived from aligning the data, see steps/align_fmllr_lats.sh)\n"
        "that have been converged to phone-level lattices with\n"
        "lattice-align-phones --replace-output-symbols=true.\n"
        "\n"
        "Usage: ctc-get-supervision [options] <ctc-transition-model> <phones-with-lengths-rspecifier> "
        "<supervision-wspecifier>\n"
        "or: ctc-get-supervision --lattice-input=true [options] <ctc-transition-model> <phone-lattice-rspecifier> "
        "<supervision-wspecifier>\n";
    
    bool lattice_input = false;
    CctcSupervisionOptions sup_opts;
    
    ParseOptions po(usage);
    sup_opts.Register(&po);
    po.Register("lattice-input", &lattice_input, "If true, expect phone "
                "lattices as input");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string ctc_trans_model_rxfilename = po.GetArg(1),
        phone_durs_or_lat_rspecifier = po.GetArg(2),
        supervision_wspecifier = po.GetArg(3);    

    CctcTransitionModel trans_model;
    ReadKaldiObject(ctc_trans_model_rxfilename, &trans_model);
    
    CctcSupervisionWriter supervision_writer(supervision_wspecifier);

    int32 num_utts_done = 0, num_utts_error = 0;

    if (lattice_input) {
      SequentialCompactLatticeReader clat_reader(phone_durs_or_lat_rspecifier);
      for (; !clat_reader.Done(); clat_reader.Next()) {
        std::string key = clat_reader.Key();
        const CompactLattice &clat = clat_reader.Value();
        CctcProtoSupervision proto_supervision;
        PhoneLatticeToProtoSupervision(clat, &proto_supervision);
        if (ProcessSupervision(sup_opts, trans_model, key,
                               &proto_supervision, &supervision_writer))
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
        CctcProtoSupervision proto_supervision;
        AlignmentToProtoSupervision(ali, &proto_supervision);
        if (ProcessSupervision(sup_opts, trans_model, key,
                               &proto_supervision, &supervision_writer))
          num_utts_done++;
        else
          num_utts_error++;
      }
    }
    KALDI_LOG << "Generated CTC supervision information for "
              << num_utts_done << " utterances, errors on "
              << num_utts_error;
    return (num_utts_done > num_utts_error ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
