// gmmbin/gmm-ismooth-stats.cc

// Copyright 2009-2011  Petr Motlicek  Chao Weng

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
#include "gmm/am-diag-gmm.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "gmm/ebw-diag-gmm.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage =
        "Apply I-smoothing to statistics, e.g. for discriminative training\n"
        "Usage:  gmm-ismooth-stats [options] [--smooth-from-model] [<src-stats-in>|<src-model-in>] <dst-stats-in> <stats-out>\n"
        "e.g.: gmm-ismooth-stats --tau=100 ml.acc num.acc smoothed.acc\n"
        "or: gmm-ismooth-stats --tau=50 --smooth-from-model 1.mdl num.acc smoothed.acc\n"
        "or: gmm-ismooth-stats --tau=100 num.acc num.acc smoothed.acc\n";
        
    bool binary_write = false;
    bool smooth_from_model = false;
    BaseFloat tau = 100;
    
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("smooth-from-model", &smooth_from_model, "If true, "
                "expect first argument to be a model file");
    po.Register("tau", &tau, "Tau value for I-smoothing");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string src_stats_or_model_filename = po.GetArg(1),
        dst_stats_filename = po.GetArg(2),
        stats_out_filename = po.GetArg(3);

    double tot_count_before, tot_count_after;

    if (src_stats_or_model_filename == dst_stats_filename) { // as an optimization, just read once.
      KALDI_ASSERT(!smooth_from_model);
      Vector<double> transition_accs;
      AccumAmDiagGmm stats;
      {
        bool binary;
        Input ki(dst_stats_filename, &binary);
        transition_accs.Read(ki.Stream(), binary);
        stats.Read(ki.Stream(), binary, true);  // true == add; doesn't matter here.
      }
      tot_count_before = stats.TotStatsCount();
      IsmoothStatsAmDiagGmm(stats, tau, &stats);
      tot_count_after = stats.TotStatsCount();
      Output ko(stats_out_filename, binary_write);
      transition_accs.Write(ko.Stream(), binary_write);
      stats.Write(ko.Stream(), binary_write);
    } else if (smooth_from_model) { // Smoothing from model...
      AmDiagGmm am_gmm;
      TransitionModel trans_model;
      Vector<double> dst_transition_accs;
      AccumAmDiagGmm dst_stats;
      { // read src model
        bool binary;
        Input ki(src_stats_or_model_filename, &binary);
        trans_model.Read(ki.Stream(), binary);
        am_gmm.Read(ki.Stream(), binary);
      }
      { // read dst stats.
        bool binary;
        Input ki(dst_stats_filename, &binary);
        dst_transition_accs.Read(ki.Stream(), binary);
        dst_stats.Read(ki.Stream(), binary, true);  // true == add; doesn't matter here.
      }
      tot_count_before = dst_stats.TotStatsCount();
      IsmoothStatsAmDiagGmmFromModel(am_gmm, tau, &dst_stats);
      tot_count_after = dst_stats.TotStatsCount();
      Output ko(stats_out_filename, binary_write);
      dst_transition_accs.Write(ko.Stream(), binary_write);
      dst_stats.Write(ko.Stream(), binary_write);
    } else { // Smooth from stats.
      Vector<double> src_transition_accs;
      Vector<double> dst_transition_accs;
      AccumAmDiagGmm src_stats;
      AccumAmDiagGmm dst_stats;
      { // read src stats.
        bool binary;
        Input ki(src_stats_or_model_filename, &binary);
        src_transition_accs.Read(ki.Stream(), binary);
        src_stats.Read(ki.Stream(), binary, true);  // true == add; doesn't matter here.
      }
      { // read dst stats.
        bool binary;
        Input ki(dst_stats_filename, &binary);
        dst_transition_accs.Read(ki.Stream(), binary);
        dst_stats.Read(ki.Stream(), binary, true);  // true == add; doesn't matter here.
      }
      tot_count_before = dst_stats.TotStatsCount();
      IsmoothStatsAmDiagGmm(src_stats, tau, &dst_stats);
      tot_count_after = dst_stats.TotStatsCount();
      
      Output ko(stats_out_filename, binary_write);
      dst_transition_accs.Write(ko.Stream(), binary_write);
      dst_stats.Write(ko.Stream(), binary_write);
    }
    KALDI_LOG << "Smoothed stats with tau = " << tau << ", count changed from "
              << tot_count_before << " to " << tot_count_after;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
