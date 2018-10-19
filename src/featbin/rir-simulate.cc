// featbin/rir-simulate.cc

// Copyright 2018 Jian Wu

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

#include "feat/wave-reader.h"
#include "feat/rir-generator.h"

int main(int argc, char const* argv[]) {
  try {
    using namespace kaldi;

    const char* usage =
        "Computes the response of an acoustic source to one or more "
        "microphones "
        "in a reverberant room using the image method.\n"
        "Reference: https://github.com/ehabets/RIR-Generator\n"
        "\n"
        "Usage: rir-simulate [options] <wav-wspecifier>\n"
        "See also: wav-reverberate\n";

    ParseOptions po(usage);

    bool report = false, normalize = false;
    po.Register("report", &report, "If true, output RirGenerator's statistics");
    po.Register("normalize", &normalize,
                "If true, normalize output room impluse response");

    RirGeneratorOptions generator_opts;
    generator_opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 1) {
      po.PrintUsage();
      exit(1);
    }

    RirGenerator generator(generator_opts);
    Matrix<BaseFloat> rir;
    BaseFloat int16_max =
        static_cast<BaseFloat>(std::numeric_limits<int16>::max());

    generator.GenerateRir(&rir);

    if (normalize) {
      rir.Scale(1.0 / rir.LargestAbsElem());
    }
    rir.Scale(int16_max);

    if (report) std::cout << generator.Report();

    std::string target_rir = po.GetArg(1);
    Output ko(target_rir, true, false);
    WaveData rir_simu(generator.Frequency(), rir);
    rir_simu.Write(ko.Stream());
  } catch (const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
  return 0;
}
