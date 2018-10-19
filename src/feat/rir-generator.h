// feat/rir-generator.h

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

#ifndef KALDI_FEAT_RIR_GENERATOR_H_
#define KALDI_FEAT_RIR_GENERATOR_H_

#include <iterator>
#include <map>

#include "base/kaldi-common.h"
#include "util/common-utils.h"

namespace kaldi {

typedef enum {
  kBidirectional,
  kHypercardioid,
  kCardioid,
  kSubcardioid,
  kOmnidirectional
} PolorPattern;

struct Point3D {
  BaseFloat x, y, z;
  Point3D() : x(0), y(0), z(0) {}
  Point3D(const Point3D &p) : x(p.x), y(p.y), z(p.z) {}

  Point3D(BaseFloat x, BaseFloat y, BaseFloat z) : x(x), y(y), z(z) {}
  BaseFloat L2Norm() const { return sqrt(x * x + y * y + z * z); }
  BaseFloat V() const { return x * y * z; }
  BaseFloat S() const { return 2 * (x * y + x * z + y * z); }
  void Scale(BaseFloat s) {
    x = x * s;
    y = y * s;
    z = z * s;
  }
  void Reset() { x = y = z = 0; }
  void CopyFromVector(const std::vector<BaseFloat> &v) {
    KALDI_ASSERT(v.size() <= 3);
    if (v.size() >= 1) x = v[0];
    if (v.size() >= 2) y = v[1];
    if (v.size() >= 3) z = v[2];
  }
};

struct RirGeneratorOptions {
  BaseFloat sound_velocity, samp_frequency;
  bool hp_filter;
  int32 num_samples, order;

  std::string source_location, receiver_location, room_topo, orientation, beta,
      microphone_type;


  RirGeneratorOptions()
      : sound_velocity(340),
        samp_frequency(16000),
        hp_filter(true),
        num_samples(-1),
        order(-1),
        source_location(""),
        receiver_location(""),
        room_topo(""),
        orientation(""),
        beta(""),
        microphone_type("omnidirectional") { }

  void Register(OptionsItf *opts) {
    opts->Register("sound-velocity", &sound_velocity, "Sound velocity in m/s");
    opts->Register("samp-frequency", &samp_frequency,
                   "Sampling frequency in Hz");
    opts->Register("hp-filter", &hp_filter,
                   "If true, high-pass filter is enabled");
    opts->Register("number-samples", &num_samples,
                   "Number of samples to calculate");
    opts->Register("order", &order,
                   "Reflection order, default is -1, i.e. maximum order");
    opts->Register("microphone-type", &microphone_type,
                   "Type of micrphone arrays("
                   "\"omnidirectional\"|\"subcardioid\"|\"cardioid\"|"
                   "\"hypercardioid\"|\"bidirectional\")");
    opts->Register("receiver-location", &receiver_location,
                   "3D-Coordinates of receivers(in meters). "
                   "Each coordinate is separated by a single semicolon, egs: "
                   "--receiver-location=\"2,1.5,2;1,1.5,2\"");
    opts->Register("source-location", &source_location,
                   "3D Coordinates of receivers(in meters). "
                   "egs: --source-location=\"2,3.5,2\"");
    opts->Register("room-topo", &room_topo,
                   "Room dimensions(in meters) egs: --room-dim=\"5,4,6\"");
    opts->Register("angle", &orientation,
                   "Direction in which the microphones are pointed, "
                   "specified using azimuth and elevation angles(in radians)");
    opts->Register("beta", &beta,
                   "6D vector specifying the reflection coefficients or "
                   "reverberation time(T60) in seconds.");
  }
};

class RirGenerator {
 public:
  explicit RirGenerator(const RirGeneratorOptions &opts)
      : opts_(opts),
        velocity_(opts.sound_velocity),
        frequency_(opts.samp_frequency),
        hp_filter_(opts.hp_filter),
        num_samples_(opts.num_samples),
        order_(opts.order) {
    ComputeDerived();
  }

  void GenerateRir(Matrix<BaseFloat> *rir);

  std::string Report();

  BaseFloat Frequency() { return frequency_; }

 private:
  RirGeneratorOptions opts_;

  BaseFloat velocity_, frequency_, revb_time_;
  bool hp_filter_;
  int32 num_samples_, order_, room_dim_, num_mics_;

  std::vector<Point3D> receiver_location_;
  Point3D source_location_, room_topo_;
  std::vector<BaseFloat> angle_, beta_;

  std::map<std::string, PolorPattern> str_to_pattern_ = {
      {"omnidirectional", kOmnidirectional},
      {"subcardioid", kSubcardioid},
      {"cardioid", kCardioid},
      {"hypercardioid", kHypercardioid},
      {"bidirectional", kBidirectional}};

  void ComputeDerived();

  BaseFloat MicrophoneSim(const Point3D &p);
};

double Sinc(double x) { return x == 0 ? 1.0 : std::sin(x) / x; }

// Compute RT60 using sabine formula
BaseFloat Sabine(const Point3D &room_topo, const std::vector<BaseFloat> &beta,
                 BaseFloat c) {
  BaseFloat V = room_topo.V();
  BaseFloat alpha = ((1 - pow(beta[0], 2)) + (1 - pow(beta[1], 2))) *
                        room_topo.y * room_topo.z +
                    ((1 - pow(beta[2], 2)) + (1 - pow(beta[3], 2))) *
                        room_topo.x * room_topo.z +
                    ((1 - pow(beta[4], 2)) + (1 - pow(beta[5], 2))) *
                        room_topo.x * room_topo.y;
  BaseFloat revb_time = 24 * Log(10.0) * V / (c * alpha);
  return revb_time;
}

}  // namespace kaldi

#endif  // KALDI_FEAT_RIR_GENERATOR_H_

