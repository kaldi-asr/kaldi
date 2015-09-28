// ctc/cctc-decoder-wrappers.h

// Copyright   2015 Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_CTC_CTC_DECODER_WRAPPERS_H_
#define KALDI_CTC_CTC_DECODER_WRAPPERS_H_

#include "itf/options-itf.h"
#include "decoder/lattice-faster-decoder.h"
#include "decoder/lattice-simple-decoder.h"
#include "decoder/decoder-wrappers.h"
#include "ctc/cctc-transition-model.h"

// This header contains CTC versions of some things declared
// in ../decoder/decoder-wrappers.h that used the TransitionModel
// (so needed to be adapted to use CctcTransitionModel).

namespace kaldi {

/// This function DecodeUtteranceLatticeFaster is used in several decoders, and
/// we have moved it here.  Note: this is really "binary-level" code as it
/// involves table readers and writers; we've just put it here as there is no
/// other obvious place to put it.  If determinize == false, it writes to
/// lattice_writer, else to compact_lattice_writer.  The writers for
/// alignments and words will only be written to if they are open.
bool DecodeUtteranceLatticeFasterCctc(
    LatticeFasterDecoder &decoder, // not const but is really an input.
    DecodableInterface &decodable, // not const but is really an input.
    const ctc::CctcTransitionModel &trans_model,
    const fst::SymbolTable *word_syms,
    std::string utt,
    double acoustic_scale,
    bool determinize,
    bool allow_partial,
    Int32VectorWriter *alignments_writer,
    Int32VectorWriter *words_writer,
    CompactLatticeWriter *compact_lattice_writer,
    LatticeWriter *lattice_writer,
    double *like_ptr);  // puts utterance's likelihood in like_ptr on success.


} // end namespace kaldi.


#endif
