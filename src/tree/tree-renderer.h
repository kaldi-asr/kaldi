// tree/tree-renderer.h

// Copyright 2012 Vassil Panayotov

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

#ifndef KALDI_TREE_TREE_RENDERER_H_
#define KALDI_TREE_TREE_RENDERER_H_

#include "base/kaldi-common.h"
#include "tree/event-map.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "fst/fstlib.h"

namespace kaldi {

// Parses a decision tree file and outputs its description in GraphViz format
class TreeRenderer {
 public:
  const static int32 kEdgeWidth; // normal width of the edges and state contours
  const static int32 kEdgeWidthQuery; // edge and state width when in query
  const static std::string kEdgeColor; // normal color for states and edges
  const static std::string kEdgeColorQuery; // edge and state color when in query

  TreeRenderer(std::istream &is, bool binary, std::ostream &os,
               fst::SymbolTable &phone_syms, bool use_tooltips)
      : phone_syms_(phone_syms), is_(is), out_(os), binary_(binary),
        N_(-1), use_tooltips_(use_tooltips), next_id_(0) {}

  // Renders the tree and if the "query" parameter is not NULL
  // a distinctly colored trace corresponding to the event.
  void Render(const EventType *query);

 private:
  // Looks-up the next token from the stream and invokes
  // the appropriate render method to visualize it
  void RenderSubTree(const EventType *query, int32 id);

  // Renders a leaf node (constant event map)
  void RenderConstant(const EventType *query, int32 id);

  // Renders a split event map node and the edges to the nodes
  // representing YES and NO sets
  void RenderSplit(const EventType *query, int32 id);

  // Renders a table event map node and the edges to its (non-null) children
  void RenderTable(const EventType *query, int32 id);

  // Makes a comma-separated string from the elements of a set of identifiers
  // If the identifiers represent phones, their symbolic representations are used
  std::string MakeEdgeLabel(const EventKeyType &key,
                            const ConstIntegerSet<EventValueType> &intset);

  // Writes the GraphViz representation of a non-leaf node to the out stream
  // A question about a phone from the context window or about pdf-class
  // is used as a label.
  void RenderNonLeaf(int32 id, const EventKeyType &key, bool in_query);

  fst::SymbolTable &phone_syms_; // phone symbols to be used as edge labels
  std::istream &is_; // the stream from which the tree is read
  std::ostream &out_; // the GraphViz representation is written to this stream
  bool binary_; // is the input stream binary?
  int32 N_, P_; // context-width and central position
  bool use_tooltips_;  // use tooltips(useful in e.g. SVG) instead of labels
  int32 next_id_; // the first unused GraphViz node ID
};

} // namespace kaldi

#endif //  KALDI_TREE_TREE_RENDERER_H_
