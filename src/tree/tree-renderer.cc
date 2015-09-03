// tree/tree-renderer.cc

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

#include <tree/tree-renderer.h>

namespace kaldi {
const int32 TreeRenderer::kEdgeWidth = 1;
const int32 TreeRenderer::kEdgeWidthQuery = 3;
const std::string TreeRenderer::kEdgeColor = "black";
const std::string TreeRenderer::kEdgeColorQuery = "red";

void
TreeRenderer::RenderNonLeaf(int32 id, const EventKeyType &key, bool in_query) {
  std::string color = in_query? kEdgeColorQuery: kEdgeColor;
  int32 width = in_query? kEdgeWidthQuery: kEdgeWidth;
  std::string label;
  if (key == kPdfClass) {
    label = "\"PdfClass = ?\"";
  } else if (key == 0) {
    if (N_ == 1 && P_ == 0) // monophone tree?
      label = "\"Phone = ?\"";
    else if (N_ == 3 && P_ == 1) // triphone tree?
      label = "\"LContext = ?\"";
  } else if (key == 2 && N_ == 3 && P_ == 1) {
    label = "\"RContext = ?\"";
  } else if (key >= 0 && key <= N_-1) {
    if (P_ == key)
      label = "\"Center = ?\"";
    else {
      std::ostringstream oss;
      oss << "\"Ctx Position " << key << " = ?\"";
      label = oss.str();
    }
  } else {
    KALDI_ERR << "Invalid decision tree key: " << key;
  }

  out_ << id << "[label=" << label << ", color=" << color
       << ", penwidth=" << width << "];" << std::endl;
}

std::string
TreeRenderer::MakeEdgeLabel(const EventKeyType &key,
                            const ConstIntegerSet<EventValueType> &intset) {
  std::ostringstream oss;
  ConstIntegerSet<EventValueType>::iterator child = intset.begin();
  for (; child != intset.end(); ++child) {
    if (child != intset.begin())
      oss << ", ";
    if (key != kPdfClass) {
      std::string phone =
          phone_syms_.Find(static_cast<kaldi::int64>(*child));
      if (phone.empty())
        KALDI_ERR << "No phone found for Phone ID " << *child;
      oss << phone;
    } else {
      oss << *child;
    }
  }

  return oss.str();
}

void TreeRenderer::RenderSplit(const EventType *query, int32 id) {
  ExpectToken(is_, binary_, "SE");
  EventKeyType key;
  ReadBasicType(is_, binary_, &key);
  ConstIntegerSet<EventValueType> yes_set;
  yes_set.Read(is_, binary_);
  ExpectToken(is_, binary_, "{");

  EventValueType value = -30000000; // just a value I guess is invalid
  if (query != NULL)
    EventMap::Lookup(*query, key, &value);
  const EventType *query_yes = yes_set.count(value)? query: NULL;
  const EventType *query_no = (query_yes == NULL)? query: NULL;
  std::string color_yes = (query_yes)? kEdgeColorQuery: kEdgeColor;
  std::string color_no = (query && !query_yes)? kEdgeColorQuery: kEdgeColor;
  int32 width_yes = (query_yes)? kEdgeWidthQuery: kEdgeWidth;
  int32 width_no = (query && !query_yes)? kEdgeWidthQuery: kEdgeWidth;
  RenderNonLeaf(id, key, (query != NULL)); // Draw the node itself
  std::string yes_label = MakeEdgeLabel(key, yes_set);
  out_ << "\t" << id << " -> " << next_id_++ << " ["; // YES edge
  if (use_tooltips_) {
    out_ << "tooltip=\"" << yes_label << "\", label=YES"
         << ", penwidth=" << width_yes << ", color=" << color_yes << "];\n";
  } else {
    out_ << "label=\"" << yes_label << "\", penwidth=" << width_yes
         << ", penwidth=" << width_yes << ", color=" << color_yes << "];\n";
  }
  RenderSubTree(query_yes, next_id_-1); // Render YES subtree
  out_ << "\t" << id << " -> " << next_id_++ << "[label=NO" // NO edge
       << ", color=" << color_no << ", penwidth=" << width_no << "];\n";
  RenderSubTree(query_no, next_id_-1); // Render NO subtree

  ExpectToken(is_, binary_, "}");
}

void TreeRenderer::RenderTable(const EventType *query, int32 id) {
  ExpectToken(is_, binary_, "TE");
  EventKeyType key;
  ReadBasicType(is_, binary_, &key);
  uint32 size;
  ReadBasicType(is_, binary_, &size);
  ExpectToken(is_, binary_, "(");

  EventValueType value = -3000000; // just a value I hope is invalid
  if (query != NULL)
    EventMap::Lookup(*query, key, &value);
  RenderNonLeaf(id, key, (query != NULL));
  for (size_t t = 0; t < size; t++) {
    std::string color = (t == value)? kEdgeColorQuery: kEdgeColor;
    int32 width = (t==value)? kEdgeWidthQuery: kEdgeWidth;
    std::ostringstream label;
    if (key == kPdfClass) {
      label << t;
    } else if (key >= 0 && key < N_) {
      if (t == 0) {
        ExpectToken(is_, binary_, "NULL"); // consume the invalid/NULL entry
        continue;
      }
      std::string phone = phone_syms_.Find(static_cast<kaldi::int64>(t));
      if (phone.empty())
          KALDI_ERR << "Phone ID found in a TableEventMap, but not in the "
                    << "phone symbol table! ID: " << t;
      label << phone;
    } else {
      KALDI_ERR << "TableEventMap: Invalid event key: " << key;
    }
    // draw the edge to the child subtree
    out_ << "\t" << id << " -> " << next_id_++ << " [label=" << label.str()
         << ", color=" << color << ", penwidth=" << width <<  "];\n";
    const EventType *query_child = (t == value)? query: NULL;
    RenderSubTree(query_child, next_id_-1); // render the child subtree
  }

  ExpectToken(is_, binary_, ")");
}

void TreeRenderer::RenderConstant(const EventType *query, int32 id) {
  ExpectToken(is_, binary_, "CE");
  EventAnswerType answer;
  ReadBasicType(is_, binary_, &answer);

  std::string color = (query!=NULL)? kEdgeColorQuery: kEdgeColor;
  int32 width = (query!=NULL)? kEdgeWidthQuery: kEdgeWidth;
  out_ << id << "[shape=doublecircle, label=" << answer
       << ",color=" << color << ", penwidth=" << width << "];\n";
}

void TreeRenderer::RenderSubTree(const EventType *query, int32 id) {
  char c = Peek(is_, binary_);
  if (c == 'N') {
    ExpectToken(is_, binary_, "NULL"); // consume NULL entries
    return;
  } else if (c == 'C') {
    RenderConstant(query, id);
  } else if (c == 'T') {
    RenderTable(query, id);
  } else if (c == 'S') {
    RenderSplit(query, id);
  } else {
    KALDI_ERR << "EventMap::read, was not expecting character " << CharToString(c)
              << ", at file position " << is_.tellg();
  }
}

void TreeRenderer::Render(const EventType *query = 0) {
  ExpectToken(is_, binary_, "ContextDependency");
  ReadBasicType(is_, binary_, &N_);
  ReadBasicType(is_, binary_, &P_);
  ExpectToken(is_, binary_, "ToPdf");
  if (query && query->size() != N_+1)
    KALDI_ERR << "Invalid query size \"" << query->size() << "\"! Expected \""
              << N_+1 << '"';
  out_ << "digraph EventMap {\n";
  RenderSubTree(query, next_id_++);
  out_ << "}\n";
  ExpectToken(is_, binary_, "EndContextDependency");
}

} // namespace kaldi
