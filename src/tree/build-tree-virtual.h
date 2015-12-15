// tree/build-tree-virtual.h

// Hainan Xu

#ifndef KALDI_TREE_BUILD_TREE_VIRTUAL_H_
#define KALDI_TREE_BUILD_TREE_VIRTUAL_H_

#include "tree/build-tree-utils.h"
#include "tree/context-dep.h"
#include "util/stl-utils.h"
#include <queue>

using std::vector;
using std::set;
using std::map;
using std::priority_queue;
using std::string;

namespace kaldi {

void WriteMultiTreeMapping(const unordered_map<int32, vector<int32> >& mappings,
                  std::ostream &os, bool binary, size_t num_trees);

void ReadMultiTreeMapping(unordered_map<int32, vector<int32> >& mappings,
                 std::istream &is, bool binary,
                 size_t num_trees = 0); // last parameter not necessary

void MappingToSparseMatrix(const unordered_map<int32, vector<int32> >& mapping,
                           SparseMatrix<BaseFloat> *out);

struct MultiTreeNodeInfo {
  EventMap* parent; 
  const EventMap* src_node;  // corresponding node in the source tree
  map<EventKeyType, set<EventValueType> > possible_values;
  vector<int32> pdfs;
};

class MultiTreePdfMap {
 public:
  MultiTreePdfMap(const vector<const EventMap*> &trees,
                  size_t cxt_length, size_t center_phone,
                  const vector<int32> &hmm_lengths);
  // return pointer owned by the caller
  EventMap* GenerateVirtualTree(
          unordered_map<int32, vector<int32> >& mappings);

 private:
  void ConnectToNextTree(const MultiTreeNodeInfo &this_node_info,
                         EventMap* this_node, vector<int32>& pdfs_of_leaf);

  void ConnectToSplitNode(const MultiTreeNodeInfo &this_node_info,
                          EventMap* this_node,
                          EventMap* yes,
                          EventMap* no,
                          EventMap* yes_node,
                          EventMap* no_node);

  void SetTrees(const vector<const EventMap*>& trees);
  void SetCxtLength(size_t length);
  void SetCenterPhone(size_t center_phone);
  void SetHmmLengths(const vector<int32> &hmm_lengths);

  // a vector of roots of trees
  // pointers *not* owned in this class
  vector<const EventMap*> trees_;
  size_t cxt_length_;
  size_t center_phone_;
  size_t num_trees_;
  vector<int32> hmm_lengths_;
  set<int> all_phones_;

  // for generation algorithm
  priority_queue<std::pair<EventMap*, int> > queue;
  typedef unordered_map<EventMap*, MultiTreeNodeInfo> InfoMap;
  InfoMap info_map;

  size_t cur_prio_;  // current priority
};

}  // namespace kaldi

#endif

