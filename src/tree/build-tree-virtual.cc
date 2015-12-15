#include "tree/build-tree-virtual.h"
#include "tree/event-map.h"

namespace kaldi {

void WriteMultiTreeMapping(const unordered_map<int32, vector<int32> >& mappings,
                  std::ostream &os, bool binary, size_t num_trees) {
/*
  The structure is
  map_size num_trees key_0 pdf0 pdf1 ... pdfm key_1 ...
*/
  WriteBasicType(os, binary, mappings.size());
  WriteBasicType(os, binary, num_trees);
  if (!binary) os << "\n";
  for (unordered_map<int32, vector<int32> >::const_iterator it 
           = mappings.begin();
       it != mappings.end(); it++) {
    WriteBasicType(os, binary, it->first);
    KALDI_ASSERT((it->second).size() == num_trees);
    if (!binary) {
      WriteToken(os, false, "->");
    }
    for (size_t i = 0; i < num_trees; i++) {
      WriteBasicType(os, binary, (it->second)[i]);
    }
    if (!binary) os << "\n";
  }
}


void ReadMultiTreeMapping(unordered_map<int32, vector<int32> >& mappings,
                  std::istream &is, bool binary, size_t num_trees) {
/*
  The structure is
  map_size num_trees key_0 pdf0 pdf1 ... pdfm key_1 ...
*/
  KALDI_ASSERT(mappings.empty());
  size_t map_size, num_trees_from_file;

  if (binary) {
    ReadBasicType(is, binary, &map_size);
    ReadBasicType(is, binary, &num_trees_from_file);
  }
  else {
    is >> map_size >> num_trees_from_file;
  }
  if (num_trees > 0) {
    // if num_trees is not passed, the default value is -1
    KALDI_ASSERT(num_trees == num_trees_from_file);
  }
  else {
    num_trees = num_trees_from_file;
  }
  int32 key, tmp;
  vector<int32> value;
  if (binary) {
    for (size_t i = 0; i < map_size; i++) {
      value.resize(0);
      ReadBasicType(is, true, &key);
      for (size_t j = 0; j < num_trees; j++) {
        ReadBasicType(is, true, &tmp);
        value.push_back(tmp);
      }
      mappings.insert(unordered_map<int32, vector<int32> >::
        value_type(key, value));
    }
  }
  else {
    for (size_t i = 0; i < map_size; i++) {
      value.resize(0);
      is >> key;
      ExpectToken(is, false, "->");
      for (size_t j = 0; j < num_trees; j++) {
        is >> tmp;
        value.push_back(tmp);
      }
      mappings.insert(unordered_map<int32, vector<int32> >::
        value_type(key, value));
    }
  }
}

MultiTreePdfMap::MultiTreePdfMap(const vector<const EventMap*> &trees,
                                 size_t cxt_length, size_t center_phone,
                                 const vector<int32> &hmm_lengths) {
  this->SetTrees(trees);
  this->SetCxtLength(cxt_length);
  this->SetCenterPhone(center_phone);
  this->SetHmmLengths(hmm_lengths);
  cur_prio_ = 0;
}

void MultiTreePdfMap::SetTrees(const vector<const EventMap*>& trees) {
  KALDI_ASSERT(!trees.empty());
  num_trees_ = trees.size();
  trees_ = trees;
}

void MultiTreePdfMap::SetCxtLength(size_t length) {
  cxt_length_ = length;
}

void MultiTreePdfMap::SetCenterPhone(size_t center_phone) {
  center_phone_ = center_phone;
}

void MultiTreePdfMap::SetHmmLengths(const vector<int32>& hmm_lengths) {
  hmm_lengths_ = hmm_lengths;

  for (size_t i = 0; i < hmm_lengths.size(); i++) {
    if (hmm_lengths[i] < 0) {  // check if it's a valid phone id
      continue;
    }
    all_phones_.insert(i);
  }
}

void MultiTreePdfMap::ConnectToNextTree(const MultiTreeNodeInfo& this_node_info,
                                        EventMap* this_node,
                                        vector<int32>& pdfs_of_leaf) {
  SplitEventMap *parent = dynamic_cast<SplitEventMap*>(this_node_info.parent);

  // the root of the next tree has to be a split map
  EventMap* new_node = new SplitEventMap();
  if (parent->yes_ == this_node) {
    parent->yes_ = new_node;
  }
  else {
    parent->no_ = new_node;
  }

  MultiTreeNodeInfo new_node_info;
  new_node_info.parent = parent;
  size_t next_tree_index = pdfs_of_leaf.size();
  new_node_info.src_node = trees_[next_tree_index];  // next tree
  new_node_info.possible_values = this_node_info.possible_values;
  new_node_info.pdfs = pdfs_of_leaf;
  delete this_node;
  info_map[new_node] = new_node_info;

  if (pdfs_of_leaf.size() == 1) {
    int32 num_pdfs = 0;

    // first tree's leaf reached, we gonna check that
    // all center phones allowed in this leaf should have the same num_pdf
    // this is not necessarily true for all case
    // (espeically for the randomly generated test code),
    // but usually true for real data recipes
    for (set<EventValueType>::iterator iter =
              new_node_info.possible_values[center_phone_].begin();
        iter != new_node_info.possible_values[center_phone_].end();
        ) {
      if (iter == new_node_info.possible_values[center_phone_].begin()) {
        num_pdfs = hmm_lengths_[*iter];  // *iter is the phone id
        if (*iter == 0) {
          KALDI_LOG << "this could be troubling!!!!";
        }
        iter++;
      }
      else {
        if (num_pdfs != hmm_lengths_[*iter]) {
          KALDI_LOG << "WARNING: allowed center phones in leaf "
                       " have different number of pdf classes "
                    << num_pdfs << " != " << hmm_lengths_[*iter]
                    << " at phone " << *iter;
          // not using the following assert since it fails on build-tree-test,
          // whose stats are randomized
          // KALDI_ASSERT(num_pdfs == hmm_lengths_[*iter]);
          break;  // if at least 2 of them aren't the same,
                  // there is no point in reducing the pdf numbers
        }

        if (++iter == new_node_info.possible_values[center_phone_].end()) {
          // now we are certain that all center phones have the same number
          // of pdf classes, we can further limit the possible values
          // by doing the following intersect operation
          set<EventValueType> new_possible_values;
          for (set<EventValueType>::iterator i =
                  new_node_info.possible_values[-1].begin();
               i != new_node_info.possible_values[-1].end();
               i++) {
            if (*i <= num_pdfs) {  // should be <= instead of <
              new_possible_values.insert(*i);
            }
          }
          new_node_info.possible_values[-1] = new_possible_values;
          break;
        }
      }
    }
  }

  // it is possible that after the intersect there is no possible pdf id
  // we only add this node if there is some possible pdf's
  if (new_node_info.possible_values[-1].size() != 0) {
    queue.push(std::make_pair(new_node, cur_prio_++));
  }
}

void MultiTreePdfMap::ConnectToSplitNode(const MultiTreeNodeInfo &this_node_info,
                                         EventMap* this_node,
                                         EventMap* yes,  // yes src
                                         EventMap* no,   // no src
                                         EventMap* yes_node,   // new yes
                                         EventMap* no_node) {  // new no
  SplitEventMap* p_split = dynamic_cast<SplitEventMap*>(this_node);
  // the possible_values for the children
  map<EventKeyType, set<EventValueType> > yes_map, no_map;
  yes_map = this_node_info.possible_values;
  no_map = yes_map;

  set<EventAnswerType> yes_keys = yes_map[p_split->key_];
  set<EventAnswerType> no_keys = yes_keys;  // they're initialized the same

  using std::cout;
  using std::endl;

  ConstIntegerSet<EventValueType>::iterator set_iter;
  for (set_iter = p_split->yes_set_.begin();
       set_iter != p_split->yes_set_.end();
       set_iter++) {
    no_keys.erase(*set_iter);  // erase by value
  }

  set<EventValueType>::iterator set_iterator;
  for (set_iterator = no_keys.begin();
       set_iterator != no_keys.end();
       set_iterator++) {
    yes_keys.erase(*set_iterator);  // yes_keys = all_keys - no_keys
  }

// need some change to make it work with const... TODO(hxu)
//  KALDI_ASSERT(yes_keys.size() + no_keys.size() == 
//                this_node_info.possible_values[p_split->key_].size());

  yes_map[p_split->key_] = yes_keys;
  no_map[p_split->key_] = no_keys;

  if (yes_keys.size() == 0) {  // then use the no node to replace this node
    SplitEventMap *parent =
           dynamic_cast<SplitEventMap*>(this_node_info.parent);
    KALDI_ASSERT(parent != NULL);

    MultiTreeNodeInfo new_node_info;
    new_node_info.parent = parent;
    new_node_info.src_node = no;
    new_node_info.possible_values = no_map;
    new_node_info.pdfs = this_node_info.pdfs;  // no change in this

    if (parent->yes_ == this_node) {
      parent->yes_ = no_node;
    }
    else {
      parent->no_ = no_node;
    }

    // set pointers to NULL so that destructor will not be called recursively
    p_split->yes_ = p_split->no_ = NULL;

    delete this_node;
    queue.push(std::make_pair(no_node, cur_prio_++));
    info_map[no_node] = new_node_info;
    delete yes_node;  // not inserted in the virtual tree
  }
  else if (no_keys.size() == 0) {  // similar procedure with yes_keys
    SplitEventMap *parent =
             dynamic_cast<SplitEventMap*>(this_node_info.parent);
    KALDI_ASSERT(parent != NULL);
    MultiTreeNodeInfo new_node_info;
    new_node_info.parent = parent;
    new_node_info.src_node = yes;
    new_node_info.possible_values = yes_map;
    new_node_info.pdfs = this_node_info.pdfs;  // no change in this

    if (parent->yes_ == this_node) {
      parent->yes_ = yes_node;
    }
    else {
      parent->no_ = yes_node;
    }
    delete this_node;
    queue.push(std::make_pair(yes_node, cur_prio_++));
    info_map[yes_node] = new_node_info;
    delete no_node;
  }
  else {  // both are non empty
    p_split->yes_ = yes_node;
    p_split->no_ = no_node;
    MultiTreeNodeInfo yes_info, no_info;
    yes_info.parent = no_info.parent = this_node;
    yes_info.src_node = yes;
    yes_info.possible_values = yes_map;
    no_info.src_node = no;
    no_info.possible_values = no_map;
    yes_info.pdfs = no_info.pdfs = this_node_info.pdfs;
    info_map[yes_node] = yes_info;
    info_map[no_node] = no_info;
    queue.push(std::make_pair(yes_node, cur_prio_++));
    queue.push(std::make_pair(no_node, cur_prio_++));
  }
}

EventMap* MultiTreePdfMap::GenerateVirtualTree(unordered_map<int32,
                                               vector<int32> >& mappings) {
  KALDI_LOG << "begin merging trees";
  KALDI_ASSERT(queue.size() == 0);
  KALDI_ASSERT(info_map.size() == 0);
  KALDI_ASSERT(trees_.size() == num_trees_);
  KALDI_ASSERT(mappings.empty());
  unordered_map<vector<int32>, int32, kaldi::VectorHasher<int32> > reversed_map;

  for (size_t i = 0; i < num_trees_; i++) {
    KALDI_ASSERT(trees_[i] != NULL);
  }

  const SplitEventMap* root_src =
           dynamic_cast<const SplitEventMap*>(trees_[0]);
  if (root_src == NULL) {
    KALDI_LOG << "Build virtual tree failed "
              << "this only happens with made-up data when the num-leaves == 1";
    return NULL;
  }

  SplitEventMap* root = new SplitEventMap();
  MultiTreeNodeInfo root_info;
  root_info.parent = NULL;  // the root doesn't have a parent
  root_info.src_node = trees_[0];

  set<EventValueType> all_pdfs;
  size_t max_index =
      *std::max_element(hmm_lengths_.begin(), hmm_lengths_.end());

  for (size_t i = 0; i <= max_index; i++) {
    all_pdfs.insert(i);  
  }
  root_info.possible_values[-1] = all_pdfs;

  for (size_t i = 0; i < cxt_length_; i++) {
    root_info.possible_values[i] = all_phones_;
    if (i != center_phone_) {
      root_info.possible_values[i].insert(0);
    }   // this is important!! hxu

  }

  root_info.pdfs.resize(0);
  info_map.insert(InfoMap::value_type(root, root_info));

  queue.push(std::make_pair(root, cur_prio_++));

  size_t new_leaf = 0;

  // the rule is process *this_node*,
  // for which only memory was allocated already by its parents,
  // and allocate memory for the children of *this_node*, but don't process them
  while (queue.size() > 0) {
    // i guess any element in the set would do and orders don't matter
    EventMap* this_node = queue.top().first;
    queue.pop();  // delete the node
    MultiTreeNodeInfo this_node_info = info_map[this_node];
    info_map.erase(this_node);

    // exactly one of the following pointers should be non-NULL
    ConstantEventMap* p_const = dynamic_cast<ConstantEventMap*>(this_node);
    SplitEventMap* p_split = dynamic_cast<SplitEventMap*>(this_node);

    if (p_const != NULL) {  // this_node is a leaf (ConstantEventMap)
      vector<int32> pdfs_of_leaf = this_node_info.pdfs;  
      const ConstantEventMap* src_node =
           dynamic_cast<const ConstantEventMap*>(this_node_info.src_node);
      KALDI_ASSERT(src_node != NULL);
      EventAnswerType new_pdf_number = src_node->answer_;  // the leaf number
      pdfs_of_leaf.push_back(new_pdf_number);

      //  if we got a new leaf for the virtual tree
      if (pdfs_of_leaf.size() == num_trees_) {
        KALDI_VLOG(1) << "new virtual leaf! " << new_leaf;
        if (reversed_map.find(pdfs_of_leaf) != reversed_map.end()) {
          KALDI_VLOG(1) << " duplicate keys , will merge ";
          p_const->answer_ = reversed_map[pdfs_of_leaf];
        }
        else {
          mappings.insert(unordered_map<int32, vector<int32> >::
            value_type(new_leaf, pdfs_of_leaf));
          reversed_map[pdfs_of_leaf] = new_leaf;
          // KALDI_VLOG(1) << "new virtual leaf! " << new_leaf;
          p_const->answer_ = new_leaf++;
        }
      }
      else {
        this->ConnectToNextTree(this_node_info, this_node, pdfs_of_leaf);
      }
    }
    else {  // this node is a SplitEventMap
      KALDI_ASSERT(p_split != NULL);
      const SplitEventMap* src_node =
           dynamic_cast<const SplitEventMap*>(this_node_info.src_node);
      KALDI_ASSERT(src_node != NULL);
      p_split->key_ = src_node->key_;
      p_split->yes_set_ = src_node->yes_set_;

      // now we need the type of the children to allocate
      // memory for the virtual tree
      EventMap* yes = src_node->yes_;
      EventMap* no = src_node->no_;
      EventMap *yes_node, *no_node;
      if (dynamic_cast<SplitEventMap*>(yes) != NULL) {
        yes_node = new SplitEventMap();
      }
      else {
        yes_node = new ConstantEventMap();
      }

      if (dynamic_cast<SplitEventMap*>(no) != NULL) {
        no_node = new SplitEventMap();
      }
      else {
        no_node = new ConstantEventMap();
      }
      this->ConnectToSplitNode(this_node_info, this_node, yes, no,
                               yes_node, no_node);
    }
  }
  return root;
}

}  // namespace kaldi
