=======================
Decision Tree Internals
=======================

This page describes the internals of the phonetic decision-tree clustering code, which are implemented as quite generic structures and algorithms. For a higher-level view that explains the overall algorithm being implemented and how it is used by the toolkit, see `How decision trees are used in Kaldi <pages/api-undefined.md#tree_externals>`_.

----------
Event maps
----------

The central concept in the decision-tree building code is the event map, represented by type EventMap. Don't be misled by the term "event" to think that this is something happening at a particular time. An event is just a set of ``(key,value)`` pairs, with no key repeated. Conceptually, it could be represented by the type :cpp:`std::map<int32,int32>`. In fact, for efficiency we represent it as a sorted vector of pairs, via the typedef:

.. code-block:: cpp

   typedef std::vector<std::pair<EventKeyType,EventValueType> > EventType;

Here, :cpp:`EventKeyType` and :cpp:`EventValueType` are just typedefs to :cpp:`int32`, but the code is easier to understand if we use distinct names for them. Think of an Event (type :cpp:`EventType`) as being like a collection of variables, except that the names and values of the variables are both integers. There is also the type :cpp:`EventAnswerType`, and this is also :cpp:`int32`. It is the type returned by the :cpp:`EventMap`; in practice it is a pdf identifier (acoustic-state index). Functions that require :cpp:`EventType` expect you to have sorted it first (e.g. by calling :cpp:`std::sort(e.begin(),e.end());` )

The purpose of the :cpp:`EventMap` class is best illustrated via an example. Suppose our phone-in-context is a/b/c; suppose we have the standard 3-state topology; and suppose we want to ask what is the index of the pdf for the central state of phone "b" in this context. The state in question would be state 1, since we use zero-based indexing. We are skimming over a few details here when we refer to "state"; see `Pdf-classes <#hmm_1pdf_class>`_ for more information. Suppose the integer indices of phones a, b and c are 10, 11 and 12 respectively. The "event" would correspond to the mapping::

      (0  -> 10), (1  -> 11), (2  -> 12), (-1 -> 1)

where 0, 1 and 2 are the positions in the 3-phone window "a/b/c", and -1 is a special index we use to encode the state-id (c.f. the constant :cpp:`kPdfClass = -1`). The way this would be represented as a sorted vector of pairs is:

.. code-block:: cpp

   // caution: not valid C++.
   EventType e = { {-1,1}, {0,10}, {1,11}, {2,12} };

Suppose that the acoustic-state index (pdf-id) corresponding to this acoustic state happens to be 1000. Then if we have an :cpp:`EventMap` "emap" representing the tree, then we would expect the following assert not to fail:

.. code-block:: cpp

   EventAnswerType ans;
   bool ret = emap.Map(e, &ans); // emap is of type EventMap; e is EventType
   KALDI_ASSERT(ret == true && ans == 1000);

So when we state that an :cpp:`EventMap` is a map from :cpp:`EventType` to :cpp:`EventAnswerType`, you can think of this roughly as a map from a phone-in-context to an integer index. The phone-in-context is represented as a set of (key-value) pairs, and in principle, we could add new keys and put in more information. Notice that the :cpp:`EventMap::Map()` function returns bool. This is because it's possible for certain events not to map to any answer at all (e.g. consider an invalid phone, or imagine what happens when the :cpp:`EventType` doesn't contain all the keys that the :cpp:`EventMap` is querying).

The :cpp:`EventMap`` is a very passive object. It does not have any capability to learn decision-trees, it is just a means to store decision-trees. Think of it as a structure that represents a function from :cpp:`EventType` to :cpp:`EventAnswerType`. :cpp:`EventMap` is a polymorphic, pure-virtual class (i.e. it cannot be instantiated because it has virtual functions that are not implemented). There are three concrete classes that implement the :cpp:`EventMap` interface:


  * :cpp:`ConstantEventMap`: Think of this as a decision-tree leaf node. This class stores an integer of type EventAnswerType and its Map function always returns that value.
  * :cpp:`SplitEventMap`: Think of this as a decision-tree non-leaf node that queries a certain key and goes to the "yes" or "no" child node depending on the answer. Its Map function calls the Map function of the appropriate child node. It stores a set of integers of type :cpp:`kAnswerType` that correspond to the "yes" child (everything else goes to "no").
  * :cpp:`TableEventMap`: This does a complete split on a particular key. A typical example is: you might first split completely on the central phone and then have a separate decision tree for each value of that phone. Internally it stores a vector of EventMap* pointers. It looks up the value corresponding to the key that it is splitting on, and calls the Map function of the :cpp:`EventMap` at the corresponding position in the vector.

The EventMap cannot really do very much apart from map :cpp:`EventType` to :cpp:`EventAnswerType`. Its interface does not provide functions that allow you to traverse the tree, either up or down. There is just one function that allows you to modify the :cpp:`EventMap`, and this is the :cpp:`EventMap::Copy()` function, declared like this (as a class-member):

.. code-block:: cpp

   virtual EventMap *Copy(const std::vector<EventMap*> &new_leaves) const;

This has an effect similar to function composition. If you call :cpp:`Copy()` with an empty vector "new_leaves", it will just return a deep copy of the entire object, copying all the pointers as it goes down the tree. However, if new_leaves is nonempty, each time the Copy() function reaches a leaf, if that leaf :cpp:`l` is within the range :cpp:`(0, new_leaves.size()-1)` and :cpp:`new_leaves[l]` is non-NULL, the :cpp:`Copy()` function will return the result of calling :cpp:`new_leaves[l]->Copy()`, rather than the :cpp:`Copy()` function of the leaf itself (which would have returned a new :cpp:`ConstantEventMap`). A typical example is where you decide to split a particular leaf, say leaf 852. You could create an object of type :cpp:`vector<EventMap*>` whose only non-NULL member is as position 852. It would contain a pointer to an object whose real type is :cpp:`SplitEventMap`, and the "yes" and "no" pointers would be to type :cpp:`ConstantEventMap` with leaf values, say, 852 and 1234 (we reuse the old leaf-node id for the new leaf). The real tree-building code is not this inefficient.

Statistics for building the tree
--------------------------------
The statistics used to build the phonetic decision tree are of the following type:

.. code-block:: cpp

   typedef std::vector<std::pair<EventType, Clusterable*> > BuildTreeStatsType;

 A reference to an object of this type is passed to all the decision-tree building routines. These statistics are expected to contain no duplicates with the same EventType member, i.e. they conceptually represent a map from EventType to Clusterable. In our current code the Clusterable objects are really of type GaussClusterable, but the tree-building code doesn't know this. A program that accumulates these statistics is acc-tree-stats.cc.

Classes and functions involved in tree-building
-----------------------------------------------
Questions (config class)
************************

The class Questions is a class that in its interaction with the tree-building code, behaves a little bit like a "configuration" class. It is really a map from the "key" value (of type EventKeyType), to a configuration class of type QuestionsForKey.

Class QuestionsForKey has a set of "questions", each of which is just a set of integers of type EventValueType; these mostly correspond to sets of phones, or if the key is -1, they would in the typical case correspond to sets of HMM-state indices, i.e. subsets of {0,1,2}. QuestionsForKey also contains a configuration class of type RefineClustersOptions. This controls tree-building behaviour in that the tree-building code will attempt to iteratively move values (e.g. phones) between the two sides of the split in order to maximize likelihood (as in K-means with :cpp:`K=2`). However, this can be turned off by setting the number of iterations of "refining clusters" to zero, which corresponds to just choosing from among a fixed list of questions. This seems to work a bit better.

Lowest-level functions
**********************
See `Low-level functions for manipulating statistics and event-maps`_ for a complete list; we summarize some important ones here.

These functions mostly involve manipulations of objects of type :cpp:`BuildTreeStatsType`, which as we saw above is a vector of pairs of :cpp:`(EventType, Clusterable*)`. The simplest ones are :cpp:`DeleteBuildTreeStats()`, :cpp:`WriteBuildTreeStats()` and :cpp:`ReadBuildTreeStats()`.

The function PossibleValues() finds out which values a particular key takes within a collection of statistics (and informs the user whether or not the key was always defined); SplitStatsByKey() will split up an object of type :cpp:`BuildTreeStatsType` into a vector :cpp:`<BuildTreeStatsType>` according to the value taken by a particular key (e.g. splitting on the central phone); :cpp:`SplitStatsByMap()` does the same but the index is not the value for that key, but the answer returned by an :cpp:`EventMap` that is provided to the function.

:cpp:`SumStats()` sums up the statistics (i.e. the :cpp:`Clusterable` objects) in an object of BuildTreeStatsType and returns corresponding :cpp:`Clusterable *` object. :cpp:`SumStatsVec()` takes an object of type :cpp`std::vector<BuildTreeStatsType>` and outputs something of type :cpp:`vector<Clusterable*>`, i.e. it's like :cpp:`SumStats()` but for a vector; it's useful in processing the output of :cpp:`SplitStatsByKey()` and :cpp:`SplitStatsByMap()`.

:cpp:`ObjfGivenMap()` evaluates the objective function given some statistics and an EventMap: it sums up all the statistics within each cluster (corresponding to each distinct answer of the EventMap::Map() function), adds up the objective functions across the clusters, and returns the total.

:cpp:`FindAllKeys()` will find all the keys that are defined in a collection of statistics, and depending on the arguments it can find all keys that are defined for all the events, or all the keys that are defined for any event (i.e. take the intersection or union of sets of defined keys).

Intermediate-level functions
****************************

The next group of functions involved in tree building are listed here and correspond to various stages of tree building. We will now only mention some representative ones.

Firstly, we point out that a lot of these functions have a parameter int32 num_leaves. The integer which this points to acts as a counter to allocate new leaves. At the start of tree building the caller would set it to zero. Each time a new leaf is required, the tree-building code takes the number it currently points to as the new leaf-id, and then increments it.

One important function is :cpp:`GetStubMap()`. This function returns a tree that has not yet been split, i.e. the pdfs do not depend on the context. The input to this function controls whether all the phones are distinct or some of them share decision-tree roots, and whether all the states within a particular phone share the same decision-tree root.

The function :cpp:`SplitDecisionTree()` takes as input an EventMap that would normally correspond to a non-split "stub" decision tree of the type created by :cpp:`GetStubMap()`, and does the decision-tree splitting until either the maximum number of leaves has been reached, or the gain from splitting a leaf is less than a specified threshold.

The function :cpp:`ClusterEventMap()` takes an EventMap and a threshold, and merges the leaves of the EventMap as long as the cost of doing so is less than the threshold. This function would normally be called after :cpp:`SplitDecisionTree()`. There are other versions of this function that operate with restrictions (e.g. to avoid merging leaves from distinct phones).

Top-level tree-building functions
*********************************
The top-level tree-building functions are listed here. These functions are called directly from the command-line programs. The most important one is :cpp:`BuildTree()`. It is given a Questions configuration class, and some information about sets of phones whose roots to share and (for each set of phones) whether to share the `pdf-classes <#hmm_1pdf_class>`_ in a single decision tree root or to have a separate root for each pdf-class. It is also passed information about the lengths of the phones, plus various thresholds. It builds the tree and returns the EventMap object which is used to construct the ContextDependency object.

Another important routine in this group is :cpp:`AutomaticallyObtainQuestions()`, which used to obtain questions via automatic clustering of phones. It clusters the phones into a tree, and for each node in the tree, all leaves accessible from that node form one question (questions are equivalent to sets of phones). This (called from ``cluster-phones.cc``) makes our recipes independent of human-generated phone sets.

The function :cpp:`AccumulateTreeStats()` accumulates statistics for training the tree, given features and an alignment that is a sequence of transition-ids (see `Integer identifiers used by TransitionModel <#hmm_1transition_model_identifiers>`_\ ). This function is defined in a different directory from the other tree-building-related functions (``hmm/`` not ``tree/``) as it depends on more code (e.g. it knows about the TransitionModel class), and we preferred to keep the dependencies of the core tree-building code to a minimum.
