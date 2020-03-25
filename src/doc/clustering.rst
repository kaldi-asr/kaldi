Clustering mechanisms in Kaldi
==============================

Introduction
------------

This page explains the generic clustering mechanisms and interfaces used in Kaldi. See `Classes and functions related to clustering <pages/api-undefined.md#group__clustering__group>`_ for a list of classes and functions involved in this. This page does not cover phonetic decision-tree clustering (see `Decision tree internals <pages/api-undefined.md#tree_internals>`_ and `How decision trees are used in Kaldi <pages/api-undefined.md#tree_externals>`_\ ), although classes and functions introduced in this page are used in lower levels of the phonetic clustering code.

The Clusterable interface
-------------------------
The Clusterable class is a pure virtual class from which the class GaussClusterable inherits (GaussClusterable represents Gaussian statistics). In the future, we will add other types of clusterable objects that inherit from Clusterable. The reason for the Clusterable class is to allow us to use generic clustering algorithms.

The central notion of the Clusterable interface is that of adding statistics together and measuring the objective function. The notion of distance between two Clusterable objects is derived from measuring the objective function of the two objects separately, then adding them together and measuring the objective function; the negative of the decrease in the objective function gives the notion of distance.

Examples of Clusterable classes that we intend to add at some point include mixture-of-Gaussian statistics derived from posteriors of a fixed, shared, mixture-of-Gaussians model, and also collections of counts of discrete observations (the objective function would be equivalent to the negated entropy of the distribution, times the number of counts).

An example of getting a pointer of type ``Clusterable*`` (which is actually of the ``GaussClusterable`` type) is as follows:

.. code-block:: cpp

   Vector<BaseFloat> x_stats(10), x2_stats(10);
   BaseFloat count = 100.0, var_floor = 0.01;
   // initialize x_stats and x2_stats e.g. as
   // x_stats = 100 * mu_i, x2_stats = 100 * (mu_i*mu_i + sigma^2_i)
   Clusterable *cl = new GaussClusterable(x_stats, x2_stats, var_floor, count);

Clustering algorithms
---------------------

We have implemented a number of generic clustering algorithms. These are listed in `Algorithms for clustering <pages/api-undefined.md#group__clustering__group__algo>`_. A data structure that is used heavily in these algorithms is a vector of pointers to the Clusterable interface class:

.. code-block:: cpp

   std::vector<Clusterable*> to_be_clustered;

The index into the vector is the index of the "point" to be clustered.

K-means and algorithms with similar interfaces
----------------------------------------------

A typical example of calling the clustering code is as follows:

.. code-block:: cpp

   std::vector<Clusterable*> to_be_clustered;
   // initialize "to_be_clustered" somehow ...
   std::vector<Clusterable*> clusters;
   int32 num_clust = 10; // requesting 10 clusters
   ClusterKMeansOptions opts; // all default.
   std::vector<int32> assignments;
   ClusterKMeans(to_be_clustered, num_clust, &clusters, &assignments, opts);

After the clustering code is called, "assignments" will tell you for each item in ``"to_be_clustered"``, which cluster it is assigned to. The ``ClusterKMeans()`` algorithm is fairly efficient even for a large number of points; click the function name for more details.

There are two more algorithms that have a similar interface to ``ClusterKMeans()``: namely, ``ClusterBottomUp()`` and ``ClusterTopDown()``. Probably the more useful one is ``ClusterTopDown()``, which should be more efficient than ``ClusterKMeans()`` if the number of clusters is large (it does a binary split, and then does a binary split on the leaves, and so on). Internally it calls ``TreeCluster()``, see below.

Tree clustering algorithm
-------------------------

The function ``TreeCluster()`` clusters points into a binary tree (the leaves won't necessarily have just one point each, you can specify a maximum number of leaves). This function is useful, for instance, when building regression trees for adaptation. See that function's documentation for a detailed explanation of its output format. The quick overview is that it numbers leaf and non-leaf nodes in topological order with the leaves first and the root last, and outputs a vector that tells you for each node what its parent is.
