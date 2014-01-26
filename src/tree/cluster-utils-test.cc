// tree/cluster-utils-test.cc

// Copyright 2009-2011  Microsoft Corporation

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

#include "tree/cluster-utils.h"
#include "tree/clusterable-classes.h"
#include "util/stl-utils.h"

namespace kaldi {
static void TestClusterUtils() {  // just some very basic tests of the GaussClusterable class.
  BaseFloat varFloor = 0.1;
  size_t dim = 1 + rand() % 10;
  size_t nGauss = 1 + rand() % 10;
  std::vector< GaussClusterable * > v(nGauss);
  for (size_t i = 0;i < nGauss;i++) {
    v[i] = new GaussClusterable(dim, varFloor);
  }
  for (size_t i = 0;i < nGauss;i++) {
    size_t nPoints = 1 + rand() % 30;
    for (size_t j = 0;j < nPoints;j++) {
      BaseFloat post = 0.5 *(rand()%3);
      Vector<BaseFloat> vec(dim);
      for (size_t k = 0;k < dim;k++) vec(k) = RandGauss();
      v[i]->AddStats(vec, post);
    }
  }
  for (size_t i = 0;i+1 < nGauss;i++) {
    BaseFloat like_before = (v[i]->Objf() + v[i+1]->Objf()) / (v[i]->Normalizer() + v[i+1]->Normalizer());
    Clusterable *tmp = v[i]->Copy();
    tmp->Add( *(v[i+1]));
    BaseFloat like_after = tmp->Objf() / tmp->Normalizer();
    std::cout << "Like_before = " << like_before <<", after = "<<like_after <<" over "<<tmp->Normalizer()<<" frames.\n";
    if (tmp->Normalizer() > 0.1)
      KALDI_ASSERT(like_after <= like_before);  // should get worse after combining stats.
    delete tmp;
  }
  for (size_t i = 0;i < nGauss;i++)
    delete v[i];
}


static void TestObjfPlus() {
  ScalarClusterable a(1.0), b(2.5);
  AssertEqual(a.Objf(), (BaseFloat)0.0);
  AssertEqual(b.Objf(), (BaseFloat)0.0);
  AssertEqual( a.ObjfPlus(b), -0.5 * (1.0-2.5)*(1.0-2.5));  // 0.5 because half-distance, squared = 1/4, times two points...
  std::cout << "Non-binary Output: "<<'\n';
  a.Write(std::cout, false);
  std::cout << "Binary Output: "<<'\n';
  a.Write(std::cout, true);
}

static void TestObjfMinus() {
  ScalarClusterable a(1.0), b(2.5);
  AssertEqual(a.Objf(), 0.0);
  AssertEqual(b.Objf(), 0.0);
  a.Add(b);
  AssertEqual(a.ObjfMinus(b), 0.0);
  a.Add(b);
  AssertEqual(a.ObjfMinus(b), -0.5 * (1.0-2.5)*(1.0-2.5));
}

static void TestDistance() {
  ScalarClusterable a(1.0), b(2.5);
  AssertEqual(a.Objf(), 0.0);
  AssertEqual(b.Objf(), 0.0);
  AssertEqual(a.ObjfPlus(b), -a.Distance(b));  // since distance is negated objf-change, and original objfs were zero.
} // end namespace kaldi


static void TestSumObjfAndSumNormalizer() {
  ScalarClusterable a(1.0), b(2.5);
  AssertEqual(a.Objf(), 0.0);
  AssertEqual(b.Objf(), 0.0);
  a.Add(b);
  std::vector<Clusterable*> vec;
  vec.push_back(&a);
  vec.push_back(&a);
  AssertEqual(SumClusterableObjf(vec), 2*vec[0]->Objf());
  AssertEqual(SumClusterableNormalizer(vec), 2*vec[0]->Normalizer());
}

static void TestSum() {
  ScalarClusterable a(1.0), b(2.5);
  std::vector<Clusterable*> vec;
  vec.push_back(&a);
  vec.push_back(&b);
  Clusterable *sum = SumClusterable(vec);
  AssertEqual(a.ObjfPlus(b),  sum->Objf());
  delete sum;
}

static void TestEnsureClusterableVectorNotNull() {
  ScalarClusterable a(1.0), b(2.5);
  std::vector<Clusterable*> vec(4);
  vec[1] = a.Copy(); vec[3] = a.Copy();
  EnsureClusterableVectorNotNull(&vec);
  KALDI_ASSERT(vec[0] != NULL && vec[2] != NULL && vec[0]->Objf() == 0 && vec[2]->Objf() == 0 && vec[0] != vec[2] && vec[0] != vec[1]);
  DeletePointers(&vec);
}

static void TestAddToClusters() {
  ScalarClusterable a(1.0), b(2.5), c(3.0);
  std::vector<Clusterable*> stats(3);
  stats[0] = a.Copy(); stats[1] = b.Copy(); stats[2] = c.Copy();
  std::vector<int32> assignments(3);
  assignments[0] = 1; assignments[1] = 1; assignments[2] = 4;
  std::vector<Clusterable*> clusters;
  std::vector<Clusterable*> clusters2;
  AddToClusters(stats, assignments, &clusters);

  AddToClusters(stats, assignments, &clusters2);  // do this twice.
  AddToClusters(stats, assignments, &clusters2);

  KALDI_ASSERT(clusters.size() == 5);
  KALDI_ASSERT(clusters[0] == NULL && clusters[1] != NULL && clusters[4] != NULL);
  for (size_t i = 0;i < 5;i++) {
    if (clusters[i] != NULL) {
      AssertEqual(clusters2[i]->Objf(), clusters[i]->Objf()*2);
    }
  }
  AssertEqual(c.Mean(), ((ScalarClusterable*)clusters[4])->Mean());
  AssertEqual( ((ScalarClusterable*)clusters[1])->Mean(), 0.5*(1.0+2.5));
  DeletePointers(&stats);
  DeletePointers(&clusters);
  DeletePointers(&clusters2);
}

static void TestAddToClustersOptimized() {
  for (size_t p = 0;p < 100;p++) {
    size_t n_stats = rand() % 5;
    n_stats = n_stats * n_stats;  // more interestingly distributed.
    std::vector<Clusterable*> stats(n_stats);
    for (size_t i = 0;i < n_stats;i++) {
      if (rand() % 5 < 4) {
        ScalarClusterable *ptr = new ScalarClusterable(RandGauss());
        if (rand() % 2 == 0) ptr->Add(*ptr);  // make count equal 2.  for more randomness.
        stats[i] = ptr;
      }  else stats[i] = NULL;  // make some zero. supposed to be robust to this.
    }
    size_t n_clust = 1 + rand() % 4;
    std::vector<int32> assignments(n_stats);
    for (size_t i = 0;i < assignments.size();i++)
      assignments[i] = rand() % n_clust;
    std::vector<Clusterable*> clusts1;
    std::vector<Clusterable*> clusts2;
    Clusterable *total = SumClusterable(stats);
    if (total == NULL) {  // no stats were non-NULL.
      KALDI_ASSERT(stats.size() == 0 || stats[0] == NULL);
      DeletePointers(&stats);
      continue;
    }
    AddToClusters(stats, assignments, &clusts1);
    AddToClustersOptimized(stats, assignments, *total, &clusts2);

    BaseFloat tot1 = SumClusterableNormalizer(stats),
        tot2 = SumClusterableNormalizer(clusts1),
        tot3 = SumClusterableNormalizer(clusts2);
    AssertEqual(tot1, tot2);
    AssertEqual(tot1, tot3);
    KALDI_ASSERT(clusts1.size() == clusts2.size());
    for (size_t i = 0;i < clusts1.size();i++) {
      if (clusts1[i] != NULL || clusts2[i] != NULL) {
        KALDI_ASSERT(clusts1[i] != NULL && clusts2[i] != NULL);
        AssertEqual(clusts1[i]->Normalizer(), clusts2[i]->Normalizer());
        AssertEqual( ((ScalarClusterable*)clusts1[i])->Mean(),
                     ((ScalarClusterable*)clusts2[i])->Mean() );
      }
    }
    delete total;
    DeletePointers(&clusts1);
    DeletePointers(&clusts2);
    DeletePointers(&stats);
  }
}


static void TestClusterBottomUp() {
  for (size_t i = 0;i < 10;i++) {
    size_t n_clust = rand() % 10;
    std::vector<Clusterable*> points;
    for (size_t j = 0;j < n_clust;j++) {
      size_t n_points = 1 + rand() % 5;
      BaseFloat clust_center  = (BaseFloat)j;
      for (size_t k = 0;k < n_points;k++) points.push_back(new ScalarClusterable(clust_center + RandUniform()*0.01));
    }

    BaseFloat max_merge_thresh = 0.1;
    size_t min_clust = rand() % 10;  // use max_merge_thresh to control #clust.
    std::vector<Clusterable*> clusters;
    std::vector<int32> assignments;

    for (size_t i = 0;i < points.size();i++) {
      size_t j = rand() % points.size();
      if (i != j) std::swap(points[i], points[j]);  // randomize order.
    }


    float ans = ClusterBottomUp(points, max_merge_thresh, min_clust, &clusters, &assignments);

    KALDI_ASSERT(ans < 0.000001);  //  objf change should be negative.
    std::cout << "Objf change from bottom-up clustering is "<<ans<<'\n';

    ClusterBottomUp(points, max_merge_thresh, min_clust, NULL, NULL);  // make sure no crash.

    if (0) {  // for debug if it breaks.
      for (size_t i = 0;i < points.size();i++) {
        std::cout << "point " << i << ": " << ((ScalarClusterable*)points[i])->Info() << " -> " << assignments[i] << "\n";
      }
      for (size_t i = 0;i < clusters.size();i++) {
        std::cout << "clust " << i << ": " << ((ScalarClusterable*)clusters[i])->Info();
      }
    }

    KALDI_ASSERT(clusters.size() == std::max(n_clust, std::min(points.size(), min_clust)));

    for (size_t i = 0;i < points.size();i++) {
      size_t j = rand() % points.size();
      BaseFloat xi = ((ScalarClusterable*)points[i])->Mean(),
          xj = ((ScalarClusterable*)points[j])->Mean();
      if (fabs(xi-xj) < 0.011) {
        if (clusters.size() == n_clust) KALDI_ASSERT(assignments[i] == assignments[j]);
      } else KALDI_ASSERT(assignments[i] != assignments[j]);
    }
    DeletePointers(&clusters);
    DeletePointers(&points);
  }

}


static void TestRefineClusters() {
  for (size_t n = 0;n < 4;n++) {
    // Test it by creating a random clustering and verifying that it does not make it worse, and
    // if done with the optimal parameters, makes it optimal.
    size_t n_clust = rand() % 10;
    std::vector<Clusterable*> points;
    for (size_t j = 0;j < n_clust;j++) {
      size_t n_points = 1 + rand() % 5;
      BaseFloat clust_center  = (BaseFloat)j;
      for (size_t k = 0;k < n_points;k++) points.push_back(new ScalarClusterable(clust_center + RandUniform()*0.01));
    }
    std::vector<Clusterable*> clusters(n_clust);
    std::vector<int32> assignments(points.size());
    for (size_t i = 0;i < clusters.size();i++) clusters[i] = new ScalarClusterable();
    // assign each point to a random cluster.
    for (size_t i = 0;i < points.size();i++) {
      assignments[i] = rand() % n_clust;
      clusters[assignments[i]]->Add(*(points[i]));
    }
    BaseFloat points_objf = SumClusterableObjf(points),
        clust_objf_before = SumClusterableObjf(clusters),
        clust_objf_after;
    AssertGeq(points_objf, clust_objf_before);

    RefineClustersOptions cfg;
    cfg.num_iters = 10000;  // very large.
    cfg.top_n = 2 + (rand() % 20);
    BaseFloat impr = RefineClusters(points, &clusters, &assignments, cfg);

    clust_objf_after = SumClusterableObjf(clusters);
    std::cout << "TestRefineClusters: objfs are: "<<points_objf<<" "<<clust_objf_before<<" "<<clust_objf_after<<", impr = "<<impr<<'\n';
    if (cfg.top_n >=(int32) n_clust) {  // check exact.
      KALDI_ASSERT(clust_objf_after <= 0.01*points.size());
    }
    AssertEqual(clust_objf_after - clust_objf_before, impr);
    DeletePointers(&clusters);
    DeletePointers(&points);

  }
}

static void TestClusterKMeans() {
  size_t n_points_tot = 0, n_wrong_tot = 0;
  for (size_t n = 0;n < 3;n++) {
    // Test it by creating a random clustering and verifying that it does not make it worse, and
    // if done with the optimal parameters, makes it optimal.
    size_t n_clust = rand() % 10;
    std::vector<Clusterable*> points;
    for (size_t j = 0;j < n_clust;j++) {
      size_t n_points = 1 + rand() % 5;
      BaseFloat clust_center  = (BaseFloat)j;
      for (size_t k = 0;k < n_points;k++) points.push_back(new ScalarClusterable(clust_center + RandUniform()*0.01));
    }
    std::vector<Clusterable*> clusters;
    std::vector<int32> assignments;
    ClusterKMeansOptions kcfg;

    BaseFloat ans = ClusterKMeans(points, n_clust, &clusters, &assignments, kcfg);

    if (n < 3) ClusterKMeans(points, n_clust, NULL, NULL, kcfg);  // make sure no crash.

    BaseFloat clust_objf = SumClusterableObjf(clusters);

    std::cout << "TestClusterKmeans: objf after clustering is: "<<clust_objf<<", impr is: "<<ans<<'\n';

    if (clusters.size() != n_clust) {
      std::cout << "Warning: unexpected number of clusters "<<clusters.size()<<" vs. "<<n_clust<<"\n";
    }
    KALDI_ASSERT(assignments.size() == points.size());

    if (clust_objf < -1.0 * points.size()) {  // a bit high...
      std::cout << "Warning: ClusterKMeans did not work quite as well as expected\n";
    }


    int32 num_wrong = 0;
    for (size_t i = 0;i < points.size();i++) {
      size_t j = rand() % points.size();
      BaseFloat xi = ((ScalarClusterable*)points[i])->Mean(),
          xj = ((ScalarClusterable*)points[j])->Mean();
      if (fabs(xi-xj) < 0.011) {
        if (assignments[i] != assignments[j]) num_wrong++;
      } else
        if (assignments[i] == assignments[j]) num_wrong++;
    }
    std::cout << "num_wrong = "<<num_wrong<<'\n';

    n_points_tot += points.size();
    n_wrong_tot += num_wrong;

    DeletePointers(&clusters);
    DeletePointers(&points);
  }
  if (n_wrong_tot*4 > n_points_tot) {
    std::cout << "Got too many wrong in k-means test [may not be fatal, but check it out.\n";
    KALDI_ASSERT(0);
  }
}


static void TestTreeCluster() {
  size_t n_points_tot = 0, n_wrong_tot = 0;
  for (size_t n = 0;n < 10;n++) {

    int32 n_clust = rand() % 10;
    std::vector<Clusterable*> points;
    for (int32 j = 0;j < n_clust;j++) {
      int32 n_points = 1 + rand() % 5;
      BaseFloat clust_center  = (BaseFloat)j;
      for (int32 k = 0;k < n_points;k++) points.push_back(new ScalarClusterable(clust_center + RandUniform()*0.01));
    }
    std::vector<Clusterable*> clusters_ext;
    std::vector<int32> assignments;
    std::vector<int32> clust_assignments;
    TreeClusterOptions tcfg;
    tcfg.thresh = 0.01;  // should prevent us splitting things in same  bucket.
    int32 num_leaves = 0;
    BaseFloat ans = TreeCluster(points, n_clust, &clusters_ext, &assignments, &clust_assignments, &num_leaves, tcfg);

    if (n < 3) TreeCluster(points, n_clust, NULL, NULL, NULL, NULL, tcfg);  // make sure no crash

    KALDI_ASSERT(num_leaves == n_clust);
    KALDI_ASSERT(clusters_ext.size() >= static_cast<size_t>(n_clust));
    std::vector<Clusterable*> clusters(clusters_ext);
    clusters.resize(n_clust);  // ignore non-leaves.
    BaseFloat clust_objf = SumClusterableObjf(clusters);

    std::cout << "TreeCluster: objf after clustering is: "<<clust_objf<<", impr is: "<<ans<<'\n';

    if (n < 2) // avoid generating too much output.
      std::cout << "Num nodes is "<<clusters_ext.size() <<", leaves "<<num_leaves;
    for (int32 i = 0;i<static_cast<int32>(clusters_ext.size());i++) {
      if (n < 2) // avoid generating too much output.
        std::cout << "Cluster "<<i<<": "<<((ScalarClusterable*)clusters_ext[i])->Info()<<", parent is: "<< clust_assignments[i]<<"\n";
      KALDI_ASSERT(clust_assignments[i]>i || (i+1 == static_cast<int32>(clusters_ext.size()) && clust_assignments[i] == i));
      if (i == static_cast<int32>(clusters_ext.size())-1)
        KALDI_ASSERT(clust_assignments[i] == i);  // top node.
    }
    DeletePointers(&clusters_ext);
    DeletePointers(&points);
  }
  if (n_wrong_tot*4 > n_points_tot) {
    std::cout << "Got too many wrong in k-means test [may not be fatal, but check it out.\n";
    KALDI_ASSERT(0);
  }
}


static void TestClusterTopDown() {
  size_t n_points_tot = 0, n_wrong_tot = 0;
  for (size_t n = 0;n < 10;n++) {

    size_t n_clust = rand() % 10;
    std::vector<Clusterable*> points;
    for (size_t j = 0;j < n_clust;j++) {
      size_t n_points = 1 + rand() % 5;
      BaseFloat clust_center  = (BaseFloat)j;
      for (size_t k = 0;k < n_points;k++) points.push_back(new ScalarClusterable(clust_center + RandUniform()*0.01));
    }
    std::vector<Clusterable*> clusters;
    std::vector<int32> assignments;
    TreeClusterOptions tcfg;
    tcfg.thresh = 0.01;  // should prevent us splitting things in same  bucket.


    BaseFloat ans = ClusterTopDown(points, n_clust, &clusters, &assignments, tcfg);

    if (n < 3)  ClusterTopDown(points, n_clust, NULL, NULL, tcfg);  // make sure doesn't crash.

    BaseFloat clust_objf = SumClusterableObjf(clusters);

    std::cout << "ClusterTopDown: objf after clustering is: "<<clust_objf<<", impr is: "<<ans<<'\n';

    if (n<=2) // avoid generating too much output.
      std::cout << "Num nodes is "<<clusters.size()<<'\n';
    for (size_t i = 0;i < clusters.size();i++) {
      if (n<=2) {  // avoid generating too much output.
        size_t old_prec = std::cout.precision();
        std::cout.precision(10);
        std::cout << "Cluster "<<i<<": "<<((ScalarClusterable*)clusters[i])->Info()<<", objf is: "<<clusters[i]->Objf()<<"\n";
        std::cout.precision(old_prec);
      }
    }
    KALDI_ASSERT(clusters.size() == n_clust);
    DeletePointers(&clusters);
    DeletePointers(&points);
  }
  if (n_wrong_tot*4 > n_points_tot) {
    std::cout << "Got too many wrong in k-means test [may not be fatal, but check it out.\n";
    KALDI_ASSERT(0);
  }
}



} // end namespace kaldi

int main() {
  using namespace kaldi;
  TestAddToClustersOptimized();
  TestObjfPlus();
  TestObjfMinus();
  TestDistance();
  TestSumObjfAndSumNormalizer();
  TestSum();
  TestEnsureClusterableVectorNotNull();
  TestAddToClusters();
  TestClusterTopDown();
  TestTreeCluster();
  TestClusterKMeans();
  TestClusterBottomUp();
  TestRefineClusters();

  for (size_t i = 0;i < 2;i++)
    TestClusterUtils();
}


