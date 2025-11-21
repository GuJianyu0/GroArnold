#ifndef _KDTREE_KNN_H_
#define _KDTREE_KNN_H_

#include <vector>
#include <array>
#include <queue>
#include <limits>
#include <algorithm>
#include <cmath>

class KDTreeKNN {
public:
    explicit KDTreeKNN(const std::vector<std::vector<double>>& points);

    // Main API: vector-based
    std::vector<int> find_k_nearest(int k, const std::vector<double>& q) const;

    // Convenience: accept std::array without forcing immediate callsite rewrites
    template <size_t N>
    std::vector<int> find_k_nearest(int k, const std::array<double, N>& q) const {
        return find_k_nearest(k, std::vector<double>(q.begin(), q.end()));
    }

private:
    struct Node {
        int idx;              // index into pts_
        int left  = -1;
        int right = -1;
        int axis  = 0;
        std::vector<double> minb, maxb; // bounding box for pruning
    };

    std::vector<std::vector<double>> pts_; // local copy (stable lifetime)
    std::vector<Node> nodes_;              // implicit tree storage
    int root_ = -1;
    int dim_  = 0;

    int build_(std::vector<int>& ids, int l, int r, int depth);
    void compute_bbox_(int node_id);
    static double sqdist_(const std::vector<double>& a, const std::vector<double>& b);

    // max-heap of current bests; pair<sqdist, index>
    using PQ = std::priority_queue<std::pair<double,int>>;
    void knn_(int node_id, const std::vector<double>& q, int k, PQ& heap) const;
};

#endif