#include "kdtree_knn.h"

KDTreeKNN::KDTreeKNN(const std::vector<std::vector<double>>& points)
: pts_(points) {
    if (pts_.empty()) { root_ = -1; dim_ = 0; return; }
    dim_ = (int)pts_[0].size();
    std::vector<int> ids(pts_.size());
    for (int i=0;i<(int)ids.size();++i) ids[i]=i;
    nodes_.reserve(pts_.size());
    root_ = build_(ids, 0, (int)ids.size(), 0);
    if (root_ != -1) compute_bbox_(root_);
}

int KDTreeKNN::build_(std::vector<int>& ids, int l, int r, int depth) {
    if (l>=r) return -1;
    int m = (l+r)/2;
    int axis = depth % dim_;
    std::nth_element(ids.begin()+l, ids.begin()+m, ids.begin()+r,
                    [&](int a, int b){ return pts_[a][axis] < pts_[b][axis]; });
    int node_id = (int)nodes_.size();
    nodes_.push_back(Node{ids[m], -1, -1, axis, {}, {}});
    nodes_[node_id].left  = build_(ids, l, m, depth+1);
    nodes_[node_id].right = build_(ids, m+1, r, depth+1);
    return node_id;
}

void KDTreeKNN::compute_bbox_(int node_id) {
    auto& nd = nodes_[node_id];
    nd.minb.assign(dim_,  std::numeric_limits<double>::infinity());
    nd.maxb.assign(dim_, -std::numeric_limits<double>::infinity());

    auto relax = [&](const std::vector<double>& p){
        for (int d=0; d<dim_; ++d){
        nd.minb[d] = std::min(nd.minb[d], p[d]);
        nd.maxb[d] = std::max(nd.maxb[d], p[d]);
        }
    };
    relax(pts_[nd.idx]);

    if (nd.left != -1){
        compute_bbox_(nd.left);
        for (int d=0; d<dim_; ++d){
        nd.minb[d] = std::min(nd.minb[d], nodes_[nd.left].minb[d]);
        nd.maxb[d] = std::max(nd.maxb[d], nodes_[nd.left].maxb[d]);
        }
    }
    if (nd.right != -1){
        compute_bbox_(nd.right);
        for (int d=0; d<dim_; ++d){
        nd.minb[d] = std::min(nd.minb[d], nodes_[nd.right].minb[d]);
        nd.maxb[d] = std::max(nd.maxb[d], nodes_[nd.right].maxb[d]);
        }
    }
}

double KDTreeKNN::sqdist_(const std::vector<double>& a, const std::vector<double>& b){
    double s=0; for (int i=0;i<(int)a.size();++i){ double d=a[i]-b[i]; s+=d*d; } return s;
}

std::vector<int> KDTreeKNN::find_k_nearest(int k, const std::vector<double>& q) const {
    if (root_==-1 || k<=0) return {};
    PQ heap;
    knn_(root_, q, k, heap);
    std::vector<int> out(heap.size());
    for (int i=(int)heap.size()-1; i>=0; --i){ out[i]=heap.top().second; heap.pop(); }
    return out;
}

void KDTreeKNN::knn_(int node_id, const std::vector<double>& q, int k, PQ& heap) const {
    if (node_id==-1) return;
    const auto& nd = nodes_[node_id];
    double d2 = sqdist_(q, pts_[nd.idx]);

    if ((int)heap.size()<k) heap.emplace(d2, nd.idx);
    else if (d2 < heap.top().first){ heap.pop(); heap.emplace(d2, nd.idx); }

    // Decide traversal order by split plane
    int axis = nd.axis;
    int near_id  = (q[axis] <= pts_[nd.idx][axis]) ? nd.left  : nd.right;
    int far_id   = (q[axis] <= pts_[nd.idx][axis]) ? nd.right : nd.left;

    knn_(near_id, q, k, heap);

    // Prune by distance to split plane / bounding box:
    double plane_d2 = (q[axis]-pts_[nd.idx][axis])*(q[axis]-pts_[nd.idx][axis]);
    double worst = (heap.size()< (size_t)k) ? std::numeric_limits<double>::infinity() : heap.top().first;

    if (plane_d2 <= worst){
        // Optional: bounding-box pruning (cheap conservative test)
        // Estimate min sqdist from q to far child's bbox:
        if (far_id!=-1){
        double minb=0;
        for (int d=0; d<dim_; ++d){
            double v=q[d];
            if (v < nodes_[far_id].minb[d]){ double dd=nodes_[far_id].minb[d]-v; minb+=dd*dd; }
            else if (v > nodes_[far_id].maxb[d]){ double dd=v-nodes_[far_id].maxb[d]; minb+=dd*dd; }
        }
        if (minb <= worst) knn_(far_id, q, k, heap);
        }
    }
}
