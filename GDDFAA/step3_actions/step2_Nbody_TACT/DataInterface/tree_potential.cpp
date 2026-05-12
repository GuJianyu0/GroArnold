// tree_potential.cpp
#include "tree_potential.h"

#include <algorithm>
#include <cstring>

namespace treegrav {

OctreePotential::OctreePotential(double G, double theta, int bucket_size, int max_depth)
    : G_(G), theta_(theta), theta2_(theta*theta), bucket_size_(bucket_size), max_depth_(max_depth)
{
    if (bucket_size_ < 1) bucket_size_ = 1;
    if (max_depth_ < 1) max_depth_ = 1;
}

void OctreePotential::clear()
{
    x_.clear(); y_.clear(); z_.clear(); m_.clear(); eps_.clear();
    perm_.clear(); perm_tmp_.clear(); perm_inv_.clear();
    nodes_.clear();
    n_ = 0;
    first_index_ = 0;
    particles_reordered_ = false;
}

void OctreePotential::set_theta(double theta)
{
    theta_ = theta;
    theta2_ = theta * theta;
}

int OctreePotential::internal_index_from_original(int original_id) const
{
    if (original_id < first_index_ || original_id >= first_index_ + n_) return -1;
    const int old0 = original_id - first_index_;
    if (particles_reordered_) {
        return perm_inv_.empty() ? -1 : perm_inv_[old0];
    }
    return old0;
}

void OctreePotential::build(const Particle* p, int n, int first_index, bool reorder_particles)
{
    clear();
    if (!p || n <= 0) return;

    n_ = n;
    first_index_ = first_index;

    x_.resize(n_);
    y_.resize(n_);
    z_.resize(n_);
    m_.resize(n_);
    eps_.resize(n_);

    // Read from p[first_index .. first_index+n-1]
    for (int i = 0; i < n_; ++i) {
        const Particle& q = p[first_index_ + i];
        x_[i] = q.x;
        y_[i] = q.y;
        z_[i] = q.z;
        m_[i] = q.m;
        eps_[i] = q.eps;
    }

    build_from_arrays_(n_, first_index_, reorder_particles);
}

void OctreePotential::build(const double* x, const double* y, const double* z,
                            const double* m, const double* eps,
                            int n, int first_index, bool reorder_particles)
{
    clear();
    if (!x || !y || !z || !m || !eps || n <= 0) return;

    n_ = n;
    first_index_ = first_index;

    x_.resize(n_);
    y_.resize(n_);
    z_.resize(n_);
    m_.resize(n_);
    eps_.resize(n_);

    for (int i = 0; i < n_; ++i) {
        const int j = first_index_ + i;
        x_[i] = x[j];
        y_[i] = y[j];
        z_[i] = z[j];
        m_[i] = m[j];
        eps_[i] = eps[j];
    }

    build_from_arrays_(n_, first_index_, reorder_particles);
}

void OctreePotential::build_from_arrays_(int n, int /*first_index*/, bool reorder_particles)
{
    if (n <= 0) return;

    // Bounding cube
    double xmin = std::numeric_limits<double>::infinity();
    double ymin = std::numeric_limits<double>::infinity();
    double zmin = std::numeric_limits<double>::infinity();
    double xmax = -std::numeric_limits<double>::infinity();
    double ymax = -std::numeric_limits<double>::infinity();
    double zmax = -std::numeric_limits<double>::infinity();

    for (int i = 0; i < n; ++i) {
        xmin = std::min(xmin, x_[i]); xmax = std::max(xmax, x_[i]);
        ymin = std::min(ymin, y_[i]); ymax = std::max(ymax, y_[i]);
        zmin = std::min(zmin, z_[i]); zmax = std::max(zmax, z_[i]);
    }

    const double cx = 0.5 * (xmin + xmax);
    const double cy = 0.5 * (ymin + ymax);
    const double cz = 0.5 * (zmin + zmax);

    const double sx = xmax - xmin;
    const double sy = ymax - ymin;
    const double sz = zmax - zmin;
    double span = std::max(sx, std::max(sy, sz));
    if (!(span > 0.0)) span = 1.0;
    double hs = 0.5 * span;
    hs *= 1.00001; // enlarge a bit to keep boundary particles inside

    // Prepare permutation arrays for in-place 8-way partitioning
    perm_.resize(n);
    perm_tmp_.resize(n);
    for (int i = 0; i < n; ++i) perm_[i] = i;

    // Reserve nodes: rough estimate (safe upper bound-ish)
    const std::size_t est_nodes = std::max<std::size_t>(64, (std::size_t)(2.0 * n / std::max(1, bucket_size_)) + 64);
    nodes_.clear();
    nodes_.reserve(est_nodes);

    Node root;
    root.cx = cx; root.cy = cy; root.cz = cz;
    root.hs = hs;
    root.start = 0;
    root.end = n;
    root.is_leaf = false;
    for (int k = 0; k < 8; ++k) root.child[k] = -1;
    nodes_.push_back(root);

    build_node_(0, 0);

    if (reorder_particles) {
        particles_reordered_ = true;

        // perm_[new] = old
        perm_inv_.assign(n, 0);
        for (int newi = 0; newi < n; ++newi) {
            perm_inv_[perm_[newi]] = newi;
        }

        // reorder particle arrays into perm-order so leaves touch contiguous memory
        std::vector<double> x2(n), y2(n), z2(n), m2(n), e2(n);
        for (int newi = 0; newi < n; ++newi) {
            const int old = perm_[newi];
            x2[newi] = x_[old];
            y2[newi] = y_[old];
            z2[newi] = z_[old];
            m2[newi] = m_[old];
            e2[newi] = eps_[old];
        }
        x_.swap(x2); y_.swap(y2); z_.swap(z2); m_.swap(m2); eps_.swap(e2);

        // after reorder, leaf ranges [start,end) directly index arrays; perm_ not needed
        perm_.clear();
        perm_tmp_.clear();
        perm_.shrink_to_fit();
        perm_tmp_.shrink_to_fit();
    } else {
        particles_reordered_ = false;
        perm_inv_.clear();
        perm_inv_.shrink_to_fit();
    }
}

void OctreePotential::build_node_(int node_id, int depth)
{
    // Node& node = nodes_[node_id];
    // const int start = node.start;
    // const int end   = node.end;
    // IMPORTANT:
    // Do NOT keep a reference (Node& node = nodes_[node_id]) across nodes_.push_back(),
    // because push_back may reallocate and invalidate references/pointers -> heap corruption -> segfault.
    const int start = nodes_[node_id].start;
    const int end   = nodes_[node_id].end;
    const int count = end - start;

    // Compute monopole properties (mass, COM) and eps_max
    double M = 0.0;
    double Mx = 0.0, My = 0.0, Mz = 0.0;
    double emax = 0.0;

    for (int k = start; k < end; ++k) {
        const int i = perm_[k];
        const double mi = m_[i];
        M  += mi;
        Mx += mi * x_[i];
        My += mi * y_[i];
        Mz += mi * z_[i];
        if (eps_[i] > emax) emax = eps_[i];
    }

    nodes_[node_id].mass = M;
    if (M > 0.0) {
        const double invM = 1.0 / M;
        nodes_[node_id].comx = Mx * invM;
        nodes_[node_id].comy = My * invM;
        nodes_[node_id].comz = Mz * invM;
    } else {
        nodes_[node_id].comx = nodes_[node_id].cx;
        nodes_[node_id].comy = nodes_[node_id].cy;
        nodes_[node_id].comz = nodes_[node_id].cz;
    }
    nodes_[node_id].eps_max = emax;

    // Leaf stop conditions
    if (count <= bucket_size_ || depth >= max_depth_ || !(nodes_[node_id].hs > 0.0)) {
        nodes_[node_id].is_leaf = true;
        for (int c = 0; c < 8; ++c) nodes_[node_id].child[c] = -1;
        return;
    }

    // Cache parent geometry locally (safe even if nodes_ reallocates later)
    const double pcx = nodes_[node_id].cx;
    const double pcy = nodes_[node_id].cy;
    const double pcz = nodes_[node_id].cz;
    const double phs = nodes_[node_id].hs;

    // 8-way partition into octants around node center
    int cnt[8] = {0,0,0,0,0,0,0,0};

    for (int k = start; k < end; ++k) {
        const int i = perm_[k];
        const int ox = (x_[i] > pcx) ? 1 : 0;
        const int oy = (y_[i] > pcy) ? 1 : 0;
        const int oz = (z_[i] > pcz) ? 1 : 0;
        const int oct = (ox) | (oy << 1) | (oz << 2);
        cnt[oct]++;
    }

    int off[8];
    off[0] = start;
    for (int o = 1; o < 8; ++o) off[o] = off[o-1] + cnt[o-1];

    int cur[8];
    for (int o = 0; o < 8; ++o) cur[o] = off[o];

    for (int k = start; k < end; ++k) {
        const int i = perm_[k];
        const int ox = (x_[i] > pcx) ? 1 : 0;
        const int oy = (y_[i] > pcy) ? 1 : 0;
        const int oz = (z_[i] > pcz) ? 1 : 0;
        const int oct = (ox) | (oy << 1) | (oz << 2);
        perm_tmp_[cur[oct]++] = i;
    }

    // copy back
    for (int k = start; k < end; ++k) perm_[k] = perm_tmp_[k];

    // create children
    const double child_hs = 0.5 * phs;
    int child_idx[8];
    for (int o = 0; o < 8; ++o) child_idx[o] = -1;

    // Push children WITHOUT touching nodes_[node_id] (may reallocate!)
    for (int o = 0; o < 8; ++o) {
        if (cnt[o] <= 0) continue;
        const int cid = (int)nodes_.size();
        child_idx[o] = cid;

        Node child;
        child.hs = child_hs;
        child.cx = pcx + ((o & 1) ? child_hs : -child_hs);
        child.cy = pcy + ((o & 2) ? child_hs : -child_hs);
        child.cz = pcz + ((o & 4) ? child_hs : -child_hs);
        child.start = off[o];
        child.end   = off[o] + cnt[o];
        child.is_leaf = false;
        for (int k = 0; k < 8; ++k) child.child[k] = -1;
        nodes_.push_back(child);
    }

    // Now safe to write parent child pointers (re-acquire by index)
    nodes_[node_id].is_leaf = false;
    for (int o = 0; o < 8; ++o) nodes_[node_id].child[o] = child_idx[o];

    // recurse using local indices
    for (int o = 0; o < 8; ++o) {
        const int cid = child_idx[o];
        if (cid >= 0) build_node_(cid, depth + 1);
    }
}

} // namespace treegrav
